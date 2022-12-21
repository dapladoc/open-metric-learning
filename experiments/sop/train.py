import os
from pathlib import Path
from pprint import pprint

import hydra
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from torch.utils.data import DataLoader

from experiments.sop.datasets import DatasetQueryGallery, DatasetWithLabels
from experiments.sop.losses import TripletLoss
from experiments.sop.metrics import EmbeddingMetrics
from experiments.sop.miners import HardTripletsMiner
from experiments.sop.models import RetrievalModule, SiamNet
from oml.const import PROJECT_ROOT, TCfg
from oml.lightning.callbacks.metric import MetricValCallback
from oml.lightning.entrypoints.parser import check_is_config_for_ddp, parse_engine_params_from_config
from oml.registry.miners import MINERS_REGISTRY, get_miner, get_miner_by_cfg
from oml.registry.optimizers import get_optimizer_by_cfg
from oml.registry.samplers import get_sampler_by_cfg
from oml.registry.schedulers import get_scheduler_by_cfg
from oml.utils.misc import dictconfig_to_dict, flatten_dict, load_dotenv, set_global_seed

MINERS_REGISTRY["hard_triplets"] = HardTripletsMiner


def pl_train(cfg: TCfg) -> None:
    """
    This is an entrypoint for the model training in metric learning setup.

    The config can be specified as a dictionary or with hydra: https://hydra.cc/.
    For more details look at ``examples/README.md``

    """
    cfg = dictconfig_to_dict(cfg)
    trainer_engine_params = parse_engine_params_from_config(cfg)
    is_ddp = check_is_config_for_ddp(trainer_engine_params)

    pprint(cfg)

    set_global_seed(cfg["seed"])

    cwd = Path.cwd()

    train_dataset = DatasetWithLabels(cwd / Path(cfg["dataset_root"]) / cfg["train_npz"])
    valid_dataset = DatasetQueryGallery(cwd / Path(cfg["dataset_root"]) / cfg["valid_npz"])

    sampler_runtime_args = {"labels": train_dataset.get_labels()}
    label2category = None
    if train_dataset.categories_key:
        label2category = dict(zip(train_dataset.labels, train_dataset.categories))
        sampler_runtime_args["label2category"] = label2category
    # note, we pass some runtime arguments to sampler here, but not all of the samplers use all of these arguments
    sampler = get_sampler_by_cfg(cfg["sampler"], **sampler_runtime_args) if cfg["sampler"] is not None else None

    model = SiamNet(train_dataset.get_n_dims())
    criterion = TripletLoss(model, **cfg["criterion"]["args"])
    optimizable_parameters = [{"lr": cfg["optimizer"]["args"]["lr"], "params": model.parameters()}]
    optimizer = get_optimizer_by_cfg(cfg["optimizer"], params=optimizable_parameters)  # type: ignore

    # unpack scheduler to the Lightning format
    if cfg.get("scheduling"):
        scheduler_kwargs = {
            "scheduler": get_scheduler_by_cfg(cfg["scheduling"]["scheduler"], optimizer=optimizer),
            "scheduler_interval": cfg["scheduling"]["scheduler_interval"],
            "scheduler_frequency": cfg["scheduling"]["scheduler_frequency"],
            "scheduler_monitor_metric": cfg["scheduling"].get("monitor_metric", None),
        }
    else:
        scheduler_kwargs = {"scheduler": None}

    if sampler is None:
        loader_train = DataLoader(
            dataset=train_dataset,
            num_workers=cfg["num_workers"],
            batch_size=cfg["bs_train"],
            drop_last=True,
            shuffle=True,
        )
    else:
        loader_train = DataLoader(dataset=train_dataset, batch_sampler=sampler, num_workers=cfg["num_workers"])

    loaders_val = DataLoader(dataset=valid_dataset, batch_size=cfg["bs_val"], num_workers=cfg["num_workers"])

    miner = get_miner(cfg["miner"]["name"], model=model, **cfg["miner"]["args"])
    module_kwargs = scheduler_kwargs

    pl_model = RetrievalModule(
        model=model,
        criterion=criterion,
        miner=miner,
        optimizer=optimizer,
        labels_key=train_dataset.labels_key,
        ones_path=Path.cwd() / "ones",
        **module_kwargs,
    )
    valid_top_k = None
    check_dataset_validity = True
    if cfg["valid_top_k"]:
        data = np.load(Path(cfg["dataset_root"]) / cfg["valid_top_k"])
        valid_top_k = data["top_k"]
        valid_gt = data["gt"]
        check_dataset_validity = False
    metrics_calc = EmbeddingMetrics(
        model=model,
        embeddings_key=pl_model.embeddings_key,
        categories_key=None,
        labels_key=valid_dataset.labels_key,
        is_query_key=valid_dataset.is_query_key,
        is_gallery_key=valid_dataset.is_gallery_key,
        check_dataset_validity=check_dataset_validity,
        validation_top_k_ids=valid_top_k,
        validation_gt=valid_gt,
        **cfg.get("metric_args", {}),
    )
    metrics_clb = MetricValCallback(metric=metrics_calc, log_images=cfg.get("log_images", False))
    ckpt_clb = pl.callbacks.ModelCheckpoint(
        dirpath=Path.cwd() / "checkpoints",
        monitor=cfg["metric_for_checkpointing"],
        mode="max",
        save_top_k=1,
        verbose=True,
        filename="best",
    )

    # Here we try to load NEPTUNE_API_TOKEN from .env file
    # You can also set it up via `export NEPTUNE_API_TOKEN=...`
    load_dotenv()
    if ("NEPTUNE_API_TOKEN" in os.environ.keys()) and (cfg["neptune_project"] is not None):
        logger = NeptuneLogger(
            api_key=os.environ["NEPTUNE_API_TOKEN"],
            project=cfg["neptune_project"],
            tags=list(cfg["tags"]) + [cfg["postfix"]] + [cwd.name],
            log_model_checkpoints=False,
        )
        # log hyper params and files
        dict_to_log = {**dictconfig_to_dict(cfg), **{"dir": cwd}}
        logger.log_hyperparams(flatten_dict(dict_to_log, sep="|"))
        logger.run["dataset"].upload(str(Path(cfg["dataset_root"]) / cfg["train_npz"]))
        # log source code
        source_files = list(map(lambda x: str(x), PROJECT_ROOT.glob("**/*.py"))) + list(
            map(lambda x: str(x), PROJECT_ROOT.glob("**/*.yaml"))
        )
        logger.run["code"].upload_files(source_files)

    else:
        logger = True

    trainer = pl.Trainer(
        max_epochs=cfg["max_epochs"],
        num_sanity_val_steps=0,
        check_val_every_n_epoch=cfg["valid_period"],
        default_root_dir=cwd,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        callbacks=[ckpt_clb, metrics_clb],
        logger=logger,
        precision=cfg.get("precision", 32),
        **trainer_engine_params,
    )
    trainer.validate(dataloaders=loaders_val, verbose=True, model=pl_model)
    # return
    if is_ddp:
        trainer.fit(model=pl_model)
    else:
        trainer.fit(model=pl_model, train_dataloaders=loader_train, val_dataloaders=loaders_val)
        # trainer.fit(model=pl_model, train_dataloaders=loader_train)


@hydra.main(config_path="configs", config_name="train_sop.yaml")
def main(cfg):
    pl_train(cfg)


if __name__ == "__main__":
    main()
