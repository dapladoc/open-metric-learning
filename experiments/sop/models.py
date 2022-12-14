from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

from experiments.sop.losses import TripletLoss
from oml.const import EMBEDDINGS_KEY, LABELS_KEY
from oml.interfaces.miners import ITripletsMiner, labels2list


class RetrievalModule(pl.LightningModule):
    """
    This is a base module to train your model with Lightning.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: TripletLoss,
        miner: ITripletsMiner,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        scheduler_interval: str = "step",
        scheduler_frequency: int = 1,
        labels_key: str = LABELS_KEY,
        embeddings_key: str = EMBEDDINGS_KEY,
        scheduler_monitor_metric: Optional[str] = None,
        freeze_n_epochs: int = 0,
        ones_path: Union[str, Path] = "",
    ):
        pl.LightningModule.__init__(self)

        self.model = model
        self.criterion = criterion
        self.miner = miner
        self.optimizer = optimizer

        self.monitor_metric = scheduler_monitor_metric
        self.scheduler = scheduler
        self.scheduler_interval = scheduler_interval
        self.scheduler_frequency = scheduler_frequency

        self.labels_key = labels_key
        self.embeddings_key = embeddings_key

        self.freeze_n_epochs = freeze_n_epochs
        self.ones_path = Path(ones_path)
        self.ones_path.mkdir(parents=True)
        self.ones_save_count = 0

    def forward(self, x1: torch.Tensor, x2) -> torch.Tensor:
        scores = self.model(x1, x2)
        return scores

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        labels = batch[self.labels_key]
        features = batch[self.embeddings_key]
        bs = len(labels)

        labels_list = labels2list(labels)
        anchor, positive, negative = self.miner.sample(features=features, labels=labels_list)
        loss = self.criterion(anchor=anchor, positive=positive, negative=negative)

        loss_name = (getattr(self.criterion, "criterion_name", "") + "_loss").strip("_")
        self.log(loss_name, loss.item(), prog_bar=True, batch_size=bs, on_step=True, on_epoch=True)

        if hasattr(self.criterion, "last_logs"):
            self.log_dict(self.criterion.last_logs, prog_bar=False, batch_size=bs, on_step=True, on_epoch=False)

        if self.scheduler is not None:
            self.log("lr", self.scheduler.get_last_lr()[0], prog_bar=True, batch_size=bs, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int, *_: Any) -> Dict[str, Any]:
        return batch

    def configure_optimizers(self) -> Any:
        if self.scheduler is None:
            return self.optimizer
        else:
            scheduler = {
                "scheduler": self.scheduler,
                "interval": self.scheduler_interval,
                "frequency": self.scheduler_frequency,
            }
            if isinstance(self.scheduler, ReduceLROnPlateau):
                scheduler["monitor"] = self.monitor_metric
            return [self.optimizer], [scheduler]

    def get_progress_bar_dict(self) -> Dict[str, Union[int, str]]:
        # https://github.com/Lightning-AI/lightning/issues/1595
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict

    def on_epoch_end(self) -> None:
        ones = self.model.one.detach().cpu().numpy()
        np.save(str(self.ones_path / f"{self.ones_save_count:05}.npy"), ones)
        self.ones_save_count += 1


class SiamNet(torch.nn.Module):
    def __init__(self, n_dims: int):
        super().__init__()
        self.one = torch.nn.Parameter(torch.ones(n_dims, 1) + (1.0e-4 * torch.rand(n_dims, 1) - 1.0e-4 / 2))

    def forward(self, x1, x2):
        diff = torch.pow(x1 - x2, 2)
        scores = torch.matmul(diff, self.one / torch.linalg.vector_norm(self.one))
        return scores
