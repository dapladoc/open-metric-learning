from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from oml.const import (
    CATEGORIES_KEY,
    IS_GALLERY_KEY,
    IS_QUERY_KEY,
    LABELS_KEY,
    PATHS_KEY,
)
from oml.datasets.base import get_retrieval_datasets
from oml.registry.models import get_extractor_by_cfg
from oml.registry.transforms import get_transforms_by_cfg
from oml.utils.misc import dictconfig_to_dict

torch.multiprocessing.set_sharing_strategy("file_system")


def extract_embeddings(model, loader, n_passes, device, features_filepath):
    embeddings = []
    paths = []
    labels = []
    is_query = []
    is_gallery = []
    categories = []
    with torch.no_grad():
        for _ in tqdm(range(n_passes)):
            for batch in tqdm(loader, leave=False):
                embeddings.append(model.forward(batch["input_tensors"].to(device)).cpu().numpy())
                labels.append(batch[LABELS_KEY].numpy())
                paths.append(batch[PATHS_KEY])
                if CATEGORIES_KEY in batch:
                    categories.append(batch[CATEGORIES_KEY])
                if IS_QUERY_KEY in batch:
                    is_query.append(batch[IS_QUERY_KEY].numpy())
                if IS_GALLERY_KEY in batch:
                    is_gallery.append(batch[IS_GALLERY_KEY].numpy())

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    paths = np.concatenate(paths)
    if categories:
        categories = np.concatenate(categories)
    if is_query:
        is_query = np.concatenate(is_query)
    if is_gallery:
        is_gallery = np.concatenate(is_gallery)
    np.savez(
        features_filepath,
        embeddings=embeddings,
        labels=labels,
        is_query=is_query,
        is_gallery=is_gallery,
        categories=categories,
        paths=paths,
    )


def main():
    cfg = OmegaConf.load("configs/extract_features.yaml")
    cfg = dictconfig_to_dict(cfg)

    transforms_train = get_transforms_by_cfg(cfg["transforms_train"])
    transforms_val = get_transforms_by_cfg(cfg["transforms_val"])

    extractor = get_extractor_by_cfg(cfg["model"])
    extractor.to(cfg["device"])
    extractor.eval()

    train_dataset, valid_dataset = get_retrieval_datasets(
        dataset_root=Path(cfg["dataset_root"]),
        transforms_train=transforms_train,
        transforms_val=transforms_val,
        dataframe_name=cfg["dataframe_name"],
    )
    loader_valid = DataLoader(
        dataset=valid_dataset, batch_size=cfg["bs_val"], num_workers=cfg["num_workers"], drop_last=False, shuffle=False
    )
    loader_train = DataLoader(
        dataset=train_dataset,
        num_workers=cfg["num_workers"],
        batch_size=cfg["bs_train"],
        drop_last=False,
        shuffle=False,
    )

    extract_embeddings(extractor, loader_train, cfg["n_passes_train"], cfg["device"], cfg["train_features_filepath"])
    extract_embeddings(extractor, loader_valid, cfg["n_passes_valid"], cfg["device"], cfg["valid_features_filepath"])


if __name__ == "__main__":
    main()
