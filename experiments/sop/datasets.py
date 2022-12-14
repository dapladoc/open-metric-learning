from pathlib import Path
from typing import Union

import numpy as np
from torch.utils.data import Dataset

from oml.const import (
    CATEGORIES_KEY,
    EMBEDDINGS_KEY,
    IS_GALLERY_KEY,
    IS_QUERY_KEY,
    LABELS_KEY,
)


class DatasetWithLabels(Dataset):
    def __init__(self, npz_filepath: Union[str, Path]):
        self.embedding_key = EMBEDDINGS_KEY
        self.labels_key = LABELS_KEY
        data = np.load(npz_filepath)
        self.embeddings = data[self.embedding_key]
        self.labels = data[self.labels_key]
        if data[CATEGORIES_KEY].shape[0] > 0:
            self.categories_key = CATEGORIES_KEY
            self.categories = data[self.categories_key]
        else:
            self.categories_key = None

    def __getitem__(self, idx: int):
        item = {self.embedding_key: self.embeddings[idx], self.labels_key: self.labels[idx]}
        if self.categories_key:
            item[self.categories_key] = self.categories[idx]
        return item

    def get_labels(self):
        return self.labels

    def get_n_dims(self):
        return self.embeddings.shape[1]

    def __len__(self):
        return len(self.labels)


class DatasetQueryGallery(Dataset):
    def __init__(self, npz_filepath: Union[str, Path]):
        self.embedding_key = EMBEDDINGS_KEY
        self.labels_key = LABELS_KEY
        self.is_query_key = IS_QUERY_KEY
        self.is_gallery_key = IS_GALLERY_KEY
        data = np.load(npz_filepath)
        self.embeddings = data[self.embedding_key]
        self.labels = data[self.labels_key]
        self.is_query = data[self.is_query_key]
        self.is_gallery = data[self.is_gallery_key]
        if data[CATEGORIES_KEY].shape[0] > 0:
            self.categories_key = CATEGORIES_KEY
            self.categories = data[self.categories_key]
        else:
            self.categories_key = None

    def __getitem__(self, idx: int):
        item = {
            self.embedding_key: self.embeddings[idx],
            self.labels_key: self.labels[idx],
            self.is_query_key: self.is_query[idx],
            self.is_gallery_key: self.is_gallery[idx],
        }
        if self.categories_key:
            item[self.categories_key] = self.categories[idx]
        return item

    def __len__(self):
        return len(self.labels)
