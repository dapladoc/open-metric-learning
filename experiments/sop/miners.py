from typing import List

import numpy as np
import torch
from torch import Tensor

from experiments.cars.metrics import pairwise_dist
from oml.interfaces.miners import ITripletsMinerInBatch, TTripletsIds, TLabels, TTriplets, labels2list
from oml.utils.misc import find_value_ids


class HardTripletsMiner(ITripletsMinerInBatch):
    """
    This miner selects the hardest triplets based on the distances between the features:

    - The hardest `positive` sample has the `maximal` distance to the anchor sample

    - The hardest `negative` sample has the `minimal` distance to the anchor sample

    """

    def __init__(self, model: torch.nn.Module, bs: int):
        self.model = model
        self.bs = bs

    def _sample(self, features: Tensor, labels: List[int]) -> TTripletsIds:
        """
        This method samples the hardest triplets inside the batch.

        Args:
            features: Features with the shape of ``[batch_size, feature_size]``
            labels: Labels with the size of ``batch_size``

        Returns:
            The batch of the triplets in the order below:
            ``(anchor, positive, negative)``

        """
        assert features.shape[0] == len(labels)

        features = features.clone().detach()

        dist_mat = pairwise_dist(model=self.model, x1=features, x2=features, bs=self.bs)

        ids_anchor, ids_pos, ids_neg, *rest = self._sample_from_distmat(distmat=dist_mat, labels=labels)

        return ids_anchor, ids_pos, ids_neg, *rest

    @staticmethod
    def _sample_from_distmat(distmat: Tensor, labels: List[int]) -> TTripletsIds:
        """
        This method samples the hardest triplets based on the given
        distances matrix. It chooses each sample in the batch as an
        anchor and then finds the hardest positive and negative pair.

        Args:
            distmat: Matrix with the shape of ``[batch_size, batch_size]``
            labels: Labels with the size of ``batch_size``

        Returns:
            The batch of the triplets with the size of ``batch_size``, order is the following:
                ``(anchor, positive, negative)``

        """
        ids_all = set(range(len(labels)))

        ids_anchor, ids_pos, ids_neg = [], [], []
        hardest_pos_dists = []
        hardest_neg_dists = []

        for i_anch, label in enumerate(labels):
            ids_label = set(find_value_ids(it=labels, value=label))

            ids_pos_cur = np.array(list(ids_label - {i_anch}), int)
            ids_neg_cur = np.array(list(ids_all - ids_label), int)

            hardest_pos_dists.append(distmat[i_anch, ids_pos_cur].max().item())
            hardest_neg_dists.append(distmat[i_anch, ids_neg_cur].min().item())

            i_pos = ids_pos_cur[distmat[i_anch, ids_pos_cur].argmax()]
            i_neg = ids_neg_cur[distmat[i_anch, ids_neg_cur].argmin()]

            ids_anchor.append(i_anch)
            ids_pos.append(i_pos)
            ids_neg.append(i_neg)

        return ids_anchor, ids_pos, ids_neg, hardest_pos_dists, hardest_neg_dists

    def sample(self, features: Tensor, labels: TLabels) -> TTriplets:
        """
        Args:
            features: Features with the shape of ``[batch_size, feature_size]``
            labels: Labels with the size of ``batch_size``

        Returns:
             Batch of triplets

        """
        # Convert labels to list
        labels = labels2list(labels)
        self._check_input_labels(labels=labels)

        ids_anchor, ids_pos, ids_neg, *rest = self._sample(features, labels=labels)

        return features[ids_anchor], features[ids_pos], features[ids_neg], *rest
