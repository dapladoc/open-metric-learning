from typing import Dict, Optional

import torch
from torch import Tensor

from oml.functional.losses import get_reduced


class TripletLoss(torch.nn.Module):

    criterion_name = "triplet"  # for better logging

    def __init__(
        self, model: torch.nn.Module, margin: Optional[float], reduction: str = "mean", need_logs: bool = False
    ):
        assert reduction in ("mean", "sum", "none")
        assert (margin is None) or (margin > 0)

        super(TripletLoss, self).__init__()

        self.model = model
        self.margin = margin
        self.reduction = reduction
        self.need_logs = need_logs
        self.last_logs: Dict[str, float] = {}

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        assert anchor.shape == positive.shape == negative.shape

        positive_dist = self.model(anchor, positive)
        negative_dist = self.model(anchor, negative)

        if self.margin is None:
            # here is the soft version of TripletLoss without margin
            loss = torch.log1p(torch.exp(positive_dist - negative_dist))
            # loss = positive_dist - negative_dist
        else:
            loss = torch.relu(self.margin + positive_dist - negative_dist)

        if self.need_logs:
            self.last_logs = {
                "active_tri": float((loss.clone().detach() > 0).float().mean()),
                "pos_dist": float(positive_dist.clone().detach().mean().item()),
                "neg_dist": float(negative_dist.clone().detach().mean().item()),
            }

        loss = get_reduced(loss, reduction=self.reduction)

        return loss
