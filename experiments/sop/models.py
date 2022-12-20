from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.functional as F
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

from experiments.sop.losses import TripletLoss
from oml.const import EMBEDDINGS_KEY, LABELS_KEY
from oml.interfaces.miners import ITripletsMiner, labels2list
from oml.models.vit.vision_transformer import DropPath, Mlp
from oml.utils.misc_torch import elementwise_dist


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
        anchor, positive, negative, *rest = self.miner.sample(features=features, labels=labels_list)
        if rest:
            self.log("hardest positive distances mean", np.mean(rest[0]).item())
            self.log("hardest negative distances mean", np.mean(rest[1]).item())
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
        pass
        # ones = self.model.one.detach().cpu().numpy()
        # np.save(str(self.ones_path / f"{self.ones_save_count:05}.npy"), ones)
        # self.ones_save_count += 1


class Attention(torch.nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention2(torch.nn.Module):
    def __init__(self, n_dims: int):
        super().__init__()
        self.n_dims = n_dims
        self.linear = nn.Linear(n_dims, n_dims)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class Attention3(torch.nn.Module):
    def __init__(self, n_dims: int):
        super().__init__()
        self.n_dims = n_dims
        self.linear = nn.Linear(2 * n_dims, n_dims * n_dims * 2)

    def forward(self, x):
        output = torch.tanh(self.linear(x))
        output = output.view(self.n_dims, self.n_dims * 2)
        return output


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        y = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SiamNet(torch.nn.Module):
    def __init__(self, n_dims):
        super().__init__()
        self.n_dims = n_dims
        self.block_1 = Block(
            n_dims,
            num_heads=1,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
        )
        self.block_2 = Block(
            n_dims,
            num_heads=1,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
        )
        self.block_3 = Block(
            n_dims,
            num_heads=1,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
        )

    def forward(self, x1, x2):
        x = torch.cat((x1.unsqueeze(1), x2.unsqueeze(1)), dim=1)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x).permute(1, 0, 2)
        x1, x2 = x
        dist = elementwise_dist(x1, x2, p=2)
        return dist
