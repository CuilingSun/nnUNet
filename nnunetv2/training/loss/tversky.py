from typing import Callable

import torch
from torch import nn

from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import softmax_helper_dim1


class SoftTverskyLoss(nn.Module):
    def __init__(self,
                 apply_nonlin: Callable = softmax_helper_dim1,
                 batch_dice: bool = False,
                 do_bg: bool = True,
                 smooth: float = 1e-5,
                 ddp: bool = True,
                 alpha: float = 0.3,
                 beta: float = 0.7):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth
        self.ddp = ddp
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor, y: torch.Tensor, loss_mask: torch.Tensor = None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # compute per-batch or per-sample stats along spatial dims
        if self.batch_dice:
            axes = [0] + list(range(2, x.ndim))
        else:
            axes = list(range(2, x.ndim))

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes=axes, mask=loss_mask, square=False)

        if self.ddp and self.batch_dice:
            tp = AllGatherGrad.apply(tp).sum(0)
            fp = AllGatherGrad.apply(fp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)

        # Tversky index per class
        denom = tp + self.alpha * fp + self.beta * fn
        ti = (tp + self.smooth) / torch.clip(denom + self.smooth, min=1e-8)

        if not self.do_bg:
            if self.batch_dice:
                ti = ti[1:]
            else:
                ti = ti[:, 1:]
        ti = ti.mean()
        return 1.0 - ti


class FocalTverskyLoss(nn.Module):
    def __init__(self,
                 apply_nonlin: Callable = softmax_helper_dim1,
                 batch_dice: bool = False,
                 do_bg: bool = True,
                 smooth: float = 1e-5,
                 ddp: bool = True,
                 alpha: float = 0.3,
                 beta: float = 0.7,
                 gamma: float = 1.5):
        super().__init__()
        self.base = SoftTverskyLoss(apply_nonlin, batch_dice, do_bg, smooth, ddp, alpha, beta)
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor, loss_mask: torch.Tensor = None):
        tversky_loss = self.base(x, y, loss_mask)
        # tversky_loss = 1 - TI, focal = (1 - TI)^gamma = (tversky_loss)^gamma
        return torch.pow(torch.clamp(tversky_loss, min=0.0, max=1.0), self.gamma)

