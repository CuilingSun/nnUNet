"""
Trainer wrapper (non-text) that mirrors your `nnUNetTrainerMultiEncoderUNet` but swaps
in `MultiEncoderAttnUNet` and exposes env-configurable τ/α/γ knobs.

Two-phase training is optionally supported via env:
- ATTN_WARMUP_EPOCHS (default 0): first N epochs only train refiner & gamma
- BASE_LR_REFINER (default None -> fall back to optimizer lr)
"""
import os
from typing import Dict, Any, Optional

from nnunetv2.nets.MultiEncoderAttnUNet import MultiEncoderAttnUNet
try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiEncoderUNet import (
        nnUNetTrainerMultiEncoderUNet as _BaseTrainer,
    )
except ImportError:
    try:
        from nnUNetTrainerMultiEncoderUNet import nnUNetTrainerMultiEncoderUNet as _BaseTrainer
    except Exception as e:
        raise ImportError("Please ensure nnUNetTrainerMultiEncoderUNet.py is importable: %r" % e)


class nnUNetTrainerMultiEncoderAttnUNet(_BaseTrainer):
    def initialize(self, training: bool = True) -> None:
        try:
            super().initialize(training)
        except TypeError:
            super().initialize()
        # Swap network to Attn variant if not already
        if not isinstance(self.network, MultiEncoderAttnUNet):
            # Recreate with same constructor args when possible
            # Fallback: assume your trainer stores plans/config to rebuild – if not, you can manually
            # construct MultiEncoderAttnUNet in your codebase and assign to self.network before calling super().
            base_net = self.network
            attn_net = None
            if hasattr(base_net, 'init_kwargs'):
                init_kwargs = getattr(base_net, 'init_kwargs', {})
                try:
                    attn_net = MultiEncoderAttnUNet(**init_kwargs)
                    attn_net.load_state_dict(base_net.state_dict(), strict=False)
                except Exception:
                    attn_net = None
            if attn_net is None:
                attn_net = MultiEncoderAttnUNet.from_existing(base_net)
            self.network = attn_net

        # Read env configs
        use_xattn = os.environ.get('NNUNET_USE_CROSS_ATTN_FINAL', '1') == '1'
        alpha = float(os.environ.get('NNUNET_CROSS_ALPHA', '0.25'))
        tau = float(os.environ.get('NNUNET_CROSS_TAU', '0.35'))
        gamma0 = float(os.environ.get('NNUNET_CROSS_GAMMA_INIT', '0.255'))
        self.network.use_cross_attn_final = use_xattn
        self.network.cross_alpha = alpha
        self.network.cross_tau = tau
        self.network.cross_gamma_init = gamma0
        if not self.network._installed:
            self.network.install_refiner()

        # Optional warmup: only train refiner + gamma first N epochs
        self._attn_warmup_epochs = int(os.environ.get('ATTN_WARMUP_EPOCHS', '0'))
        self._refiner_param_names = set()
        if self.network._refiner is not None:
            for n, _ in self.network._refiner.named_parameters():
                self._refiner_param_names.add(f"_refiner.{n}")
        if self.network._refine_head is not None:
            for n, _ in self.network._refine_head.named_parameters():
                self._refiner_param_names.add(f"_refine_head.{n}")
        # Include gamma
        self._refiner_param_names.add("_refiner.gamma")

    def on_epoch_start(self, epoch: Optional[int] = None) -> None:
        super().on_epoch_start()
        if epoch is None:
            epoch = getattr(self, 'current_epoch', 0)
        if self._attn_warmup_epochs > 0 and epoch < self._attn_warmup_epochs:
            # freeze all except refiner & gamma
            for n, p in self.network.named_parameters():
                p.requires_grad = (n in self._refiner_param_names)
        elif self._attn_warmup_epochs > 0 and epoch == self._attn_warmup_epochs:
            for _, p in self.network.named_parameters():
                p.requires_grad = True
