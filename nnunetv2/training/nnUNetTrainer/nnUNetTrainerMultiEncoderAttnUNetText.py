"""
Text-enabled trainer that mirrors your `nnUNetTrainerMultiEncoderUNetText` but enables
and routes the τ/α/γ knobs into `MultiEncoderAttnUNet`.
"""
import os
from typing import Optional

import torch

from nnunetv2.nets.MultiEncoderAttnUNet import MultiEncoderAttnUNet
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiEncoderUNet import (
    nnUNetTrainerMultiEncoderUNet,
)
try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiEncoderUNetText import (
        nnUNetTrainerMultiEncoderUNetText as _BaseTextTrainer,
    )
except ImportError:
    try:
        from nnUNetTrainerMultiEncoderUNetText import (
            nnUNetTrainerMultiEncoderUNetText as _BaseTextTrainer,
        )
    except Exception as e:
        raise ImportError("Please ensure nnUNetTrainerMultiEncoderUNetText.py is importable: %r" % e)


class nnUNetTrainerMultiEncoderAttnUNetText(_BaseTextTrainer):
    def build_network_architecture(self, *args, **kwargs):
        """Ensure inference builds the attention variant instead of the plain MultiEncoderUNet."""
        base_network = nnUNetTrainerMultiEncoderUNet.build_network_architecture(self, *args, **kwargs)

        if isinstance(base_network, MultiEncoderAttnUNet):
            return base_network

        attn_net = None
        if hasattr(base_network, 'init_kwargs'):
            init_kwargs = getattr(base_network, 'init_kwargs', {})
            try:
                attn_net = MultiEncoderAttnUNet(**init_kwargs)
                attn_net.load_state_dict(base_network.state_dict(), strict=False)
            except Exception:
                attn_net = None

        if attn_net is None:
            attn_net = MultiEncoderAttnUNet.from_existing(base_network)

        return attn_net

    def initialize(self, training: bool = True) -> None:
        try:
            super().initialize(training)
        except TypeError:
            super().initialize()
        # ensure network is attn variant
        if not isinstance(self.network, MultiEncoderAttnUNet):
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

        # wire env configs
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
        if getattr(self, 'was_initialized_from_checkpoint', False) is False:
            if self.network._refiner is not None:
                with torch.no_grad():
                    self.network._refiner.gamma.fill_(gamma0)

        # optional warmup
        self._attn_warmup_epochs = int(os.environ.get('ATTN_WARMUP_EPOCHS', '0'))
        lr_refiner = os.environ.get('BASE_LR_REFINER')
        self._refiner_lr = float(lr_refiner) if lr_refiner is not None else None
        self._refiner_param_names = set()
        if self.network._refiner is not None:
            for n, _ in self.network._refiner.named_parameters():
                self._refiner_param_names.add(f"_refiner.{n}")
        if self.network._refine_head is not None:
            for n, _ in self.network._refine_head.named_parameters():
                self._refiner_param_names.add(f"_refine_head.{n}")
        self._refiner_param_names.add("_refiner.gamma")

    def on_epoch_start(self, epoch: Optional[int] = None) -> None:
        super().on_epoch_start()
        if epoch is None:
            epoch = getattr(self, 'current_epoch', 0)
        if self._attn_warmup_epochs > 0 and epoch < self._attn_warmup_epochs:
            for n, p in self.network.named_parameters():
                p.requires_grad = (n in self._refiner_param_names)
            if hasattr(self, 'optimizer') and len(self.optimizer.param_groups) >= 2:
                self.optimizer.param_groups[0]['lr'] = 0.0
        elif self._attn_warmup_epochs > 0 and epoch == self._attn_warmup_epochs:
            for _, p in self.network.named_parameters():
                p.requires_grad = True
            if hasattr(self, 'optimizer') and len(self.optimizer.param_groups) >= 2:
                base_lr_env = os.environ.get('BASE_LR')
                if self._refiner_lr is not None:
                    base_lr = self._refiner_lr
                elif base_lr_env is not None:
                    base_lr = float(base_lr_env)
                else:
                    base_lr = self.optimizer.param_groups[1]['lr'] / 5
                self.optimizer.param_groups[0]['lr'] = base_lr
