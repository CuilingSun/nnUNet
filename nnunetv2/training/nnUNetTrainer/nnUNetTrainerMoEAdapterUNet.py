import torch
from torch import nn
from typing import List, Tuple, Union
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

from nnunetv2.nets.MoEAdapterUNet import (
    get_multi_encoder_unet_2d_from_plans,
    get_multi_encoder_unet_3d_from_plans,
)

class nnUNetTrainerMoEAdapterUNet(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Improve sampling & validation stability for sparse lesions
        # - more FG oversampling increases the chance to see positives in train/val patches
        # - more val iterations stabilizes online pseudo dice
        self.oversample_foreground_percent = 0.66
        self.probabilistic_oversampling = True
        self.num_val_iterations_per_epoch = 200
        # Allow overrides from env for quick tuning without code edits
        import os
        try:
            if 'NNUNET_OVERSAMPLE_FG' in os.environ:
                self.oversample_foreground_percent = float(os.environ['NNUNET_OVERSAMPLE_FG'])
            if 'NNUNET_VAL_ITERS' in os.environ:
                self.num_val_iterations_per_epoch = int(os.environ['NNUNET_VAL_ITERS'])
        except Exception:
            pass

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        """
        Override to ignore the plans' default architecture and instead build the
        MoEAdapterUNet directly from our Plans/Configuration. We keep the
        signature compatible with the base class so nnUNetTrainer.initialize can
        call us without errors.
        """
        # decide 2D vs 3D from the current configuration
        patch_size = self.configuration_manager.patch_size
        is_3d = len(patch_size) == 3
        if is_3d:
            model = get_multi_encoder_unet_3d_from_plans(
                self.plans_manager, self.dataset_json, self.configuration_manager,
                num_input_channels, deep_supervision=enable_deep_supervision
            )
        else:
            model = get_multi_encoder_unet_2d_from_plans(
                self.plans_manager, self.dataset_json, self.configuration_manager,
                num_input_channels, deep_supervision=enable_deep_supervision
            )

        self.print_to_log_file(f"[MODEL] Using MoEAdapterUNet: {model.__class__.__name__}")
        return model

    def _build_loss(self):
        """Use Dice + TopK CE for sparse foreground robustness.
        Keeps DS weighting consistent with nnUNet defaults.
        """
        from nnunetv2.training.loss.compound_losses import DC_and_topk_loss
        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
        import numpy as np

        assert not self.label_manager.has_regions, "regions not supported by this trainer"

        loss = DC_and_topk_loss(
            {"batch_dice": True, "smooth": 1e-5, "do_bg": False, "ddp": self.is_ddp},
            {"k": 10, "label_smoothing": 0.05},
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            # exponentially decaying weights; ignore the lowest resolution
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
