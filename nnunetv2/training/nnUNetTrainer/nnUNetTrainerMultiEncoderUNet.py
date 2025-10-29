import os

import torch
from torch import nn
from typing import List, Tuple, Union
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results

from nnunetv2.nets.MultiEncoderUNet import (
    get_multi_encoder_unet_2d_from_plans,
    get_multi_encoder_unet_3d_from_plans,
)

class nnUNetTrainerMultiEncoderUNet(nnUNetTrainer):
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
        *args,
        **kwargs,
    ) -> nn.Module:
        """Build MultiEncoderUNet for both training (instance call) and inference (class call).

        nnUNet's inference utilities call this as a class method, so we need to gracefully
        handle missing `self` and fetch configuration objects from kwargs instead.
        """

        args_list = list(args)

        # Defensive: if python passes self positionally we remove it to avoid confusion.
        if args_list and args_list[0] is self:
            args_list.pop(0)

        trainer_instance = self if isinstance(self, nnUNetTrainerMultiEncoderUNet) else None

        architecture_class_name = kwargs.pop('architecture_class_name', None)
        arch_init_kwargs = kwargs.pop('arch_init_kwargs', None)
        arch_init_kwargs_req_import = kwargs.pop('arch_init_kwargs_req_import', None)
        num_input_channels = kwargs.pop('num_input_channels', None)
        num_output_channels = kwargs.pop('num_output_channels', None)
        enable_deep_supervision = kwargs.pop('enable_deep_supervision', True)

        def _pop_or(current):
            return current if current is not None else (args_list.pop(0) if args_list else None)

        if trainer_instance is None and architecture_class_name is None:
            architecture_class_name = self

        architecture_class_name = _pop_or(architecture_class_name)
        arch_init_kwargs = _pop_or(arch_init_kwargs) or {}
        arch_init_kwargs_req_import = _pop_or(arch_init_kwargs_req_import)
        num_input_channels = _pop_or(num_input_channels)
        num_output_channels = _pop_or(num_output_channels)

        if args_list:
            enable_deep_supervision = args_list.pop(0)

        if architecture_class_name is None or num_input_channels is None or num_output_channels is None:
            raise RuntimeError(
                "build_network_architecture expects architecture_class_name, num_input_channels and num_output_channels"
            )

        plans_manager = arch_init_kwargs.get('plans_manager') or kwargs.get('plans_manager')
        configuration_manager = arch_init_kwargs.get('configuration_manager') or kwargs.get('configuration_manager')
        dataset_json = arch_init_kwargs.get('dataset_json') or kwargs.get('dataset_json')

        if trainer_instance is not None:
            plans_manager = plans_manager or trainer_instance.plans_manager
            configuration_manager = configuration_manager or trainer_instance.configuration_manager
            dataset_json = dataset_json or trainer_instance.dataset_json

        # Fallback for inference: rebuild managers from env hints if they are missing
        if plans_manager is None or configuration_manager is None or dataset_json is None:
            plans_path_env = os.environ.get('NNUNET_PLANS_FILE')
            results_dir = os.environ.get('NNUNET_RESULTS_DIR') or os.environ.get('nnUNet_results', nnUNet_results)
            dataset_id = os.environ.get('NNUNET_DATASET') or os.environ.get('nnUNet_dataset')
            plans_name = os.environ.get('NNUNET_PLANS', 'nnUNetPlans')
            config_name = os.environ.get('NNUNET_CONFIG')
            preprocessed_root = os.environ.get('nnUNet_preprocessed', nnUNet_preprocessed)

            candidate_plans = []
            candidate_dataset_json = None

            if plans_path_env and os.path.isfile(plans_path_env):
                candidate_plans.append(plans_path_env)

            if results_dir:
                model_output_dir = kwargs.get('model_output_dir') or arch_init_kwargs.get('model_output_dir')
                if model_output_dir:
                    trainer_dir = model_output_dir
                elif dataset_id:
                    trainer_dir = os.path.join(
                        results_dir,
                        maybe_convert_to_dataset_name(dataset_id),
                        f"{trainer_instance.__class__.__name__ if trainer_instance else self.__name__}__{plans_name}__{config_name or '3d_fullres'}",
                    )
                else:
                    trainer_dir = None

                if trainer_dir:
                    trainer_plans = os.path.join(trainer_dir, 'plans.json')
                    if os.path.isfile(trainer_plans):
                        candidate_plans.append(trainer_plans)
                    trainer_dataset_json = os.path.join(trainer_dir, 'dataset.json')
                    if os.path.isfile(trainer_dataset_json):
                        candidate_dataset_json = load_json(trainer_dataset_json)

            if dataset_id and preprocessed_root:
                dataset_folder = os.path.join(preprocessed_root, maybe_convert_to_dataset_name(dataset_id))
                candidate_plans.append(os.path.join(dataset_folder, f'{plans_name}.json'))
                dataset_json_path = os.path.join(dataset_folder, 'dataset.json')
                if os.path.isfile(dataset_json_path):
                    candidate_dataset_json = load_json(dataset_json_path)

            if plans_manager is None:
                for p in candidate_plans:
                    if os.path.isfile(p):
                        plans_manager = PlansManager(p)
                        break

            if configuration_manager is None and plans_manager is not None:
                cfg_name = (
                    config_name
                    or (trainer_instance.configuration_manager.configuration_name if trainer_instance else None)
                    or '3d_fullres'
                )
                try:
                    configuration_manager = plans_manager.get_configuration(cfg_name)
                except KeyError:
                    available = list(plans_manager.configurations.keys())
                    configuration_manager = plans_manager.get_configuration(available[0]) if available else None

            if dataset_json is None:
                if arch_init_kwargs.get('dataset_json_path') and os.path.isfile(arch_init_kwargs['dataset_json_path']):
                    dataset_json = load_json(arch_init_kwargs['dataset_json_path'])
                elif candidate_dataset_json is not None:
                    dataset_json = candidate_dataset_json

        if plans_manager is None or configuration_manager is None:
            raise RuntimeError("plans_manager and configuration_manager must be provided for network construction.")

        # decide 2D vs 3D from the current configuration
        if len(configuration_manager.patch_size) == 2:
            model = get_multi_encoder_unet_2d_from_plans(
                plans_manager,
                dataset_json,
                configuration_manager,
                num_input_channels,
                deep_supervision=enable_deep_supervision,
            )
        elif len(configuration_manager.patch_size) == 3:
            model = get_multi_encoder_unet_3d_from_plans(
                plans_manager,
                dataset_json,
                configuration_manager,
                num_input_channels,
                deep_supervision=enable_deep_supervision,
            )
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")

        print("MultiEncoderUNet:", model.__class__.__name__)
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
