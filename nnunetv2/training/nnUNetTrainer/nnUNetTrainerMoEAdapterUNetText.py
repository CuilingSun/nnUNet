from __future__ import annotations

import torch
from torch import nn
from typing import Any, List, Optional, Tuple, Union

from batchgenerators.utilities.file_and_folder_operations import join, load_json

# Base trainers we extend to keep all behaviors & method names
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiEncoderUNet import (
    nnUNetTrainerMultiEncoderUNet,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiEncoderUNetText import (
    nnUNetTrainerMultiEncoderUNetText,
)

# Import MoEAdapterUNet factories (keep original names for drop-in)
from nnunetv2.nets.MoEAdapterUNet import (
    get_multi_encoder_unet_2d_from_plans,
    get_multi_encoder_unet_3d_from_plans,
)

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.plans_handling.plans_handler import (
    PlansManager,
    ConfigurationManager,
)


class nnUNetTrainerMoEAdapterUNetText(nnUNetTrainerMultiEncoderUNetText):
    """
    Text-augmented trainer for MoEAdapterUNet.

    We inherit all text-handling utilities from `nnUNetTrainerMultiEncoderUNetText`
    (env-driven text dims, heatmap alignment losses, FiLM/gating, etc.).
    The only change is swapping the network factory in `build_network_architecture`.
    """

    # Keep identical method name/signature so downstream code doesn't break
    def build_network_architecture(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> nn.Module:
        """Build MoEAdapterUNet with text conditioning for both training and inference.

        When called from training (`self` is an instance) the signature matches the
        base trainer. When called from inference (`trainer_class.build_network_architecture`)
        the first positional argument will actually be the architecture class name and
        no instance is passed. We handle both cases here.
        """

        # Normalize arguments: support both instance calls (training) and class calls (prediction)
        args_list = list(args)

        # Remove leading instance if python supplied it in args (should not happen, but safeguard)
        if args_list and args_list[0] is self:
            args_list.pop(0)

        trainer_instance = self if isinstance(self, nnUNetTrainerMoEAdapterUNetText) else None

        # Collect positional / keyword arguments with fallbacks
        architecture_class_name = kwargs.pop('architecture_class_name', None)
        arch_init_kwargs = kwargs.pop('arch_init_kwargs', None)
        arch_init_kwargs_req_import = kwargs.pop('arch_init_kwargs_req_import', None)
        num_input_channels = kwargs.pop('num_input_channels', None)
        num_output_channels = kwargs.pop('num_output_channels', None)
        enable_deep_supervision = kwargs.pop('enable_deep_supervision', True)
        plans_manager_kw = kwargs.pop('plans_manager', None)
        configuration_manager_kw = kwargs.pop('configuration_manager', None)
        dataset_json_kw = kwargs.pop('dataset_json', None)

        def _pop_or(existing):
            return existing if existing is not None else (args_list.pop(0) if args_list else None)

        if trainer_instance is None and architecture_class_name is None:
            architecture_class_name = self

        architecture_class_name = _pop_or(architecture_class_name)
        arch_init_kwargs = _pop_or(arch_init_kwargs)
        arch_init_kwargs_req_import = _pop_or(arch_init_kwargs_req_import)
        num_input_channels = _pop_or(num_input_channels)
        num_output_channels = _pop_or(num_output_channels)

        if args_list:
            enable_deep_supervision = args_list.pop(0)

        if architecture_class_name is None or num_input_channels is None or num_output_channels is None:
            raise RuntimeError(
                "build_network_architecture received insufficient parameters. "
                "Expected architecture_class_name, num_input_channels and num_output_channels."
            )

        arch_init_kwargs = arch_init_kwargs or {}
        pm = arch_init_kwargs.get("plans_manager") or plans_manager_kw
        cm = arch_init_kwargs.get("configuration_manager") or configuration_manager_kw
        dataset_json = dataset_json_kw

        if trainer_instance is not None:
            pm = pm or trainer_instance.plans_manager
            cm = cm or trainer_instance.configuration_manager
            dataset_json = trainer_instance.dataset_json
            log_fn = trainer_instance.print_to_log_file
        else:
            log_fn = lambda *a, **k: None

        if pm is None or cm is None:
            raise RuntimeError(
                "plans_manager and configuration_manager must be provided either via arch_init_kwargs or via the "
                "trainer instance."
            )

        if dataset_json is None and 'dataset_json' in arch_init_kwargs:
            dataset_json = arch_init_kwargs['dataset_json']

        if dataset_json is None:
            try:
                dataset_name = pm.dataset_name if hasattr(pm, 'dataset_name') else None
                if dataset_name is not None and nnUNet_preprocessed is not None:
                    ds_path = join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name), 'dataset.json')
                    dataset_json = load_json(ds_path)
            except Exception:
                dataset_json = None

        if dataset_json is None and trainer_instance is not None:
            dataset_json = trainer_instance.dataset_json

        if dataset_json is None:
            raise RuntimeError(
                "Unable to resolve dataset_json for build_network_architecture. Please provide it via kwargs or "
                "ensure nnUNet_preprocessed path is set."
            )

        if trainer_instance is not None and (cm is None or not hasattr(cm, "patch_size")):
            cm = trainer_instance.configuration_manager

        if cm is None:
            raise RuntimeError("configuration_manager could not be resolved for network construction.")

        patch_size = cm.patch_size
        is_3d = len(patch_size) == 3

        if is_3d:
            model = get_multi_encoder_unet_3d_from_plans(
                pm,
                dataset_json,
                cm,
                num_input_channels,
                deep_supervision=enable_deep_supervision,
            )
        else:
            model = get_multi_encoder_unet_2d_from_plans(
                pm,
                dataset_json,
                cm,
                num_input_channels,
                deep_supervision=enable_deep_supervision,
            )

        if trainer_instance is not None:
            cross_flag = int(bool(getattr(model, "use_cross_attn_final", False)))
            adapter_flag = int(bool(getattr(model, "use_text_adapter", False)))
            log_fn(
                f"[MODEL] Using MoEAdapterUNet (Text): {model.__class__.__name__} "
                f"| cross_attn={cross_flag} | text_adapter={adapter_flag}"
            )

        return model
