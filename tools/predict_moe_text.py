#!/usr/bin/env python
"""Custom prediction script for MoE text-conditioned nnU-Net models."""

import argparse
from pathlib import Path

import torch
from torch.serialization import add_safe_globals
from batchgenerators.utilities.file_and_folder_operations import load_json, maybe_mkdir_p

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMoEAdapterUNetText import (
    nnUNetTrainerMoEAdapterUNetText,
)
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference helper for MoE Adapter UNet Text models.")
    parser.add_argument("--model-dir", required=True,
                        help="Path to the trainer output directory (contains fold_x subdirectories).")
    parser.add_argument("--fold", default="1",
                        help="Fold to use for inference. Default: 1")
    parser.add_argument("--checkpoint", default="checkpoint_best.pth",
                        help="Checkpoint filename within the fold directory.")
    parser.add_argument("--input", required=True,
                        help="Directory with input cases (expects nnUNet-style channel numbering).")
    parser.add_argument("--output", required=True,
                        help="Directory for predictions (created if missing).")
    parser.add_argument("--device", default="cuda",
                        help="Torch device for inference (e.g. cuda, cuda:0, cpu).")
    parser.add_argument("--step-size", type=float, default=0.5,
                        help="Sliding window step size. Default: 0.5")
    parser.add_argument("--disable-tta", action="store_true",
                        help="Disable mirroring-based test-time augmentation.")
    parser.add_argument("--save-probabilities", action="store_true",
                        help="Save probability maps alongside segmentations.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing predictions in the output directory.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose predictor output.")
    return parser.parse_args()


def resolve_dataset_json(dataset_name: str) -> dict:
    if nnUNet_preprocessed is None:
        raise RuntimeError("nnUNet_preprocessed is not set; cannot locate dataset.json")
    ds_folder = Path(nnUNet_preprocessed) / maybe_convert_to_dataset_name(dataset_name)
    dataset_json_file = ds_folder / "dataset.json"
    if not dataset_json_file.is_file():
        raise RuntimeError(f"dataset.json not found at {dataset_json_file}")
    return load_json(dataset_json_file)


def load_checkpoint(checkpoint_path: Path) -> dict:
    import numpy.core.multiarray  # type: ignore

    add_safe_globals([numpy.core.multiarray.scalar])
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def initialize_network(plans_manager: PlansManager,
                       dataset_json: dict,
                       configuration_name: str,
                       num_input_channels: int,
                       num_output_channels: int,
                       checkpoint: dict,
                       device: torch.device):
    configuration_manager = plans_manager.get_configuration(configuration_name)

    arch_kwargs = dict(configuration_manager.network_arch_init_kwargs or {})
    arch_kwargs.setdefault("plans_manager", plans_manager)
    arch_kwargs.setdefault("configuration_manager", configuration_manager)
    arch_kwargs.setdefault("dataset_json", dataset_json)

    network = nnUNetTrainerMoEAdapterUNetText.build_network_architecture(
        configuration_manager.network_arch_class_name,
        arch_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=False,
    )

    network.load_state_dict(checkpoint["network_weights"])
    network = network.to(device)
    network.eval()
    return network, configuration_manager


def main() -> None:
    args = parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    fold_dir = model_dir / f"fold_{args.fold}"
    checkpoint_path = fold_dir / args.checkpoint

    if not fold_dir.is_dir():
        raise FileNotFoundError(f"Fold directory not found: {fold_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    plans_file = model_dir / "plans.json"
    if not plans_file.is_file():
        raise FileNotFoundError(f"plans.json not found in {model_dir}")

    plans = load_json(plans_file)
    dataset_name = plans["dataset_name"]
    dataset_json = load_json(model_dir / "dataset.json") if (model_dir / "dataset.json").is_file() else resolve_dataset_json(dataset_name)

    checkpoint = load_checkpoint(checkpoint_path)
    configuration_name = checkpoint["init_args"]["configuration"]

    plans_manager = PlansManager(plans_file)
    configuration_manager = plans_manager.get_configuration(configuration_name)
    num_input_channels = len(dataset_json.get("channel_names", {}))
    label_manager = plans_manager.get_label_manager(dataset_json)
    num_output_channels = label_manager.num_segmentation_heads

    device = torch.device(args.device)

    network, configuration_manager = initialize_network(
        plans_manager,
        dataset_json,
        configuration_name,
        num_input_channels,
        num_output_channels,
        checkpoint,
        device,
    )

    inference_allowed_axes = checkpoint.get("inference_allowed_mirroring_axes", None)

    predictor = nnUNetPredictor(
        tile_step_size=args.step_size,
        use_gaussian=True,
        use_mirroring=not args.disable_tta,
        perform_everything_on_device=device.type == "cuda",
        device=device,
        verbose=args.verbose,
        allow_tqdm=args.verbose,
    )

    predictor.manual_initialization(
        network,
        plans_manager,
        configuration_manager,
        [checkpoint['network_weights']],
        dataset_json,
        nnUNetTrainerMoEAdapterUNetText.__name__,
        inference_allowed_axes,
    )

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    maybe_mkdir_p(output_dir)

    predictor.predict_from_files(
        str(input_dir),
        str(output_dir),
        save_probabilities=args.save_probabilities,
        overwrite=args.overwrite,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )


if __name__ == "__main__":
    main()
