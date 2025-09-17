#!/usr/bin/env python
import os
import sys
import json
import torch
import numpy as np

from batchgenerators.utilities.file_and_folder_operations import join, isfile

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.nets.MultiEncoderUNet import (
    get_multi_encoder_unet_2d_from_plans,
    get_multi_encoder_unet_3d_from_plans,
)


def main():
    # inputs
    dataset_id_or_name = sys.argv[1] if len(sys.argv) > 1 else "Dataset2201_picai"
    configuration = sys.argv[2] if len(sys.argv) > 2 else "3d_fullres"
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    # env paths
    nnUNet_raw = os.environ.get("nnUNet_raw")
    nnUNet_preproc = os.environ.get("nnUNet_preprocessed")
    if not nnUNet_raw or not nnUNet_preproc:
        print("ERROR: nnUNet_raw or nnUNet_preprocessed not set.")
        sys.exit(1)

    # resolve dataset name
    ds_name = maybe_convert_to_dataset_name(dataset_id_or_name)

    # load dataset.json (raw) and plans (preprocessed)
    dataset_json_file = join(nnUNet_raw, ds_name, "dataset.json")
    plans_file = join(nnUNet_preproc, ds_name, "nnUNetPlans.json")
    if not isfile(dataset_json_file):
        print(f"ERROR: dataset.json not found: {dataset_json_file}")
        sys.exit(2)
    if not isfile(plans_file):
        print(f"ERROR: nnUNetPlans.json not found: {plans_file}")
        sys.exit(3)

    with open(dataset_json_file, "r") as f:
        dataset_json = json.load(f)
    plans_manager = PlansManager(plans_file)
    configuration_manager = plans_manager.get_configuration(configuration)

    # build network
    dim = len(configuration_manager.patch_size)
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    deep_supervision = True
    if dim == 2:
        net = get_multi_encoder_unet_2d_from_plans(
            plans_manager, dataset_json, configuration_manager, num_input_channels, deep_supervision
        )
    elif dim == 3:
        net = get_multi_encoder_unet_3d_from_plans(
            plans_manager, dataset_json, configuration_manager, num_input_channels, deep_supervision
        )
    else:
        print(f"ERROR: Unsupported dimensionality: {dim}")
        sys.exit(4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device).eval()

    # dummy input
    patch_size = configuration_manager.patch_size
    x = torch.randn((batch_size, num_input_channels, *patch_size), dtype=torch.float32, device=device)

    # optional text input depending on modulation and text head configuration
    text = None
    try:
        # peek into decoder config via attributes
        text_dim = getattr(net, "text_embed_dim", None)
        modulation = getattr(net, "modulation", "none")
    except Exception:
        text_dim = None
        modulation = "none"

    if text_dim is not None and modulation in ("film", "gate"):
        # honor env setting NNUNET_TEXT_EMBED_DIM if present, else use model's
        d_env = os.environ.get("NNUNET_TEXT_EMBED_DIM")
        d = int(d_env) if d_env is not None else int(text_dim)
        text = torch.randn((batch_size, d), dtype=torch.float32, device=device)

    # forward
    with torch.no_grad():
        try:
            out = net(x, text=text, return_extra=True)
        except TypeError:
            out = net(x)

    if isinstance(out, dict):
        seg = out.get("seg")
        sim = out.get("sim")
    else:
        seg = out
        sim = None

    # summarize
    if isinstance(seg, (list, tuple)):
        print(f"OK: got {len(seg)} deep supervision outputs; top shape: {tuple(seg[0].shape)}")
    else:
        print(f"OK: got logits shape: {tuple(seg.shape)}")
    if sim is not None:
        print(f"OK: got similarity map shape: {tuple(sim.shape)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

