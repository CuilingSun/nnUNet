"""Trainer for the three-encoder nnU-Net variant that mirrors PlainConvUNet."""

from __future__ import annotations

import os
from typing import Optional

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nets.MultiEncodernnUNet import build_multi_encoder_from_plans


class nnUNetTrainerMultiEncodernnUNet(nnUNetTrainer):
    """Drop-in trainer that swaps the backbone for the multi-encoder network."""

    ENV_PRETRAIN = "NNUNET_LEGACY_PRETRAIN"

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        # Ignore architecture_class_name because we always build our custom network
        net = build_multi_encoder_from_plans(
            self.plans_manager,
            self.dataset_json,
            self.configuration_manager,
            num_input_channels,
            deep_supervision=enable_deep_supervision,
        )
        return net

    def initialize(self):
        super().initialize()
        self._maybe_load_legacy_pretrain()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _maybe_load_legacy_pretrain(self) -> None:
        path = os.environ.get(self.ENV_PRETRAIN, "").strip()
        if not path:
            return
        if not os.path.isfile(path):
            self.print_to_log_file(
                f"[LEGACY-PRETRAIN] Ignored missing checkpoint: {path}")
            return

        try:
            ckpt = torch.load(path, map_location="cpu")
        except Exception as err:  # pragma: no cover - defensive logging
            self.print_to_log_file(
                f"[LEGACY-PRETRAIN] Failed to load checkpoint ({err})")
            return

        state = None
        if isinstance(ckpt, dict):
            for key in ("network_weights", "state_dict", "model_state_dict"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    state = ckpt[key]
                    break
        if state is None:
            if isinstance(ckpt, dict):
                state = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
            else:
                state = ckpt

        if not isinstance(state, dict):
            self.print_to_log_file(
                "[LEGACY-PRETRAIN] Checkpoint does not seem to contain a state dict; skipping.")
            return

        load_res = self.network.load_state_dict(state, strict=False)
        missing = getattr(load_res, "missing_keys", [])
        unexpected = getattr(load_res, "unexpected_keys", [])
        self.print_to_log_file(
            f"[LEGACY-PRETRAIN] loaded from={path} missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            self.print_to_log_file(f"[LEGACY-PRETRAIN] Missing example: {missing[:5]}")
        if unexpected:
            self.print_to_log_file(f"[LEGACY-PRETRAIN] Unexpected example: {unexpected[:5]}")
