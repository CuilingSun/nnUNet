#!/usr/bin/env bash
set -euo pipefail

# Wrapper around run_text_tversky_pretrained.sh that stores outputs under
# /data2/yyp4247/data/nnUNet_results/0921/umamba_pretrain.
# Usage mirrors the original script:
#   bash scripts/run_text_tversky_pretrained_0921.sh <GPU_ID> <DATASET> [FOLD] [CONFIG] [PLANS]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NNUNET_RESULTS_DIR="/data2/yyp4247/data/nnUNet_results/0921/umamba_pretrain"
export NNUNET_LEGACY_PRETRAIN="${NNUNET_LEGACY_PRETRAIN:-/data2/yyp4247/nnUNet/pretrained/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_final.pth}"
export NNUNET_TRAINER="${NNUNET_TRAINER:-nnUNetTrainerMultiEncodernnUNetText}"
export nnUNet_compile="${nnUNet_compile:-0}"

bash "${SCRIPT_DIR}/run_text_tversky_pretrained.sh" "$@"
