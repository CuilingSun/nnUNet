#!/usr/bin/env bash
set -euo pipefail

# Attention-enabled Text trainer wrapper using Tversky+TopKCE loss.
# Usage: bash tools/run_text_attn_tversky.sh <GPU_ID> [FOLD] [DATASET] [CONFIG] [PLANS]

# Ensure the attention trainer is used by default (can be overridden externally).
export NNUNET_TRAINER="${NNUNET_TRAINER:-nnUNetTrainerMultiEncoderAttnUNetText}"

# Default cross-attention controls (override before calling if desired).
export NNUNET_USE_CROSS_ATTN_FINAL="${NNUNET_USE_CROSS_ATTN_FINAL:-1}"
export NNUNET_CROSS_TAU="${NNUNET_CROSS_TAU:-0.35}"
export NNUNET_CROSS_ALPHA="${NNUNET_CROSS_ALPHA:-0.25}"
export NNUNET_CROSS_GAMMA_INIT="${NNUNET_CROSS_GAMMA_INIT:-0.255}"
export ATTN_WARMUP_EPOCHS="${ATTN_WARMUP_EPOCHS:-5}"
export BASE_LR_REFINER="${BASE_LR_REFINER:-5e-4}"

exec bash "$(dirname "$0")/run_text_single.sh" "${1:?Need GPU}" "${2:-0}" "${3:-Dataset2203_picai_split}" "${4:-3d_fullres}" "${5:-nnUNetPlans}" tversky

# - 用法: bash tools/run_text_attn_tversky.sh GPU_ID [FOLD] [DATASET] [CONFIG] [PLANS]
# - 示例: QUICK=1 bash tools/run_text_attn_tversky.sh 4 0 Dataset2203_picai_split 3d_fullres nnUNetPlans
