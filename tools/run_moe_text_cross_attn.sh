#!/usr/bin/env bash
set -euo pipefail
# Launcher for MoE Text trainer with cross-attention + text adapter enabled.
# Mirrors tools/run_moe_text_tversky.sh but flips on the cross-attention head
# and defaults the decoder modulation to a gating adapter.
# Usage: bash tools/run_moe_text_cross_attn.sh <GPU_ID> [FOLD] [DATASET] [CONFIG] [PLANS]

export NNUNET_TRAINER="${NNUNET_TRAINER:-nnUNetTrainerMoEAdapterUNetText}"
export NNUNET_USE_CROSS_ATTN_FINAL="${NNUNET_USE_CROSS_ATTN_FINAL:-0}"
export NNUNET_USE_TEXT_ADAPTOR="${NNUNET_USE_TEXT_ADAPTOR:-0}"

exec bash "$(dirname "$0")/run_text_single.sh" \
  "${1:?Need GPU}" "${2:-0}" "${3:-Dataset2203_picai_split}" "${4:-3d_fullres}" "${5:-nnUNetPlans}" tversky
