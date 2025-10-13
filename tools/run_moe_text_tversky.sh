#!/usr/bin/env bash
set -euo pipefail
# Wrapper to launch the MoE text trainer with Tversky+TopKCE loss.
# Usage: bash tools/run_moe_text_tversky.sh <GPU_ID> [FOLD] [DATASET] [CONFIG] [PLANS]
# Example: QUICK=1 bash tools/run_moe_text_tversky.sh 0 0 Dataset2203_picai_split 3d_fullres nnUNetPlans

export NNUNET_TRAINER="${NNUNET_TRAINER:-nnUNetTrainerMoEAdapterUNetText}"
exec bash "$(dirname "$0")/run_text_single.sh" "${1:?Need GPU}" "${2:-0}" "${3:-Dataset2203_picai_split}" "${4:-3d_fullres}" "${5:-nnUNetPlans}" tversky

# - Env overrides (optional):
#   NNUNET_TEXT_PROMPTS, NNUNET_TEXT_EMBED_DIM, NNUNET_TEXT_MODULATION, NNUNET_LAMBDA_ALIGN, NNUNET_LAMBDA_HEAT, ...
#   See tools/run_text_single.sh for the full list.
