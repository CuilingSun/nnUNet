#!/usr/bin/env bash
set -euo pipefail
# Wrapper to launch Text trainer with Dice+TopKCE (default)
# Usage: bash tools/run_text_dice.sh <GPU_ID> [FOLD] [DATASET] [CONFIG] [PLANS]
exec bash "$(dirname "$0")/run_text_single.sh" "${1:?Need GPU}" "${2:-0}" "${3:-Dataset2202_picai_split}" "${4:-3d_fullres}" "${5:-nnUNetPlans}"

    # - 用法: bash tools/run_text_dice.sh GPU_ID [FOLD] [DATASET] [CONFIG] [PLANS]
    # - 示例: QUICK=1 bash tools/run_text_dice.sh 4 0 Dataset2202_picai_split 3d_fullres nnUNetPlans