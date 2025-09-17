#!/usr/bin/env bash
set -euo pipefail
# Wrapper to launch HighRecallROI with Dice+TopKCE
# Usage: bash tools/run_highrecall_dice.sh <GPU_ID> [FOLD] [DATASET] [CONFIG] [PLANS]
exec bash "$(dirname "$0")/run_highrecall_single.sh" "${1:?Need GPU}" dice "${2:-0}" "${3:-Dataset2202_picai_split}" "${4:-3d_fullres}" "${5:-nnUNetPlans}"

