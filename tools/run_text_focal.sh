#!/usr/bin/env bash
set -euo pipefail
# Wrapper to launch Text trainer with Focal Tversky+TopKCE
# Usage: bash tools/run_text_focal.sh <GPU_ID> [FOLD] [DATASET] [CONFIG] [PLANS]
exec bash "$(dirname "$0")/run_text_single.sh" "${1:?Need GPU}" "${2:-0}" "${3:-Dataset2202_picai_split}" "${4:-3d_fullres}" "${5:-nnUNetPlans}" focal_tversky

#    - 用法: bash tools/run_text_focal.sh GPU_ID [FOLD] [DATASET] [CONFIG] [PLANS]
#    - 示例: QUICK=1 NNUNET_TVERSKY_ALPHA=0.25 NNUNET_TVERSKY_BETA=0.75 NNUNET_FTVERSKY_GAMMA=1.33 bash tools/run_text_focal.sh 5 0