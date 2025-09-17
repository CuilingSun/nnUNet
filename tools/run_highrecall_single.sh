#!/usr/bin/env bash
set -euo pipefail

# Run a single HighRecallROI training with selectable loss (DiceTopKCE or TverskyTopKCE).
# Usage:
#   bash tools/run_highrecall_single.sh <GPU_ID> <dice|tversky> [FOLD] [DATASET] [CONFIG] [PLANS]
# Examples:
#   bash tools/run_highrecall_single.sh 0 dice     0 Dataset2202_picai_split 3d_fullres nnUNetPlans
#   bash tools/run_highrecall_single.sh 1 tversky  0 Dataset2202_picai_split 3d_fullres nnUNetPlans
# Env knobs (override as needed):
#   QUICK=1 (default) uses NNUNET_ITERS_PER_EPOCH=200, NNUNET_VAL_ITERS=100
#   QUICK=0 uses NNUNET_ITERS_PER_EPOCH=300, NNUNET_VAL_ITERS=200
#   nnUNet_raw/nnUNet_preprocessed/nnUNet_results: default to /data2/yyp4247/data/*
#   NNUNET_OVERSAMPLE_FG, NNUNET_POS_CASE_WEIGHT, NNUNET_CE_W_BG, NNUNET_CE_W_FG, nnUNet_n_proc_DA

GPU="${1:?Need GPU id}"
CHOICE="${2:?Need loss choice: dice|tversky}"
FOLD="${3:-0}"
DATASET="${4:-Dataset2202_picai_split}"
CONFIG="${5:-3d_fullres}"
PLANS="${6:-nnUNetPlans}"

# Default nnU-Net paths (overridable)
export nnUNet_raw="${nnUNet_raw:-/data2/yyp4247/data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/data2/yyp4247/data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-/data2/yyp4247/data/nnUNet_results}"

# Compile off by default for stability
export nnUNet_compile="${nnUNet_compile:-0}"

# ROI defaults (can be overridden by caller)
GLAND_ROI_DIR_DEFAULT="/data2/yyp4247/data/picai_labels/anatomical_delineations/whole_gland/AI/Bosma22b"
export NNUNET_ROI_DIR="${NNUNET_ROI_DIR:-$GLAND_ROI_DIR_DEFAULT}"
export NNUNET_ROI_MARGIN="${NNUNET_ROI_MARGIN:-8,64,64}"

# High-recall knobs (overridable)
export NNUNET_OVERSAMPLE_FG="${NNUNET_OVERSAMPLE_FG:-0.9}"
export NNUNET_POS_CASE_WEIGHT="${NNUNET_POS_CASE_WEIGHT:-3.0}"
export NNUNET_CE_W_BG="${NNUNET_CE_W_BG:-0.2}"
export NNUNET_CE_W_FG="${NNUNET_CE_W_FG:-0.8}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-1}"

# QUICK vs FULL
QUICK=${QUICK:-1}
if [[ "$QUICK" == "1" ]]; then
  export NNUNET_ITERS_PER_EPOCH=${NNUNET_ITERS_PER_EPOCH:-200}
  export NNUNET_VAL_ITERS=${NNUNET_VAL_ITERS:-100}
  MODE_STR="QUICK"
else
  export NNUNET_ITERS_PER_EPOCH=${NNUNET_ITERS_PER_EPOCH:-300}
  export NNUNET_VAL_ITERS=${NNUNET_VAL_ITERS:-200}
  MODE_STR="FULL"
fi

# Map choice -> trainer and tag
case "$CHOICE" in
  dice)
    TR="nnUNetTrainerHighRecallROI"
    TAG="DiceTopKCE"
    ;;
  tversky)
    TR="nnUNetTrainerHighRecallROI_Tversky"
    TAG="TverskyTopKCE"
    ;;
  *) echo "Unknown choice: $CHOICE (use dice|tversky)"; exit 2;;
esac

# Logging
timestamp() { date +"%Y%m%d_%H%M%S"; }
LOG_ROOT="${LOG_ROOT:-/data2/yyp4247/nnUNet/log}"
mkdir -p "$LOG_ROOT"
LOGDIR="${LOG_ROOT}/logs_${DATASET}_${CONFIG}_${TR}"
mkdir -p "$LOGDIR"
LOGFILE_TRAIN="${LOGDIR}/train_${TAG}_$(timestamp).log"

echo "[${TR}] GPU=${GPU} | MODE=${MODE_STR} | ROI=${NNUNET_ROI_DIR} | FOLD=${FOLD}"

# Launch within its own session so it can be stopped independently via PGID later
setsid bash -lc "\
  export CUDA_VISIBLE_DEVICES='$GPU'; \
  NNUNET_ITERS_PER_EPOCH=${NNUNET_ITERS_PER_EPOCH} \
  NNUNET_VAL_ITERS=${NNUNET_VAL_ITERS} \
  NNUNET_OVERSAMPLE_FG=${NNUNET_OVERSAMPLE_FG} \
  nnUNetv2_train '$DATASET' '$CONFIG' '$FOLD' -tr '$TR' -p '$PLANS' -num_gpus 1 \
" > "$LOGFILE_TRAIN" 2>&1 &

echo "Launched -> $LOGFILE_TRAIN"

