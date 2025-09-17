#!/usr/bin/env bash
set -euo pipefail

# Quick launcher to train and validate 1 fold with three high-recall trainers,
# then run a small postprocessing sweep and evaluation.
#
# Usage:
#   bash tools/run_picai_highrecall_compare.sh [FOLD] [DATASET] [CONFIG]
# Defaults:
#   FOLD=0, DATASET=Dataset2202_picai_split, CONFIG=3d_fullres
#
# Notes:
# - ROI (prostate gland) mask directory defaults to the path you provided.
# - ROI margin defaults to 8,64,64 (z,y,x) in preprocessed voxels.
# - Adjust CE weights / LR / batch size via env exports below if desired.

# Default nnU-Net paths (can be overridden by pre-set env)
export nnUNet_raw="${nnUNet_raw:-/data2/yyp4247/data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/data2/yyp4247/data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-/data2/yyp4247/data/nnUNet_results}"

# Disable torch.compile (Inductor) unless explicitly enabled to avoid Dynamo/Inductor crashes
export nnUNet_compile="${nnUNet_compile:-0}"

FOLD="${1:-0}"
DATASET="${2:-Dataset2202_picai_split}"
CONFIG="${3:-3d_fullres}"
PLANS="${PLANS:-nnUNetPlans}"

# Prostate gland ROI directory (fallback if NNUNET_ROI_DIR not set)
GLAND_ROI_DIR_DEFAULT="/data2/yyp4247/data/picai_labels/anatomical_delineations/whole_gland/AI/Bosma22b"

# Default training knobs (can be overridden before calling this script)
export NNUNET_OVERSAMPLE_FG="${NNUNET_OVERSAMPLE_FG:-0.9}"
export NNUNET_POS_CASE_WEIGHT="${NNUNET_POS_CASE_WEIGHT:-3.0}"
export NNUNET_CE_W_BG="${NNUNET_CE_W_BG:-0.2}"
export NNUNET_CE_W_FG="${NNUNET_CE_W_FG:-0.8}"
export NNUNET_ROI_DIR="${NNUNET_ROI_DIR:-$GLAND_ROI_DIR_DEFAULT}"
export NNUNET_ROI_MARGIN="${NNUNET_ROI_MARGIN:-8,64,64}"

# Optional stabilizers for small batch sizes (uncomment if needed)
# export NNUNET_LR=${NNUNET_LR:-0.0015}
# export NNUNET_BATCH_SIZE=${NNUNET_BATCH_SIZE:-2}

TRAINERS=(
  nnUNetTrainerHighRecallROI
  nnUNetTrainerHighRecallROI_Tversky
  nnUNetTrainerHighRecallROI_FocalTversky
)

timestamp() { date +"%Y%m%d_%H%M%S"; }

PY_GETS=python3

# Resolve important paths via nnU-Net Python API
PP_BASE=$($PY_GETS - "$DATASET" <<'PY'
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
import os,sys
print(os.path.join(nnUNet_preprocessed, maybe_convert_to_dataset_name(sys.argv[1])))
PY
)

DJFILE="${PP_BASE}/dataset.json"
PFILE="${PP_BASE}/${PLANS}.json"
GTFOLDER="${PP_BASE}/gt_segmentations"

if [[ ! -f "$DJFILE" ]]; then echo "dataset.json not found: $DJFILE"; exit 1; fi
if [[ ! -f "$PFILE" ]]; then echo "plans file not found: $PFILE"; exit 1; fi
if [[ ! -d "$GTFOLDER" ]]; then echo "gt_segmentations not found: $GTFOLDER"; exit 1; fi

echo "Using ROI dir: ${NNUNET_ROI_DIR} | ROI margin: ${NNUNET_ROI_MARGIN}"

# Postprocessing sweep settings (min-voxels). 0 means no size filtering, only keep-largest if enabled.
SWEEP_MIN_VOXELS=(0 10 20 50 100)
KEEP_LARGEST=1

LOG_ROOT="${LOG_ROOT:-.}"
mkdir -p "$LOG_ROOT"
for TR in "${TRAINERS[@]}"; do
  echo "=== Training with ${TR} (fold ${FOLD}, ${DATASET}/${CONFIG}) ==="
  LOGDIR="${LOG_ROOT}/logs_${DATASET}_${CONFIG}_${TR}"
  mkdir -p "$LOGDIR"
  case "$TR" in
    nnUNetTrainerHighRecallROI) LOSS_TAG="DiceTopKCE";;
    nnUNetTrainerHighRecallROI_Tversky) LOSS_TAG="TverskyTopKCE";;
    nnUNetTrainerHighRecallROI_FocalTversky) LOSS_TAG="FocalTverskyTopKCE";;
    *) LOSS_TAG="$TR";;
  esac

  # Train (single GPU by default; adjust with CUDA_VISIBLE_DEVICES / -num_gpus)
  nnUNetv2_train "$DATASET" "$CONFIG" "$FOLD" -tr "$TR" -p "$PLANS" -num_gpus "${NUM_GPUS:-1}" 2>&1 | tee -a "$LOGDIR/train_${LOSS_TAG}_$(timestamp).log"

  # Export validation predictions using best checkpoint
  nnUNetv2_train "$DATASET" "$CONFIG" "$FOLD" -tr "$TR" -p "$PLANS" --val --val_best 2>&1 | tee -a "$LOGDIR/val_${LOSS_TAG}_$(timestamp).log"

  # Locate model output and validation folder via Python helper
  OUTFOLD=$($PY_GETS - "$DATASET" "$TR" "$PLANS" "$CONFIG" "$FOLD" <<'PY'
from nnunetv2.utilities.file_path_utilities import get_output_folder
import os,sys
print(get_output_folder(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], fold=int(sys.argv[5])))
PY
  )

  VALFOLDER="${OUTFOLD}/validation"
  if [[ ! -d "$VALFOLDER" ]]; then echo "Validation folder not found: $VALFOLDER"; exit 1; fi

  echo "Validation outputs in: $VALFOLDER"
  # Baseline evaluation (no postprocess)
  nnUNetv2_evaluate_folder "$GTFOLDER" "$VALFOLDER" -djfile "$DJFILE" -pfile "$PFILE" -o "$VALFOLDER/summary.json"

  # Postprocess sweep
  for MV in "${SWEEP_MIN_VOXELS[@]}"; do
    OUTPP="${VALFOLDER}_pp_min${MV}$( [[ "$KEEP_LARGEST" == "1" ]] && echo _lcc )"
    mkdir -p "$OUTPP"
    python tools/postprocess_cc.py --in "$VALFOLDER" --out "$OUTPP" --labels 1 --min-voxels "$MV" $( [[ "$KEEP_LARGEST" == "1" ]] && echo --keep-largest )
    nnUNetv2_evaluate_folder "$GTFOLDER" "$OUTPP" -djfile "$DJFILE" -pfile "$PFILE" -o "$OUTPP/summary.json"
  done

done

echo "All trainings, validations, and sweeps submitted. Check nnUNet_results and validation folders for outputs."
