#!/usr/bin/env bash
set -euo pipefail

# Run three high-recall trainers in parallel on three GPUs, each end-to-end:
# training -> export validation (best ckpt) -> baseline eval -> postprocess sweep + eval.
#
# Usage:
#   bash tools/run_picai_highrecall_compare_parallel.sh GPU0 GPU1 GPU2 [FOLD] [DATASET] [CONFIG]
# Example:
#   bash tools/run_picai_highrecall_compare_parallel.sh 0 1 2 0 Dataset2202_picai_split 3d_fullres
#
# Notes:
# - Ensure your CPU有足够的核数。建议下调每进程DA进程数： nnUNet_n_proc_DA=8 (或更低)。
# - 每个子进程内使用 -num_gpus 1，并通过 CUDA_VISIBLE_DEVICES 绑定到单卡。

# Default nnU-Net paths (can be overridden by pre-set env)
export nnUNet_raw="${nnUNet_raw:-/data2/yyp4247/data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/data2/yyp4247/data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-/data2/yyp4247/data/nnUNet_results}"

# Disable torch.compile (Inductor) unless explicitly enabled to avoid Dynamo/Inductor crashes
export nnUNet_compile="${nnUNet_compile:-0}"

GPU0="${1:?Need GPU id for trainer A}"
GPU1="${2:?Need GPU id for trainer B}"
GPU2="${3:?Need GPU id for trainer C}"
FOLD="${4:-0}"
DATASET="${5:-Dataset2202_picai_split}"
CONFIG="${6:-3d_fullres}"
PLANS="${PLANS:-nnUNetPlans}"

# Default ROI dir and margin (can be overridden via env before calling)
GLAND_ROI_DIR_DEFAULT="/data2/yyp4247/data/picai_labels/anatomical_delineations/whole_gland/AI/Bosma22b"
export NNUNET_ROI_DIR="${NNUNET_ROI_DIR:-$GLAND_ROI_DIR_DEFAULT}"
export NNUNET_ROI_MARGIN="${NNUNET_ROI_MARGIN:-8,64,64}"

# Default high-recall knobs (overridable)
export NNUNET_OVERSAMPLE_FG="${NNUNET_OVERSAMPLE_FG:-0.9}"
export NNUNET_POS_CASE_WEIGHT="${NNUNET_POS_CASE_WEIGHT:-3.0}"
export NNUNET_CE_W_BG="${NNUNET_CE_W_BG:-0.2}"
export NNUNET_CE_W_FG="${NNUNET_CE_W_FG:-0.8}"

# Optional stabilizers for small batch
# export NNUNET_LR=${NNUNET_LR:-0.0015}
# export NNUNET_BATCH_SIZE=${NNUNET_BATCH_SIZE:-2}

# DA processes per GPU to avoid CPU争用（可按CPU核数调整）
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-4}"

TRAINERS=(
  nnUNetTrainerHighRecallROI
  nnUNetTrainerHighRecallROI_Tversky
  nnUNetTrainerHighRecallROI_FocalTversky
)
GPUS=("$GPU0" "$GPU1" "$GPU2")

SWEEP_MIN_VOXELS=(0 10 20 50 100)
KEEP_LARGEST=1

timestamp() { date +"%Y%m%d_%H%M%S"; }

PY_GETS=python3
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
[[ -f "$DJFILE" ]] || { echo "dataset.json not found: $DJFILE"; exit 1; }
[[ -f "$PFILE" ]] || { echo "plans not found: $PFILE"; exit 1; }
[[ -d "$GTFOLDER" ]] || { echo "gt_segmentations not found: $GTFOLDER"; exit 1; }

run_one() {
  local TR="$1" GPU="$2"
  local LOG_ROOT="${LOG_ROOT:-.}"
  mkdir -p "$LOG_ROOT"
  local LOGDIR="${LOG_ROOT}/logs_${DATASET}_${CONFIG}_${TR}"
  mkdir -p "$LOGDIR"
  local LOSS_TAG
  case "$TR" in
    nnUNetTrainerHighRecallROI) LOSS_TAG="DiceTopKCE";;
    nnUNetTrainerHighRecallROI_Tversky) LOSS_TAG="TverskyTopKCE";;
    nnUNetTrainerHighRecallROI_FocalTversky) LOSS_TAG="FocalTverskyTopKCE";;
    *) LOSS_TAG="$TR";;
  esac
  echo "[${TR}] Using GPU ${GPU}, ROI=${NNUNET_ROI_DIR}, MARGIN=${NNUNET_ROI_MARGIN}"

  (
    export CUDA_VISIBLE_DEVICES="$GPU"
    nnUNetv2_train "$DATASET" "$CONFIG" "$FOLD" -tr "$TR" -p "$PLANS" -num_gpus 1 2>&1 | tee -a "$LOGDIR/train_${LOSS_TAG}_$(timestamp).log"
    nnUNetv2_train "$DATASET" "$CONFIG" "$FOLD" -tr "$TR" -p "$PLANS" --val --val_best 2>&1 | tee -a "$LOGDIR/val_${LOSS_TAG}_$(timestamp).log"

    local OUTFOLD
    OUTFOLD=$($PY_GETS - "$DATASET" "$TR" "$PLANS" "$CONFIG" "$FOLD" <<'PY'
from nnunetv2.utilities.file_path_utilities import get_output_folder
import os,sys
print(get_output_folder(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], fold=int(sys.argv[5])))
PY
    )

    local VALFOLDER="${OUTFOLD}/validation"
    nnUNetv2_evaluate_folder "$GTFOLDER" "$VALFOLDER" -djfile "$DJFILE" -pfile "$PFILE" -o "$VALFOLDER/summary.json"

    for MV in "${SWEEP_MIN_VOXELS[@]}"; do
      local OUTPP="${VALFOLDER}_pp_min${MV}$( [[ "$KEEP_LARGEST" == "1" ]] && echo _lcc )"
      mkdir -p "$OUTPP"
      python tools/postprocess_cc.py --in "$VALFOLDER" --out "$OUTPP" --labels 1 --min-voxels "$MV" $( [[ "$KEEP_LARGEST" == "1" ]] && echo --keep-largest )
      nnUNetv2_evaluate_folder "$GTFOLDER" "$OUTPP" -djfile "$DJFILE" -pfile "$PFILE" -o "$OUTPP/summary.json"
    done
  ) &
}

for i in 0 1 2; do
  run_one "${TRAINERS[$i]}" "${GPUS[$i]}"
done

echo "Launched 3 parallel trainings on GPUs: ${GPU0}, ${GPU1}, ${GPU2}. Waiting..."
wait
echo "All done. Check nnUNet_results and logs_*/ for outputs."
