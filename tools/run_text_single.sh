#!/usr/bin/env bash
set -euo pipefail

# Run a single Text trainer (nnUNetTrainerMultiEncoderUNetText) job with convenient defaults.
# Usage:
#   bash tools/run_text_single.sh <GPU_ID> [FOLD] [DATASET] [CONFIG] [PLANS]
# Example:
#   bash tools/run_text_single.sh 4 0 Dataset2202_picai_split 3d_fullres nnUNetPlans
#
# Notes:
# - QUICK=1 (default): NNUNET_ITERS_PER_EPOCH=200, NNUNET_VAL_ITERS=100
# - QUICK=0:          NNUNET_ITERS_PER_EPOCH=300, NNUNET_VAL_ITERS=200
# - You can override any NNUNET_* env var before calling this script.

GPU="${1:?Need GPU id}"
FOLD="${2:-0}"
DATASET="${3:-Dataset2202_picai_split}"
CONFIG="${4:-3d_fullres}"
PLANS="${5:-nnUNetPlans}"
# Optional 6th arg: loss kind for Text trainer: dice|tversky|focal_tversky
LOSS_KIND_ARG="${6:-}"

# Normalize/select NNUNET_TEXT_LOSS value (trainer expects: dice_topk|tversky_topk|focal_tversky_topk)
LOSS_ENV=""
if [[ -n "${LOSS_KIND_ARG}" ]]; then
  # lower-case for matching
  case "${LOSS_KIND_ARG,,}" in
    dice)
      LOSS_ENV="dice_topk" ;;
    tversky)
      LOSS_ENV="tversky_topk" ;;
    focal_tversky|ftversky|focal-tversky)
      LOSS_ENV="focal_tversky_topk" ;;
    *)
      echo "Unknown loss kind: ${LOSS_KIND_ARG} (use dice|tversky|focal_tversky)" >&2
      exit 2 ;;
  esac
fi

# Default nnU-Net paths (overridable). Allow redirecting results to a separate folder to avoid clashes.
export nnUNet_raw="${nnUNet_raw:-/data2/yyp4247/data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/data2/yyp4247/data/nnUNet_preprocessed}"
BASE_NNUNET_RESULTS="${nnUNet_results:-/data2/yyp4247/data/nnUNet_results}"
# If NNUNET_RESULTS_DIR is set, use it directly; else if NNUNET_RESULTS_TAG is set, append to base under experiments/<TAG>
if [[ -n "${NNUNET_RESULTS_DIR:-}" ]]; then
  export nnUNet_results="${NNUNET_RESULTS_DIR}"
elif [[ -n "${NNUNET_RESULTS_TAG:-}" ]]; then
  export nnUNet_results="${BASE_NNUNET_RESULTS}/experiments/${NNUNET_RESULTS_TAG}"
else
  export nnUNet_results="${BASE_NNUNET_RESULTS}"
fi

# Turn off compile by default for stability
export nnUNet_compile="${nnUNet_compile:-0}"

# QUICK / FULL defaults
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

# Text defaults (overridable): prompt/model + aux losses with curriculum
export NNUNET_TEXT_PROMPTS="${NNUNET_TEXT_PROMPTS:-prostate lesion}"
export NNUNET_TEXT_MODEL="${NNUNET_TEXT_MODEL:-hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224}"
export NNUNET_TEXT_EMBED_DIM="${NNUNET_TEXT_EMBED_DIM:-512}"
export NNUNET_TEXT_MODULATION="${NNUNET_TEXT_MODULATION:-none}"   # none|film|gate
export NNUNET_USE_ALIGNMENT_HEAD="${NNUNET_USE_ALIGNMENT_HEAD:-1}"
export NNUNET_RETURN_HEATMAP="${NNUNET_RETURN_HEATMAP:-1}"
export NNUNET_LAMBDA_ALIGN="${NNUNET_LAMBDA_ALIGN:-0.2}"
export NNUNET_LAMBDA_HEAT="${NNUNET_LAMBDA_HEAT:-0.5}"
export NNUNET_AUX_WARMUP_EPOCHS="${NNUNET_AUX_WARMUP_EPOCHS:-80}"
export NNUNET_AUX_RAMP_EPOCHS="${NNUNET_AUX_RAMP_EPOCHS:-50}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export NNUNET_SKIP_OPENCLIP="${NNUNET_SKIP_OPENCLIP:-0}"
# Disable saving heatmaps by default (can override externally if needed)
export NNUNET_SAVE_HEATMAPS="${NNUNET_SAVE_HEATMAPS:-0}"
export NNUNET_EXPORT_FULL_HEATMAP="${NNUNET_EXPORT_FULL_HEATMAP:-0}"
export NNUNET_EXPORT_HEATMAP_NIFTI="${NNUNET_EXPORT_HEATMAP_NIFTI:-0}"
export NNUNET_HEATMAP_DIRNAME="${NNUNET_HEATMAP_DIRNAME:-validation_heatmaps}"

# Precompute dataset/plans paths for evaluation & postprocess
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

# Logging
timestamp() { date +"%Y%m%d_%H%M%S"; }
LOG_ROOT="${LOG_ROOT:-/data2/yyp4247/nnUNet/log}"
mkdir -p "$LOG_ROOT"
LOGDIR="${LOG_ROOT}/logs_${DATASET}_${CONFIG}_nnUNetTrainerMultiEncoderUNetText"
mkdir -p "$LOGDIR"
LOGFILE_TRAIN="${LOGDIR}/train_TextTopKCE_$(timestamp).log"
LOGFILE_VAL="${LOGDIR}/val_TextTopKCE_$(timestamp).log"

echo "[nnUNetTrainerMultiEncoderUNetText] GPU=${GPU} | MODE=${MODE_STR} | FOLD=${FOLD} | PROMPTS='${NNUNET_TEXT_PROMPTS}'"

# Compute output folder for later evaluation
OUTFOLD=$($PY_GETS - "$DATASET" "nnUNetTrainerMultiEncoderUNetText" "$PLANS" "$CONFIG" "$FOLD" <<'PY'
from nnunetv2.utilities.file_path_utilities import get_output_folder
import os,sys
print(get_output_folder(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], fold=int(sys.argv[5])))
PY
)
VALFOLDER="${OUTFOLD}/validation"

# Postprocess sweep settings
SWEEP_LIST="${SWEEP_LIST:-0 10 20 50 100}"
KEEP_LARGEST="${KEEP_LARGEST:-1}"

# Launch as independent session for safe stopping later: train -> val_best -> eval -> postprocess sweeps
setsid bash -c "\
  set -euo pipefail; \
  export CUDA_VISIBLE_DEVICES='$GPU'; \
  ${LOSS_ENV:+export NNUNET_TEXT_LOSS='${LOSS_ENV}'; } \
  NNUNET_ITERS_PER_EPOCH=${NNUNET_ITERS_PER_EPOCH} \
  NNUNET_VAL_ITERS=${NNUNET_VAL_ITERS} \
  nnUNetv2_train '$DATASET' '$CONFIG' '$FOLD' -tr nnUNetTrainerMultiEncoderUNetText -p '$PLANS' -num_gpus 1 2>&1 | tee -a "$LOGFILE_TRAIN"; \
  if [ -f \"$OUTFOLD/checkpoint_best.pth\" ]; then \
    nnUNetv2_train '$DATASET' '$CONFIG' '$FOLD' -tr nnUNetTrainerMultiEncoderUNetText -p '$PLANS' --val --val_best 2>&1 | tee -a "$LOGFILE_VAL"; \
  else \
    echo \"[run_text_single] Skip validation: no checkpoint_best.pth yet in $OUTFOLD\" | tee -a \"$LOGFILE_VAL\"; \
  fi; \
  if [ -d \"$VALFOLDER\" ]; then \
    nnUNetv2_evaluate_folder \"$GTFOLDER\" \"$VALFOLDER\" -djfile \"$DJFILE\" -pfile \"$PFILE\" -o \"$VALFOLDER/summary.json\" 2>&1 | tee -a \"$LOGFILE_VAL\"; \
    for MV in $SWEEP_LIST; do \
      OUTPP=\"${VALFOLDER}_pp_min\${MV}\"; \
      if [ \"$KEEP_LARGEST\" = \"1\" ]; then OUTPP=\"\${OUTPP}_lcc\"; fi; \
      mkdir -p \"\$OUTPP\"; \
      python tools/postprocess_cc.py --in \"$VALFOLDER\" --out \"\$OUTPP\" --labels 1 --min-voxels \"\${MV}\" $( [ \"$KEEP_LARGEST\" = \"1\" ] && echo --keep-largest ); \
      nnUNetv2_evaluate_folder \"$GTFOLDER\" \"\$OUTPP\" -djfile \"$DJFILE\" -pfile \"$PFILE\" -o \"\$OUTPP/summary.json\" 2>&1 | tee -a \"$LOGFILE_VAL\"; \
    done; \
  else \
    echo \"[run_text_single] Skip evaluation/postprocess: validation folder not found: $VALFOLDER\" | tee -a \"$LOGFILE_VAL\"; \
  fi \
" >/dev/null 2>&1 &

echo "Launched -> $LOGFILE_TRAIN (and $LOGFILE_VAL). Output: $OUTFOLD"
