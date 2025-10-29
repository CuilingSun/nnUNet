#!/usr/bin/env bash
set -euo pipefail

# Run a single Text trainer (nnUNetTrainerMultiEncoderUNetText) job with convenient defaults.
# Usage:
#   bash tools/run_text_single.sh <GPU_ID> [FOLD] [DATASET] [CONFIG] [PLANS]
# Optional env: NNUNET_PRETRAINED_WEIGHTS=/path/to/checkpoint.pth to finetune
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
TRAINER="${NNUNET_TRAINER:-nnUNetTrainerMultiEncoderUNetText}"

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
PY_GETS=${PY_GETS:-$(command -v python)}
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
LOGDIR="${LOG_ROOT}/logs_${DATASET}_${CONFIG}_${TRAINER}"
mkdir -p "$LOGDIR"
LOGFILE_TRAIN="${LOGDIR}/train_TextTopKCE_$(timestamp).log"
LOGFILE_VAL="${LOGDIR}/val_TextTopKCE_$(timestamp).log"

echo "[${TRAINER}] GPU=${GPU} | MODE=${MODE_STR} | FOLD=${FOLD} | PROMPTS='${NNUNET_TEXT_PROMPTS}'"

# Compute output folder for later evaluation
OUTFOLD=$($PY_GETS - "$DATASET" "$TRAINER" "$PLANS" "$CONFIG" "$FOLD" <<'PY'
from nnunetv2.utilities.file_path_utilities import get_output_folder
import os,sys
print(get_output_folder(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], fold=int(sys.argv[5])))
PY
)
VALFOLDER_BASE="${OUTFOLD}/validation"
VALFOLDER_FINAL="${OUTFOLD}/validation_final"
VALFOLDER_BEST="${OUTFOLD}/validation_best"

# Postprocess sweep settings
SWEEP_LIST="${SWEEP_LIST:-0 10 20 50 100}"
KEEP_LARGEST="${KEEP_LARGEST:-1}"

PRETRAINED_WEIGHTS="${NNUNET_PRETRAINED_WEIGHTS:-}" # optional: fine-tune from checkpoint
PRETRAINED_ARG=""
if [[ -n "${PRETRAINED_WEIGHTS}" ]]; then
  PRETRAINED_ARG="-pretrained_weights '${PRETRAINED_WEIGHTS}'"
fi

# Build the command block once
CMD_BLOCK="set +u; set -eo pipefail; \
  export CUDA_VISIBLE_DEVICES='$GPU'; \
  ${LOSS_ENV:+export NNUNET_TEXT_LOSS='${LOSS_ENV}'; } \
  NNUNET_ITERS_PER_EPOCH=${NNUNET_ITERS_PER_EPOCH}; \
  NNUNET_VAL_ITERS=${NNUNET_VAL_ITERS}; \
  if [ -n \"${PRETRAINED_WEIGHTS}\" ]; then echo \"[run_text_single] Pretrained checkpoint: ${PRETRAINED_WEIGHTS}\" | tee -a \"$LOGFILE_TRAIN\"; fi; \
  nnUNetv2_train '$DATASET' '$CONFIG' '$FOLD' -tr '$TRAINER' -p '$PLANS' -num_gpus 1 ${PRETRAINED_ARG} 2>&1 | tee -a \"$LOGFILE_TRAIN\"; \
  eval_and_postprocess() { \
    local folder=\"\$1\"; \
    local tag=\"\$2\"; \
    if [ -d \"\$folder\" ]; then \
      echo \"[run_text_single] Evaluating \${tag} (\${folder})\" | tee -a \"$LOGFILE_VAL\"; \
      nnUNetv2_evaluate_folder \"$GTFOLDER\" \"\$folder\" -djfile \"$DJFILE\" -pfile \"$PFILE\" -o \"\$folder/summary.json\" 2>&1 | tee -a \"$LOGFILE_VAL\"; \
      for MV in $SWEEP_LIST; do \
        OUTPP=\"\${folder}_pp_min\${MV}\"; \
        if [ \"$KEEP_LARGEST\" = \"1\" ]; then OUTPP=\"\${OUTPP}_lcc\"; fi; \
        mkdir -p \"\$OUTPP\"; \
        EXTRA_ARG=''; \
        if [ \"$KEEP_LARGEST\" = \"1\" ]; then EXTRA_ARG='--keep-largest'; fi; \
        \"${PY_GETS}\" tools/postprocess_cc.py --in \"\$folder\" --out \"\$OUTPP\" --labels 1 --min-voxels \"\${MV}\" \$EXTRA_ARG; \
        nnUNetv2_evaluate_folder \"$GTFOLDER\" \"\$OUTPP\" -djfile \"$DJFILE\" -pfile \"$PFILE\" -o \"\$OUTPP/summary.json\" 2>&1 | tee -a \"$LOGFILE_VAL\"; \
      done; \
    else \
      echo \"[run_text_single] Skip evaluation/postprocess: folder not found: \$folder\" | tee -a \"$LOGFILE_VAL\"; \
    fi; \
  }; \
  if [ -f \"$OUTFOLD/checkpoint_best.pth\" ]; then \
    echo \"[run_text_single] Validation (final checkpoint)\" | tee -a \"$LOGFILE_VAL\"; \
    nnUNetv2_train '$DATASET' '$CONFIG' '$FOLD' -tr '$TRAINER' -p '$PLANS' --val 2>&1 | tee -a \"$LOGFILE_VAL\"; \
    if [ -d \"$VALFOLDER_BASE\" ]; then \"${PY_GETS}\" - <<'PY' "$VALFOLDER_BASE" "$VALFOLDER_FINAL"; \
import os, shutil, sys
src, dst = sys.argv[1:3]
if os.path.isdir(dst):
    shutil.rmtree(dst)
shutil.move(src, dst)
PY
    fi; \
    eval_and_postprocess \"$VALFOLDER_FINAL\" \"validation_final\"; \
    if [ -f \"$VALFOLDER_FINAL/summary.json\" ]; then mv \"$VALFOLDER_FINAL/summary.json\" \"$VALFOLDER_FINAL/summary_last.json\"; fi; \
    echo \"[run_text_single] Validation (best checkpoint: $OUTFOLD/checkpoint_best.pth)\" | tee -a \"$LOGFILE_VAL\"; \
    nnUNetv2_train '$DATASET' '$CONFIG' '$FOLD' -tr '$TRAINER' -p '$PLANS' --val --val_best 2>&1 | tee -a \"$LOGFILE_VAL\"; \
    if [ -d \"$VALFOLDER_BASE\" ]; then \"${PY_GETS}\" - <<'PY' "$VALFOLDER_BASE" "$VALFOLDER_BEST"; \
import os, shutil, sys
src, dst = sys.argv[1:3]
if os.path.isdir(dst):
    shutil.rmtree(dst)
shutil.move(src, dst)
PY
    fi; \
    eval_and_postprocess \"$VALFOLDER_BEST\" \"validation_best\"; \
    if [ -f \"$VALFOLDER_BEST/summary.json\" ]; then mv \"$VALFOLDER_BEST/summary.json\" \"$VALFOLDER_BEST/summary_best.json\"; fi; \
  else \
    echo \"[run_text_single] Skip validation: no checkpoint_best.pth yet in $OUTFOLD\" | tee -a \"$LOGFILE_VAL\"; \
  fi"

# Always run in foreground so logs stream to terminal and Ctrl+C works
bash -c "$CMD_BLOCK"
