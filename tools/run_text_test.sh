#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <input_dir> <output_dir> [GPU_ID] [MODEL_PATH]" >&2
  exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
GPU_ID="${3:-5}"
MODEL_SOURCE="${4:-}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "nnunetv2" ]]; then
  if command -v conda >/dev/null 2>&1; then
    set +u
    eval "$(conda shell.bash hook)"
    conda activate nnunetv2 >/dev/null 2>&1 || {
      echo "[ERROR] Failed to activate nnunetv2 environment." >&2
      exit 1
    }
    set -u
    echo "[INFO] Activated environment nnunetv2"
  else
    echo "[ERROR] Please activate the nnunetv2 conda environment before running this script." >&2
    exit 1
  fi
else
  echo "[INFO] Environment nnunetv2 already active"
fi

echo "[INFO] Using CUDA_VISIBLE_DEVICES=${GPU_ID}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

export nnUNet_raw="${nnUNet_raw:-/data2/yyp4247/data/nnUNet_data_filtered/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/data2/yyp4247/data/nnUNet_data_filtered/nnUNet_preprocessed}"
BASE_NNUNET_RESULTS="${nnUNet_results:-/data2/yyp4247/data/nnUNet_data_filtered/nnUNet_results}"
export NNUNET_RESULTS_TAG="${NNUNET_RESULTS_TAG:-attn_full_g0.10_w0_a0.10_t0.45}"
if [[ -n "${NNUNET_RESULTS_DIR:-}" ]]; then
  export nnUNet_results="${NNUNET_RESULTS_DIR}"
elif [[ -n "${NNUNET_RESULTS_TAG:-}" ]]; then
  export nnUNet_results="${BASE_NNUNET_RESULTS}/experiments/${NNUNET_RESULTS_TAG}"
else
  export nnUNet_results="${BASE_NNUNET_RESULTS}"
fi

echo "[INFO] nnUNet_raw=${nnUNet_raw}"
echo "[INFO] nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "[INFO] nnUNet_results=${nnUNet_results}"

export NNUNET_TEXT_PROMPTS="${NNUNET_TEXT_PROMPTS:-prostate lesion}"
export NNUNET_TEXT_MODEL="${NNUNET_TEXT_MODEL:-hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224}"
export NNUNET_TEXT_EMBED_DIM="${NNUNET_TEXT_EMBED_DIM:-512}"
export NNUNET_TEXT_MODULATION="${NNUNET_TEXT_MODULATION:-none}"
export NNUNET_USE_ALIGNMENT_HEAD="${NNUNET_USE_ALIGNMENT_HEAD:-0}"
export NNUNET_RETURN_HEATMAP="${NNUNET_RETURN_HEATMAP:-0}"
export NNUNET_LAMBDA_ALIGN="${NNUNET_LAMBDA_ALIGN:-0}"
export NNUNET_LAMBDA_HEAT="${NNUNET_LAMBDA_HEAT:-0}"
export NNUNET_AUX_WARMUP_EPOCHS="${NNUNET_AUX_WARMUP_EPOCHS:-0}"
export NNUNET_AUX_RAMP_EPOCHS="${NNUNET_AUX_RAMP_EPOCHS:-0}"
export NNUNET_SKIP_OPENCLIP="${NNUNET_SKIP_OPENCLIP:-0}"
export NNUNET_USE_CROSS_ATTN_FINAL="${NNUNET_USE_CROSS_ATTN_FINAL:-1}"
export NNUNET_CROSS_GAMMA_INIT="${NNUNET_CROSS_GAMMA_INIT:-0.10}"
export NNUNET_CROSS_ALPHA="${NNUNET_CROSS_ALPHA:-0.10}"
export NNUNET_CROSS_TAU="${NNUNET_CROSS_TAU:-0.45}"
export ATTN_WARMUP_EPOCHS="${ATTN_WARMUP_EPOCHS:-0}"
export NNUNET_USE_TEXT_ADAPTOR="${NNUNET_USE_TEXT_ADAPTOR:-1}"

echo "[INFO] Text prompt='${NNUNET_TEXT_PROMPTS}' model='${NNUNET_TEXT_MODEL}'"

DATASET_ID="${NNUNET_DATASET:-Dataset2203_picai_split}"
CONFIG_NAME="${NNUNET_CONFIG:-3d_fullres}"
PLANS_NAME="${NNUNET_PLANS:-nnUNetPlans}"
TRAINER_NAME="${NNUNET_TEST_TRAINER:-nnUNetTrainerMultiEncoderAttnUNetText}"
FOLD_ID="${NNUNET_FOLD:-1}"
CHECKPOINT_FILE="${NNUNET_CHECKPOINT_FILE:-checkpoint_best.pth}"
USE_MANUAL_MODEL=0
MODEL_DIR=""
PY_BIN="${PY_BIN:-$(command -v python)}"

if [[ -n "${MODEL_SOURCE}" ]]; then
  if [[ -f "${MODEL_SOURCE}" ]]; then
    CHECKPOINT_FILE="$(basename "${MODEL_SOURCE}")"
    FOLD_DIR="$(dirname "${MODEL_SOURCE}")"
    FOLD_BASE="$(basename "${FOLD_DIR}")"
    if [[ "${FOLD_BASE}" =~ ^fold_([0-9]+)$ ]]; then
      FOLD_ID="${BASH_REMATCH[1]}"
      MODEL_DIR="$(dirname "${FOLD_DIR}")"
      USE_MANUAL_MODEL=1
    else
      echo "[ERROR] Unable to infer fold id from ${FOLD_DIR}. Expected folder named fold_<id>." >&2
      exit 1
    fi
  elif [[ -d "${MODEL_SOURCE}" ]]; then
    SOURCE_BASE="$(basename "${MODEL_SOURCE}")"
    if [[ "${SOURCE_BASE}" =~ ^fold_([0-9]+)$ ]]; then
      FOLD_ID="${BASH_REMATCH[1]}"
      MODEL_DIR="$(dirname "${MODEL_SOURCE}")"
      USE_MANUAL_MODEL=1
    else
      shopt -s nullglob
      FOLD_CANDIDATES=("${MODEL_SOURCE}"/fold_*)
      shopt -u nullglob
      if (( ${#FOLD_CANDIDATES[@]} > 0 )); then
        MODEL_DIR="${MODEL_SOURCE}"
        USE_MANUAL_MODEL=1
      else
        export nnUNet_results="${MODEL_SOURCE}"
      fi
    fi
  else
    echo "[ERROR] MODEL_PATH '${MODEL_SOURCE}' does not exist." >&2
    exit 1
  fi
fi

if [[ "${USE_MANUAL_MODEL}" == "1" ]]; then
  MANUAL_RESULTS_ROOT="$(dirname "$(dirname "${MODEL_DIR}")")"
  export nnUNet_results="${MANUAL_RESULTS_ROOT}"
  echo "[INFO] Using manual model directory: ${MODEL_DIR} (fold ${FOLD_ID}, checkpoint ${CHECKPOINT_FILE})"
  echo "[INFO] Overriding nnUNet_results root => ${nnUNet_results}"
else
  echo "[INFO] Using nnUNet results root: ${nnUNet_results}"
fi
echo "[INFO] Dataset=${DATASET_ID} Config=${CONFIG_NAME} Plans=${PLANS_NAME} Trainer=${TRAINER_NAME} Fold=${FOLD_ID}"

echo "[INFO] Creating output dir: ${OUTPUT_DIR}"
mkdir -p "$OUTPUT_DIR"

EVAL_DISABLED="${NNUNET_SKIP_EVAL:-0}"
GT_OVERRIDE="${NNUNET_GT_DIR:-}"
DATASET_PREP_DIR="$(${PY_BIN} - "$DATASET_ID" <<'PY'
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
import os, sys
print(os.path.join(nnUNet_preprocessed, maybe_convert_to_dataset_name(sys.argv[1])))
PY
)"
GT_DIR="${GT_OVERRIDE:-${DATASET_PREP_DIR}/gt_segmentations}"
RAW_DATASET_DIR="${nnUNet_raw}/$(basename "${DATASET_PREP_DIR}")"
if [[ ! -d "${GT_DIR}" ]] && [[ -d "${RAW_DATASET_DIR}/labelsTs" ]]; then
  GT_DIR="${RAW_DATASET_DIR}/labelsTs"
fi
DATASET_JSON="${DATASET_PREP_DIR}/dataset.json"
PLANS_JSON="${DATASET_PREP_DIR}/${PLANS_NAME}.json"
SUMMARY_JSON="${OUTPUT_DIR}/summary.json"

set -x
nnUNetv2_predict \
  -i "$INPUT_DIR" \
  -o "$OUTPUT_DIR" \
  -d "$DATASET_ID" \
  -c "$CONFIG_NAME" \
  -p "$PLANS_NAME" \
  -tr "$TRAINER_NAME" \
  -f "$FOLD_ID" \
  -chk "$CHECKPOINT_FILE" \
  --save_probabilities \
  --verbose
set +x

echo "[INFO] Prediction finished. Results written to ${OUTPUT_DIR}"

if [[ "${EVAL_DISABLED}" != "1" ]]; then
  if [[ -d "${GT_DIR}" ]] && [[ -f "${DATASET_JSON}" ]] && [[ -f "${PLANS_JSON}" ]]; then
    echo "[INFO] Running evaluation with ground truth from ${GT_DIR}"
    set -x
    nnUNetv2_evaluate_folder \
      "${GT_DIR}" \
      "${OUTPUT_DIR}" \
      -djfile "${DATASET_JSON}" \
      -pfile "${PLANS_JSON}" \
      -o "${SUMMARY_JSON}"
    set +x
    echo "[INFO] Evaluation summary written to ${SUMMARY_JSON}"
  else
    echo "[INFO] Skipping evaluation (missing GT folder or dataset/plans json)."
    echo "       GT_DIR=${GT_DIR} dataset.json exists? $([[ -f "${DATASET_JSON}" ]] && echo yes || echo no)"
    echo "       ${PLANS_NAME}.json exists? $([[ -f "${PLANS_JSON}" ]] && echo yes || echo no)"
    echo "       (set NNUNET_SKIP_EVAL=1 to silence this check)"
  fi
else
  echo "[INFO] Evaluation skipped because NNUNET_SKIP_EVAL=${EVAL_DISABLED}"
fi
