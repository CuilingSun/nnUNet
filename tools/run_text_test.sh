#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <input_dir> <output_dir> [GPU_ID]" >&2
  exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
GPU_ID="${3:-5}"

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
export nnUNet_results="${nnUNet_results:-/data2/yyp4247/data/nnUNet_data_filtered/nnUNet_results}"

echo "[INFO] nnUNet_raw=${nnUNet_raw}"
echo "[INFO] nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "[INFO] nnUNet_results=${nnUNet_results}"

export NNUNET_TEXT_PROMPTS="${NNUNET_TEXT_PROMPTS:-prostate lesion}"
export NNUNET_TEXT_MODEL="${NNUNET_TEXT_MODEL:-hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224}"
export NNUNET_TEXT_EMBED_DIM="${NNUNET_TEXT_EMBED_DIM:-512}"
export NNUNET_TEXT_MODULATION="${NNUNET_TEXT_MODULATION:-none}"
export NNUNET_USE_ALIGNMENT_HEAD="${NNUNET_USE_ALIGNMENT_HEAD:-1}"
export NNUNET_RETURN_HEATMAP="${NNUNET_RETURN_HEATMAP:-1}"
export NNUNET_LAMBDA_ALIGN="${NNUNET_LAMBDA_ALIGN:-0.2}"
export NNUNET_LAMBDA_HEAT="${NNUNET_LAMBDA_HEAT:-0.5}"
export NNUNET_AUX_WARMUP_EPOCHS="${NNUNET_AUX_WARMUP_EPOCHS:-80}"
export NNUNET_AUX_RAMP_EPOCHS="${NNUNET_AUX_RAMP_EPOCHS:-50}"
export NNUNET_SKIP_OPENCLIP="${NNUNET_SKIP_OPENCLIP:-0}"
export NNUNET_USE_CROSS_ATTN_FINAL="${NNUNET_USE_CROSS_ATTN_FINAL:-1}"
export NNUNET_USE_TEXT_ADAPTOR="${NNUNET_USE_TEXT_ADAPTOR:-1}"

echo "[INFO] Text prompt='${NNUNET_TEXT_PROMPTS}' model='${NNUNET_TEXT_MODEL}'"

echo "[INFO] Creating output dir: ${OUTPUT_DIR}"
mkdir -p "$OUTPUT_DIR"

set -x
nnUNetv2_predict \
  -i "$INPUT_DIR" \
  -o "$OUTPUT_DIR" \
  -d Dataset2203_picai_split \
  -c 3d_fullres \
  -p nnUNetPlans \
  -tr nnUNetTrainerMoEAdapterUNetText \
  -f 1 \
  -chk checkpoint_best.pth \
  --save_probabilities \
  --verbose
set +x

echo "[INFO] Prediction finished. Results written to ${OUTPUT_DIR}"
