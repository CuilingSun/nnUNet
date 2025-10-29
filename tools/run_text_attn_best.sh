#!/usr/bin/env bash
set -euo pipefail

# Attention-enabled Text trainer wrapper using the tuned cross-attention hyperparameters
# (gamma=0.10, alpha=0.10, tau=0.45, warmup=0) and loading a pretrained multi-encoder checkpoint.
# Usage:
#   bash tools/run_text_attn_best.sh <GPU_ID> [FOLD] [DATASET] [CONFIG] [PLANS] [PRETRAINED_CKPT]
#
# Example:
#   bash tools/run_text_attn_best.sh 4 1 Dataset2203_picai_split 3d_fullres nnUNetPlans \
#       /data2/yyp4247/data/nnUNet_data_filtered/nnUNet_results/Dataset2203_picai_split/nnUNetTrainerMultiEncoderUNetText__nnUNetPlans__3d_fullres/fold_1/checkpoint_best.pth

GPU="${1:?Need GPU id}"
FOLD="${2:-1}"
DATASET="${3:-Dataset2203_picai_split}"
CONFIG="${4:-3d_fullres}"
PLANS="${5:-nnUNetPlans}"
PRETRAIN_CKPT_DEFAULT="/data2/yyp4247/data/nnUNet_data_filtered/nnUNet_results/Dataset2203_picai_split/nnUNetTrainerMultiEncoderUNetText__nnUNetPlans__3d_fullres/fold_${FOLD}/checkpoint_best.pth"
PRETRAIN_CKPT="${6:-${PRETRAIN_CKPT_DEFAULT}}"

# Keep results under the filtered workspace experiments tree by default.
export nnUNet_results="${nnUNet_results:-/data2/yyp4247/data/nnUNet_data_filtered/nnUNet_results}"

export NNUNET_TRAINER="${NNUNET_TRAINER:-nnUNetTrainerMultiEncoderAttnUNetText}"

# Tuned cross-attention controls.
export NNUNET_USE_CROSS_ATTN_FINAL="${NNUNET_USE_CROSS_ATTN_FINAL:-1}"
export NNUNET_CROSS_GAMMA_INIT="${NNUNET_CROSS_GAMMA_INIT:-0.10}"
export NNUNET_CROSS_ALPHA="${NNUNET_CROSS_ALPHA:-0.12}"
export NNUNET_CROSS_TAU="${NNUNET_CROSS_TAU:-0.44}"
export ATTN_WARMUP_EPOCHS="${ATTN_WARMUP_EPOCHS:-0}"
export BASE_LR_REFINER="${BASE_LR_REFINER:-5e-4}"

if [[ -f "${PRETRAIN_CKPT}" ]]; then
  export NNUNET_PRETRAINED_WEIGHTS="${PRETRAIN_CKPT}"
  echo "[run_text_attn_best] Using pretrained checkpoint: ${NNUNET_PRETRAINED_WEIGHTS}" >&2
else
  echo "[WARN] Pretrained checkpoint not found at ${PRETRAIN_CKPT}; continuing without it." >&2
  unset NNUNET_PRETRAINED_WEIGHTS
fi

# Disable alignment/heatmap auxiliary branches for this finetune.
export NNUNET_USE_ALIGNMENT_HEAD=0
export NNUNET_RETURN_HEATMAP=0
export NNUNET_LAMBDA_ALIGN=0
export NNUNET_LAMBDA_HEAT=0
export NNUNET_AUX_WARMUP_EPOCHS=0
export NNUNET_AUX_RAMP_EPOCHS=0
export NNUNET_SAVE_HEATMAPS=0
export NNUNET_EXPORT_FULL_HEATMAP=0
export NNUNET_EXPORT_HEATMAP_NIFTI=0

# Tag results so outputs land in experiments/attn_full_g..._w..._a..._t...
TAG_SUFFIX="g${NNUNET_CROSS_GAMMA_INIT}_w${ATTN_WARMUP_EPOCHS}_a${NNUNET_CROSS_ALPHA}_t${NNUNET_CROSS_TAU}"
export NNUNET_RESULTS_TAG="${NNUNET_RESULTS_TAG:-attn_full_${TAG_SUFFIX}}"

exec bash "$(dirname "$0")/run_text_single.sh" "${GPU}" "${FOLD}" "${DATASET}" "${CONFIG}" "${PLANS}" tversky
