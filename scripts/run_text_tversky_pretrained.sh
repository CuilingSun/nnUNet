#!/usr/bin/env bash
set -euo pipefail

# Run Text trainer with Tversky+TopK, load pretrained image encoder, and use a fixed prompt.
# Usage:
#   bash /data2/yyp4247/nnUNet/scripts/run_text_tversky_pretrained.sh <GPU_ID> <DATASET_ID_OR_NAME> [FOLD] [CONFIG] [PLANS]
# Example:
#   bash /data2/yyp4247/nnUNet/scripts/run_text_tversky_pretrained.sh 0 Dataset2202_picai_split 0 3d_fullres nnUNetPlans

GPU="${1:?Need GPU id}"
DATASET="${2:?Need dataset id_or_name}"
FOLD="${3:-0}"
CONFIG="${4:-3d_fullres}"
PLANS="${5:-nnUNetPlans}"

# 1) Legacy PlainConv checkpoint (mapped onto all encoders)
export NNUNET_LEGACY_PRETRAIN="${NNUNET_LEGACY_PRETRAIN:-/data2/yyp4247/nnUNet/pretrained/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_final.pth}"
export NNUNET_TRAINER="${NNUNET_TRAINER:-nnUNetTrainerMultiEncodernnUNetText}"
export nnUNet_compile="${nnUNet_compile:-0}"

# 2) Text prompt & embedding config
export NNUNET_TEXT_PROMPTS="${NNUNET_TEXT_PROMPTS:-prostate lesion}"
export NNUNET_TEXT_EMBED_DIM="${NNUNET_TEXT_EMBED_DIM:-512}"
export NNUNET_TEXT_MODEL="${NNUNET_TEXT_MODEL:-hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224}"
# Try to use online model unless explicitly offline
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export NNUNET_SKIP_OPENCLIP="${NNUNET_SKIP_OPENCLIP:-0}"

# 3) Loss: Tversky + TopK
LOSS_KIND="tversky"

# 4) Stable aux-loss schedule (warmup + ramp)
#    Override externally if desired. Defaults chosen for stability with pretrained encoder.
export NNUNET_LAMBDA_ALIGN="${NNUNET_LAMBDA_ALIGN:-0.2}"
export NNUNET_LAMBDA_HEAT="${NNUNET_LAMBDA_HEAT:-0.5}"
export NNUNET_AUX_WARMUP_EPOCHS="${NNUNET_AUX_WARMUP_EPOCHS:-15}"
export NNUNET_AUX_RAMP_EPOCHS="${NNUNET_AUX_RAMP_EPOCHS:-15}"
export NNUNET_AUX_DYNAMIC="${NNUNET_AUX_DYNAMIC:-1}"
export NNUNET_AUX_LOSS_STABLE_EPOCHS="${NNUNET_AUX_LOSS_STABLE_EPOCHS:-5}"
export NNUNET_AUX_LOSS_REL_CHANGE="${NNUNET_AUX_LOSS_REL_CHANGE:-0.05}"
export NNUNET_AUX_EMA_DICE_THRESH="${NNUNET_AUX_EMA_DICE_THRESH:-0.30}"
# NNUNET_AUX_HARD_START_EPOCH is optional; if unset, defaults to WARMUP+RAMP

# 5) Quick mode by default; override QUICK=0 for longer runs
export QUICK="${QUICK:-1}"

# 6) Results root (override to save under a custom directory)
#    This integrates with tools/run_text_single.sh which honors NNUNET_RESULTS_DIR.
export NNUNET_RESULTS_DIR="${NNUNET_RESULTS_DIR:-/data2/yyp4247/data/nnUNet_results/0921/umamba_pretrain}"
export PY_GETS="${PY_GETS:-/data2/yyp4247/.conda/envs/nnunetv2/bin/python}"

# Delegate to the generic runner (handles logging, val, eval, postprocess)
# Stream output to terminal by default for debugging
export FOREGROUND="${FOREGROUND:-1}"
bash "$(dirname "$0")/../tools/run_text_single.sh" "$GPU" "$FOLD" "$DATASET" "$CONFIG" "$PLANS" "$LOSS_KIND"
