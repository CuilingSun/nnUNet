#!/usr/bin/env bash
set -euo pipefail

# 0) env
# source "$HOME/.bashrc"
# conda activate nnunetv2

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# 1) paths
export nnUNet_raw=/data2/yyp4247/data/nnUNet_raw
export nnUNet_preprocessed=/data2/yyp4247/data/nnUNet_preprocessed
export nnUNet_results=/data2/yyp4247/data/nnUNet_results
mkdir -p "$nnUNet_preprocessed" "$nnUNet_results"

# 2) dataset / gpu
DATASET_ID=2202
FOLD=2
GPU=${GPU:-}
# Respect external CUDA_VISIBLE_DEVICES unless GPU is provided
if [[ -n "$GPU" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Mixed precision stability & learning rate: allow safe defaults, overridable by caller
export NNUNET_LR=${NNUNET_LR:-0.003}                 # safer LR; caller can override

# 3) data aug quick-fix
export NNUNET_DISABLE_GAUSS_BLUR=1
export NNUNET_DISABLE_AFFINE=1

# Quick vs. Full training switch
# QUICK=1  -> fast feedback (fewer iters/epoch, lighter validation/exports, AMP on, fewer DA workers)
# QUICK=0  -> full training settings (default nnU-Net iters/epoch), still conservative on DA workers for stability
QUICK=${QUICK:-1}
if [[ "$QUICK" == "1" ]]; then
  export NNUNET_DISABLE_AMP=${NNUNET_DISABLE_AMP:-0}   # AMP on for speed
  export nnUNet_n_proc_DA=${nnUNet_n_proc_DA:-1}       # start with 1 worker; try 2 if stable
  export NNUNET_SAVE_HEATMAPS=${NNUNET_SAVE_HEATMAPS:-0}
  export NNUNET_EXPORT_FULL_HEATMAP=${NNUNET_EXPORT_FULL_HEATMAP:-0}
  export NNUNET_HEATMAP_DIRNAME=${NNUNET_HEATMAP_DIRNAME:-validation_heatmaps}
  export NNUNET_EXPORT_HEATMAP_NIFTI=${NNUNET_EXPORT_HEATMAP_NIFTI:-0}
  # Increase iterations/epoch for more stable metrics (QUICK mode: default 200)
  export NNUNET_ITERS_PER_EPOCH=${NNUNET_ITERS_PER_EPOCH:-200}
  # Increase validation iterations to smooth metrics
  export NNUNET_VAL_ITERS=${NNUNET_VAL_ITERS:-100}
  MODE_STR="QUICK"
else
  export NNUNET_DISABLE_AMP=${NNUNET_DISABLE_AMP:-0}
  export nnUNet_n_proc_DA=${nnUNet_n_proc_DA:-1}
  # Use higher default here as well for consistency
  export NNUNET_ITERS_PER_EPOCH=${NNUNET_ITERS_PER_EPOCH:-300}
  # Higher validation iterations for full mode
  export NNUNET_VAL_ITERS=${NNUNET_VAL_ITERS:-200}
  export NNUNET_SAVE_HEATMAPS=${NNUNET_SAVE_HEATMAPS:-0}
  export NNUNET_EXPORT_FULL_HEATMAP=${NNUNET_EXPORT_FULL_HEATMAP:-0}
  export NNUNET_HEATMAP_DIRNAME=${NNUNET_HEATMAP_DIRNAME:-validation_heatmaps}
  export NNUNET_EXPORT_HEATMAP_NIFTI=${NNUNET_EXPORT_HEATMAP_NIFTI:-0}
  MODE_STR="FULL"
fi

# 4) text/heatmap settings (ENABLED for nnUNetTrainerMultiEncoderUNetText)
# You can override these via environment variables before calling this script.
export NNUNET_TEXT_PROMPTS=${NNUNET_TEXT_PROMPTS:-"prostate lesion"}
export NNUNET_TEXT_MODEL=${NNUNET_TEXT_MODEL:-"hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"}
export NNUNET_TEXT_EMBED_DIM=${NNUNET_TEXT_EMBED_DIM:-512}
# Stabilize first: disable modulation and aux losses by default; user can opt-in later
export NNUNET_TEXT_MODULATION=${NNUNET_TEXT_MODULATION:-none}   # none | film | gate
export NNUNET_USE_ALIGNMENT_HEAD=${NNUNET_USE_ALIGNMENT_HEAD:-1}
# Request network to return heatmap (extras['sim']) so the trainer can apply heatmap loss
export NNUNET_RETURN_HEATMAP=${NNUNET_RETURN_HEATMAP:-1}
# Enable heatmap/align aux losses with a curriculum: warmup (no aux), then linear ramp-in
export NNUNET_LAMBDA_ALIGN=${NNUNET_LAMBDA_ALIGN:-0.2}
export NNUNET_LAMBDA_HEAT=${NNUNET_LAMBDA_HEAT:-0.5}
export NNUNET_AUX_WARMUP_EPOCHS=${NNUNET_AUX_WARMUP_EPOCHS:-80}
export NNUNET_AUX_RAMP_EPOCHS=${NNUNET_AUX_RAMP_EPOCHS:-50}
# For speed during training, disable heavy heatmap exports; re-enable later for dedicated validation runs
export NNUNET_SAVE_HEATMAPS=${NNUNET_SAVE_HEATMAPS:-0}
export NNUNET_EXPORT_FULL_HEATMAP=${NNUNET_EXPORT_FULL_HEATMAP:-0}
export NNUNET_HEATMAP_DIRNAME=${NNUNET_HEATMAP_DIRNAME:-validation_heatmaps}
export NNUNET_EXPORT_HEATMAP_NIFTI=${NNUNET_EXPORT_HEATMAP_NIFTI:-0}
# Default to offline to avoid blocking downloads; trainer falls back gracefully
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
# Or explicitly skip OpenCLIP initialization
export NNUNET_SKIP_OPENCLIP=${NNUNET_SKIP_OPENCLIP:-0}
# Offline alternative: provide a local embedding and skip model download
# export NNUNET_TEXT_EMBED_PATH=/path/to/text_embedding_${NNUNET_TEXT_EMBED_DIM}.npy

# 5) logging
RUN_DIR="/data2/yyp4247/nnUNet/log/$(date +'%Y%m%d_%H%M%S')"
mkdir -p "$RUN_DIR"
echo "Logs -> $RUN_DIR (mode: $MODE_STR)"
export PYTHONUNBUFFERED=1
# Avoid Matplotlib font cache on slow HOME (set to local log dir)
export MPLBACKEND=Agg
export MPLCONFIGDIR="$RUN_DIR/.mpl"
mkdir -p "$MPLCONFIGDIR"

# 6) preprocess (如已做过可注释)
# nnUNetv2_plan_and_preprocess -d "$DATASET_ID" -c 3d_fullres -np 8 2>&1 | tee "$RUN_DIR/pproc.log"

# 7) train
# Recommended: disable torch.compile due to known env issues
export nnUNet_compile=0

# Train
NNUNET_ITERS_PER_EPOCH=${NNUNET_ITERS_PER_EPOCH} \
NNUNET_VAL_ITERS=${NNUNET_VAL_ITERS} \
NNUNET_OVERSAMPLE_FG=${NNUNET_OVERSAMPLE_FG:-0.9} \
NNUNET_DISABLE_AMP=${NNUNET_DISABLE_AMP} \
NNUNET_LR=${NNUNET_LR} \
nnUNetv2_train "$DATASET_ID" 3d_fullres "$FOLD" \
  -tr nnUNetTrainerMultiEncoderUNetText --npz 2>&1 | tee "$RUN_DIR/train.log"
