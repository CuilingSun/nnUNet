#!/usr/bin/env bash
set -euo pipefail

# Quick smoke + 3-minute probe to debug training startup.

# 0) (optional) conda
# source "$HOME/.bashrc" && conda activate nnunetv2

# 1) Paths (edit if needed)
export nnUNet_raw=${nnUNet_raw:-/data2/yyp4247/U-Mamba/data/nnUNet_raw}
export nnUNet_preprocessed=${nnUNet_preprocessed:-/data2/yyp4247/U-Mamba/data/nnUNet_preprocessed}
export nnUNet_results=${nnUNet_results:-/data2/yyp4247/U-Mamba/data/nnUNet_results}
mkdir -p "$nnUNet_preprocessed" "$nnUNet_results"

# 2) Dataset, config, fold, GPU
DATASET_ID=${DATASET_ID:-2201}
CONFIG=${CONFIG:-3d_fullres}
FOLD=${FOLD:-0}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU

# 3) Stability flags
export MPLBACKEND=${MPLBACKEND:-Agg}
export nnUNet_compile=${nnUNet_compile:-0}

echo "== ENV =="
python -c "import sys, torch; print('python', sys.version); print('cuda?', torch.cuda.is_available())"
echo "raw=$nnUNet_raw"; echo "pproc=$nnUNet_preprocessed"; echo "results=$nnUNet_results";

echo "== SMOKE (no text modulation) =="
unset NNUNET_TEXT_EMBED_DIM || true
export NNUNET_TEXT_MODULATION=none
# resolve repo root so script works regardless of CWD
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
python "$REPO_ROOT/scripts/smoke_test_me_text.py" Dataset${DATASET_ID}_picai "$CONFIG" 1 || true

echo "== SMOKE (with text FiLM) =="
export NNUNET_TEXT_EMBED_DIM=${NNUNET_TEXT_EMBED_DIM:-512}
export NNUNET_TEXT_MODULATION=${NNUNET_TEXT_MODULATION:-film}
export NNUNET_USE_ALIGNMENT_HEAD=${NNUNET_USE_ALIGNMENT_HEAD:-1}
export NNUNET_RETURN_HEATMAP=${NNUNET_RETURN_HEATMAP:-1}
python "$REPO_ROOT/scripts/smoke_test_me_text.py" Dataset${DATASET_ID}_picai "$CONFIG" 1 || true

echo "== TRAIN PROBE (3 min, no gating) =="
export NNUNET_TEXT_MODULATION=none
timeout 180s nnUNetv2_train "$DATASET_ID" "$CONFIG" "$FOLD" -tr nnUNetTrainerMultiEncoderUNetText --npz || true

echo "== TRAIN PROBE (3 min, FiLM) =="
export NNUNET_TEXT_EMBED_DIM=${NNUNET_TEXT_EMBED_DIM:-512}
export NNUNET_TEXT_MODULATION=film
export NNUNET_USE_ALIGNMENT_HEAD=1
export NNUNET_RETURN_HEATMAP=1
timeout 180s nnUNetv2_train "$DATASET_ID" "$CONFIG" "$FOLD" -tr nnUNetTrainerMultiEncoderUNetText --npz || true

echo "== TAIL LOG =="
LOGDIR="$nnUNet_results/Dataset${DATASET_ID}_picai/nnUNetTrainerMultiEncoderUNetText__nnUNetPlans__${CONFIG}/fold_${FOLD}"
if [[ -d "$LOGDIR" ]]; then
  tail -n 200 "$LOGDIR"/training_log_*.txt || true
else
  echo "No log dir yet: $LOGDIR"
fi

echo "Debug run finished."
