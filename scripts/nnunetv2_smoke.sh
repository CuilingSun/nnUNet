#!/usr/bin/env bash
set -euo pipefail

############################################
# nnU-Net v2 quick smoke runner
# - Sets env paths
# - Optional preprocess
# - Minimal training run
# - Logs to /data2/yyp4247/nnUNet/log/<ts>/
#
# Usage:
#  GPU=0 ./nnunetv2_smoke.sh [DATASET] [FOLD] [CONFIG]
#    DATASET: Dataset name (e.g., Dataset2201_picai) or id (e.g., 2201). Default: Dataset2201_picai
#    FOLD   : 0..4, default 0
#    CONFIG : nnU-Net config, default 3d_fullres
#
# Flags (env vars):
#   PREPROCESS=1              # run plan&preprocess first
#   DISABLE_BLUR=1            # disable GaussianBlur transform
#   DISABLE_AFFINE=1          # disable spatial/affine transform
#   WORKERS=0                 # dataloader workers (0 for single-thread debug)
#   TRAINER=nnUNetTrainer     # override trainer class name
#   EPOCHS=                   # (optional) not used by stock CLI; here for completeness
#   TEXT=0/1                  # if 1, export a default BiomedCLIP text setup (needs deps)
#
# Examples:
#   GPU=0 ./nnunetv2_smoke.sh 2201 0 3d_fullres
#   GPU=1 PREPROCESS=1 DISABLE_AFFINE=1 WORKERS=0 ./nnunetv2_smoke.sh Dataset2201_picai
############################################

# --------- positional args with defaults ----------
DATASET="${1:-Dataset2201_picai}"   # name or id
FOLD="${2:-0}"
CONFIG="${3:-3d_fullres}"

# --------- env / paths ----------
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

export nnUNet_raw="/data2/yyp4247/U-Mamba/data/nnUNet_raw"
export nnUNet_preprocessed="/data2/yyp4247/U-Mamba/data/nnUNet_preprocessed"
export nnUNet_results="/data2/yyp4247/U-Mamba/data/nnUNet_results"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

# Optional debug-friendly threading caps
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# Dataloader workers (0 = main thread; good for捕捉首个异常)
export NNUNET_DATALOADER_WORKERS="${WORKERS:-2}"

# Toggle problematic augs quickly (always export defaults under set -u)
export NNUNET_DISABLE_GAUSS_BLUR="${DISABLE_BLUR:-0}"
export NNUNET_DISABLE_AFFINE="${DISABLE_AFFINE:-0}"

# Optional text branch quick-setup (requires: pip install transformers open_clip_torch)
if [[ "${TEXT:-0}" == "1" ]]; then
  export NNUNET_TEXT_PROMPTS="${NNUNET_TEXT_PROMPTS:-prostate lesion}"
  export NNUNET_TEXT_MODEL="${NNUNET_TEXT_MODEL:-hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224}"
  export NNUNET_TEXT_EMBED_DIM="${NNUNET_TEXT_EMBED_DIM:-512}"
  export NNUNET_USE_ALIGNMENT_HEAD="${NNUNET_USE_ALIGNMENT_HEAD:-1}"
  export NNUNET_RETURN_HEATMAP="${NNUNET_RETURN_HEATMAP:-0}"
  export NNUNET_TEXT_MODULATION="${NNUNET_TEXT_MODULATION:-film}"
fi

# Trainer override (optional)
TRAINER="${TRAINER:-}"

# --------- logging ----------
TS="$(date +'%Y%m%d_%H%M%S')"
RUN_DIR="/data2/yyp4247/nnUNet/log/${TS}"
mkdir -p "$RUN_DIR"
echo "Logs -> $RUN_DIR"
python -V 2>&1 | tee "$RUN_DIR/env.txt"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a "$RUN_DIR/env.txt"
echo "raw=$nnUNet_raw" | tee -a "$RUN_DIR/env.txt"
echo "pproc=$nnUNet_preprocessed" | tee -a "$RUN_DIR/env.txt"
echo "results=$nnUNet_results" | tee -a "$RUN_DIR/env.txt"
echo "workers=$NNUNET_DATALOADER_WORKERS blur=$NNUNET_DISABLE_GAUSS_BLUR${NNUNET_DISABLE_GAUSS_BLUR:+} affine=$NNUNET_DISABLE_AFFINE${NNUNET_DISABLE_AFFINE:+}" \
  | tee -a "$RUN_DIR/env.txt"

# --------- dataset existence checks ----------
is_number='^[0-9]+$'
if [[ "$DATASET" =~ $is_number ]]; then
  DATASET_ID="$DATASET"
  # try to infer name like Dataset<ID>_*
  CAND=$(ls -1 "$nnUNet_preprocessed" 2>/dev/null | grep -E "^Dataset${DATASET_ID}_" || true)
  if [[ -n "$CAND" ]]; then
    DATASET_NAME="$(echo "$CAND" | head -n1)"
  else
    DATASET_NAME=""  # may still be fine if preprocess will run
  fi
else
  DATASET_NAME="$DATASET"
  # try to extract ID from name: Dataset2201_foo -> 2201
  if [[ "$DATASET_NAME" =~ ^Dataset([0-9]+)_ ]]; then
    DATASET_ID="${BASH_REMATCH[1]}"
  else
    DATASET_ID=""
  fi
fi

# --------- optional preprocess ----------
if [[ "${PREPROCESS:-0}" == "1" ]]; then
  if [[ -z "${DATASET_ID:-}" ]]; then
    echo "[ERR] PREPROCESS=1 但无法从 \"$DATASET\" 解析出数字 ID（形如 2201）。"
    echo "     请用数字 ID 调用：  PREPROCESS=1 ./nnunetv2_smoke.sh 2201"
    exit 2
  fi
  echo "[pproc] nnUNetv2_plan_and_preprocess -d $DATASET_ID -c $CONFIG -np ${PPROC_PROCS:-8}"
  nnUNetv2_plan_and_preprocess -d "$DATASET_ID" -c "$CONFIG" -np "${PPROC_PROCS:-8}" 2>&1 | tee "$RUN_DIR/preprocess.log"
fi

# --------- choose train args (id or name) ----------
TRAIN_ARGS_ID=()
if [[ -n "${DATASET_ID:-}" ]]; then
  TRAIN_ARGS_ID=("$DATASET_ID" "$CONFIG" "$FOLD")
fi
TRAIN_ARGS_NAME=("$DATASET_NAME" "$CONFIG" "$FOLD")

# Prefer name if the preprocessed folder exists
if [[ -n "${DATASET_NAME:-}" && -d "$nnUNet_preprocessed/$DATASET_NAME" ]]; then
  echo "[train] using dataset NAME: $DATASET_NAME"
  ARGS=("${TRAIN_ARGS_NAME[@]}")
elif [[ -n "${TRAIN_ARGS_ID[*]:-}" ]]; then
  echo "[train] using dataset ID: $DATASET_ID"
  ARGS=("${TRAIN_ARGS_ID[@]}")
else
  echo "[ERR] 无法确定数据集。请检查 $nnUNet_preprocessed 下是否存在对应目录，或使用数字 ID。"
  exit 3
fi

# --------- run training (tee to log) ----------
set -x
if [[ -n "$TRAINER" ]]; then
  nnUNetv2_train "${ARGS[@]}" -tr "$TRAINER" --npz 2>&1 | tee "$RUN_DIR/train.log"
else
  nnUNetv2_train "${ARGS[@]}" --npz 2>&1 | tee "$RUN_DIR/train.log"
fi
set +x

echo "Done. See logs in: $RUN_DIR"
