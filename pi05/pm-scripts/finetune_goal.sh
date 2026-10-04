#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_NAME="pi05_libero_goal_poisoned_no_noops"
DEFAULT_EXP_NAME="goal_ft_$(date +%Y%m%d-%H%M%S)"
EXP_NAME="${1:-$DEFAULT_EXP_NAME}"

: "${HF_LEROBOT_HOME:=$ROOT_DIR/libero_datasets_ps}"
: "${CUDA_VISIBLE_DEVICES:=3}"
: "${NORM_STATS_CUDA_VISIBLE_DEVICES:=${CUDA_VISIBLE_DEVICES%%,*}}"
: "${TRAIN_BATCH_SIZE:=12}"
: "${XLA_PYTHON_CLIENT_MEM_FRACTION:=0.90}"
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True,max_split_size_mb:128}"
: "${WANDB_MODE:=disabled}"
: "${TRAIN_BACKEND:=pytorch}"
: "${LOCAL_PI05_BASE:=/data1/zhaoxueyang_191/VLA_code/models/pi05_base}"
: "${TRAIN_NUM_WORKERS:=4}"
: "${PYTORCH_COMPILE_MODE:=None}"
: "${OPENPI_USE_ZERO_OPTIMIZER:=1}"
: "${ALLOW_CPU_TRAIN:=0}"

echo "[Info] config=$CONFIG_NAME"
echo "[Info] exp_name=$EXP_NAME"
echo "[Info] HF_LEROBOT_HOME=$HF_LEROBOT_HOME"
echo "[Info] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[Info] NORM_STATS_CUDA_VISIBLE_DEVICES=$NORM_STATS_CUDA_VISIBLE_DEVICES"
echo "[Info] TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
echo "[Info] PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "[Info] WANDB_MODE=$WANDB_MODE"
echo "[Info] TRAIN_BACKEND=$TRAIN_BACKEND"
echo "[Info] LOCAL_PI05_BASE=$LOCAL_PI05_BASE"
echo "[Info] TRAIN_NUM_WORKERS=$TRAIN_NUM_WORKERS"
echo "[Info] PYTORCH_COMPILE_MODE=$PYTORCH_COMPILE_MODE"
echo "[Info] OPENPI_USE_ZERO_OPTIMIZER=$OPENPI_USE_ZERO_OPTIMIZER"
echo "[Info] ALLOW_CPU_TRAIN=$ALLOW_CPU_TRAIN"

if [[ "${SKIP_NORM_STATS:-0}" != "1" ]]; then
  echo "[Step] compute_norm_stats"
  if [[ -n "${MAX_FRAMES:-}" ]]; then
    HF_LEROBOT_HOME="$HF_LEROBOT_HOME" CUDA_VISIBLE_DEVICES="$NORM_STATS_CUDA_VISIBLE_DEVICES" \
      uv run --no-sync scripts/compute_norm_stats.py --config-name "$CONFIG_NAME" --max-frames "$MAX_FRAMES"
  else
    HF_LEROBOT_HOME="$HF_LEROBOT_HOME" CUDA_VISIBLE_DEVICES="$NORM_STATS_CUDA_VISIBLE_DEVICES" \
      uv run --no-sync scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"
  fi
else
  echo "[Skip] compute_norm_stats"
fi

if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
  else
    NUM_VISIBLE_DEVICES=1
  fi
else
  NUM_VISIBLE_DEVICES="$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")"
fi

if (( TRAIN_BATCH_SIZE % NUM_VISIBLE_DEVICES != 0 )); then
  echo "[Error] batch_size ($TRAIN_BATCH_SIZE) must be divisible by visible devices ($NUM_VISIBLE_DEVICES)." >&2
  echo "[Hint] Set TRAIN_BATCH_SIZE to a multiple of $NUM_VISIBLE_DEVICES; for 3 GPUs, 12 is a safer starting point than 24." >&2
  exit 1
fi

TRAIN_ARGS=(--exp-name "$EXP_NAME" --batch-size "$TRAIN_BATCH_SIZE")
if [[ "${RESUME:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--resume)
else
  if [[ "${OVERWRITE:-1}" == "1" ]]; then
    TRAIN_ARGS+=(--overwrite)
  fi
fi

if [[ "$WANDB_MODE" == "disabled" ]]; then
  TRAIN_ARGS+=(--no-wandb-enabled)
fi

case "$TRAIN_BACKEND" in
  jax)
    TRAIN_ENTRY=(scripts/train.py "$CONFIG_NAME")
    ;;
  pytorch)
    if [[ ! -f "$LOCAL_PI05_BASE/model.safetensors" ]]; then
      echo "[Error] Local pi05 base not found: $LOCAL_PI05_BASE/model.safetensors" >&2
      exit 1
    fi

    TORCH_CHECK_OUTPUT="$(uv run --no-sync python -c 'import torch; print(1 if torch.cuda.is_available() else 0); print(torch.__version__)')"
    TORCH_CUDA_OK="$(echo "$TORCH_CHECK_OUTPUT" | sed -n '1p')"
    TORCH_VERSION_STR="$(echo "$TORCH_CHECK_OUTPUT" | sed -n '2p')"
    if [[ "$TORCH_CUDA_OK" != "1" && "$ALLOW_CPU_TRAIN" != "1" ]]; then
      echo "[Error] PyTorch CUDA is not available (torch=$TORCH_VERSION_STR)." >&2
      echo "[Hint] Current env appears CPU-only; GPU training will be extremely slow and look stuck." >&2
      echo "[Hint] Install CUDA-enabled torch, or set ALLOW_CPU_TRAIN=1 if you intentionally train on CPU." >&2
      exit 1
    fi

    TRAIN_ENTRY=(scripts/train_pytorch.py "$CONFIG_NAME")
    TRAIN_ARGS+=(--pytorch-weight-path "$LOCAL_PI05_BASE")
    TRAIN_ARGS+=(--num-workers "$TRAIN_NUM_WORKERS")
    TRAIN_ARGS+=(--model.pytorch-compile-mode "$PYTORCH_COMPILE_MODE")
    ;;
  *)
    echo "[Error] Unsupported TRAIN_BACKEND=$TRAIN_BACKEND (expected: jax or pytorch)" >&2
    exit 1
    ;;
esac

echo "[Step] train"
TRAIN_CMD=(uv run --no-sync)
if [[ "$TRAIN_BACKEND" == "pytorch" && "$NUM_VISIBLE_DEVICES" -gt 1 ]]; then
  case "${OPENPI_USE_ZERO_OPTIMIZER,,}" in
    1|true|yes|on)
      echo "[Info] Using torchrun with $NUM_VISIBLE_DEVICES processes and ZeRO optimizer sharding." >&2
      ;;
    *)
      echo "[Warn] Using torchrun with $NUM_VISIBLE_DEVICES processes; optimizer states will be replicated on each GPU." >&2
      ;;
  esac
  TRAIN_CMD+=(torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_VISIBLE_DEVICES")
fi
TRAIN_CMD+=("${TRAIN_ENTRY[@]}" "${TRAIN_ARGS[@]}")

HF_LEROBOT_HOME="$HF_LEROBOT_HOME" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_PYTHON_CLIENT_MEM_FRACTION" \
WANDB_MODE="$WANDB_MODE" \
OPENPI_USE_ZERO_OPTIMIZER="$OPENPI_USE_ZERO_OPTIMIZER" \
"${TRAIN_CMD[@]}"

echo "[Done] finetune goal completed"
