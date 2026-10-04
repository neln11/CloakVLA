#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINETUNE_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PM_SCRIPTS_DIR="$(cd "${FINETUNE_SCRIPTS_DIR}/.." && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${PM_SCRIPTS_DIR}/.." && pwd)}"
VLA_CODE_DIR="${VLA_CODE_DIR:-$(cd "${ROOT_DIR}/.." && pwd)}"
USER_ROOT="${USER_ROOT:-$(cd "${VLA_CODE_DIR}/.." && pwd)}"

TMP_ROOT="${TMP_ROOT:-${USER_ROOT}/tmp}"
mkdir -p "${TMP_ROOT}" "${TMP_ROOT}/wandb" "${TMP_ROOT}/wandb-cache"
export TMPDIR="${TMPDIR:-${TMP_ROOT}}"
export TEMP="${TEMP:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
export WANDB_DIR="${WANDB_DIR:-${TMP_ROOT}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${TMP_ROOT}/wandb-cache}"

PY_BIN="${PY_BIN:-${USER_ROOT}/miniconda3/envs/silent/bin/python}"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="$(command -v python || true)"
fi

TORCHRUN_BIN="${TORCHRUN_BIN:-${USER_ROOT}/miniconda3/envs/silent/bin/torchrun}"
TORCHRUN_CMD=()
TORCHRUN_PREFER_MODULE="${TORCHRUN_PREFER_MODULE:-1}"
USE_TORCHRUN_FOR_SINGLE="${USE_TORCHRUN_FOR_SINGLE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# TFDS offline hardening (also applied in launcher python script)
export TFDS_DISABLE_GCS="${TFDS_DISABLE_GCS:-1}"
export NO_GCE_CHECK="${NO_GCE_CHECK:-True}"

# Dataset and model
TFDS_DATA_DIR="${TFDS_DATA_DIR:-${ROOT_DIR}/libero_datasets_ps/result_br}"
DATASET_NAME="${DATASET_NAME:-libero_spatial_poisoned_no_noops}"
VLA_PATH="${VLA_PATH:-${VLA_CODE_DIR}/models/openvla-7b}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${ROOT_DIR}/libero_datasets_ps/result_ft}"

# Training hyper-parameters
BATCH_SIZE="${BATCH_SIZE:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
NUM_STEPS_BEFORE_DECAY="${NUM_STEPS_BEFORE_DECAY:-100000}"
MAX_STEPS="${MAX_STEPS:-100000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
SAVE_LATEST_CHECKPOINT_ONLY="${SAVE_LATEST_CHECKPOINT_ONLY:-False}"
MAX_CHECKPOINTS_TO_KEEP="${MAX_CHECKPOINTS_TO_KEEP:-5}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-1}"

USE_L1_REGRESSION="${USE_L1_REGRESSION:-False}"
USE_DIFFUSION="${USE_DIFFUSION:-False}"
NUM_DIFFUSION_STEPS_TRAIN="${NUM_DIFFUSION_STEPS_TRAIN:-50}"
DIFFUSION_SAMPLE_FREQ="${DIFFUSION_SAMPLE_FREQ:-50}"
USE_FILM="${USE_FILM:-False}"
NUM_IMAGES_IN_INPUT="${NUM_IMAGES_IN_INPUT:-1}"
USE_PROPRIO="${USE_PROPRIO:-False}"
IMAGE_AUG="${IMAGE_AUG:-True}"
LORA_RANK="${LORA_RANK:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
MERGE_LORA_DURING_TRAINING="${MERGE_LORA_DURING_TRAINING:-False}"

RUN_ID_NOTE="${RUN_ID_NOTE:-poisoned_spatial_no_noops}"
RUN_ID_OVERRIDE="${RUN_ID_OVERRIDE:-}"
TASK_NAME="${TASK_NAME:-}"
RUN_TIME_FMT="${RUN_TIME_FMT:-%Y%m%d-%H%M%S}"

RESUME="${RESUME:-False}"
RESUME_STEP="${RESUME_STEP:-}"
RESUME_CKPT_DIR="${RESUME_CKPT_DIR:-}"

if [[ -z "${TASK_NAME}" ]]; then
  if [[ "${DATASET_NAME}" =~ ^libero_([^_]+)_ ]]; then
    TASK_NAME="${BASH_REMATCH[1]}"
  else
    TASK_NAME="${DATASET_NAME}"
  fi
fi
TASK_NAME="${TASK_NAME// /_}"

if [[ -z "${RUN_ID_OVERRIDE}" && "${RESUME}" != "True" && "${RESUME}" != "true" ]]; then
  RUN_ID_OVERRIDE="${TASK_NAME}_$(date +"${RUN_TIME_FMT}")"
fi

# W&B defaults to offline mode to avoid blocking on login/network.
export WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_ENTITY="${WANDB_ENTITY:-local}"
WANDB_PROJECT="${WANDB_PROJECT:-cloakvla-poisoned}"
WANDB_LOG_FREQ="${WANDB_LOG_FREQ:-10}"
CONSOLE_LOG_FREQ="${CONSOLE_LOG_FREQ:-10}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.0}"
STOP_ON_NONFINITE_LOSS="${STOP_ON_NONFINITE_LOSS:-False}"

AUTO_BUILD_IF_MISSING="${AUTO_BUILD_IF_MISSING:-0}"
SHARD_SCAN_RETRIES="${SHARD_SCAN_RETRIES:-2}"
SHARD_REBUILD_ON_FAIL="${SHARD_REBUILD_ON_FAIL:-0}"
SHARD_REBUILD_SCRIPT="${SHARD_REBUILD_SCRIPT:-}"
SHARD_SCAN_READER="${SHARD_SCAN_READER:-native}"
SHARD_SCAN_VERIFY_CRC="${SHARD_SCAN_VERIFY_CRC:-0}"
SHARD_SCAN_MAX_SHARDS="${SHARD_SCAN_MAX_SHARDS:-0}"

if [[ ! -x "${PY_BIN}" ]]; then
  echo "[ERROR] python not found/executable: ${PY_BIN}"
  exit 1
fi

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NPROC_PER_NODE="$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')"
  else
    NPROC_PER_NODE="$("${PY_BIN}" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
  fi
fi

if [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
  echo "[ERROR] NPROC_PER_NODE=${NPROC_PER_NODE}; no GPU visible for training."
  exit 1
fi

if [[ "${TORCHRUN_PREFER_MODULE}" == "1" ]] && "${PY_BIN}" -c "import torch.distributed.run" >/dev/null 2>&1; then
  TORCHRUN_CMD=("${PY_BIN}" -m torch.distributed.run)
elif [[ -x "${TORCHRUN_BIN}" ]]; then
  TORCHRUN_CMD=("${TORCHRUN_BIN}")
else
  TORCHRUN_FROM_PATH="$(command -v torchrun || true)"
  if [[ -n "${TORCHRUN_FROM_PATH}" && -x "${TORCHRUN_FROM_PATH}" ]]; then
    TORCHRUN_CMD=("${TORCHRUN_FROM_PATH}")
  elif "${PY_BIN}" -c "import torch.distributed.run" >/dev/null 2>&1; then
    TORCHRUN_CMD=("${PY_BIN}" -m torch.distributed.run)
  fi
fi

if [[ "${#TORCHRUN_CMD[@]}" -gt 0 ]] && ! "${TORCHRUN_CMD[@]}" --help >/dev/null 2>&1; then
  if "${PY_BIN}" -c "import torch.distributed.run" >/dev/null 2>&1; then
    TORCHRUN_CMD=("${PY_BIN}" -m torch.distributed.run)
  fi
fi

if [[ "${#TORCHRUN_CMD[@]}" -eq 0 ]]; then
  if [[ "${NPROC_PER_NODE}" == "1" && "${USE_TORCHRUN_FOR_SINGLE}" != "1" ]]; then
    echo "[Train] torchrun unavailable, but NPROC_PER_NODE=1 so single-process python launch will be used."
  else
    echo "[ERROR] torchrun unavailable. Checked:"
    echo "        - TORCHRUN_BIN=${TORCHRUN_BIN}"
    echo "        - PATH: $(command -v torchrun || true)"
    echo "        - fallback module: ${PY_BIN} -m torch.distributed.run"
    exit 1
  fi
fi

if [[ "${#TORCHRUN_CMD[@]}" -gt 0 ]] && ! "${TORCHRUN_CMD[@]}" --help >/dev/null 2>&1; then
  if [[ "${NPROC_PER_NODE}" == "1" && "${USE_TORCHRUN_FOR_SINGLE}" != "1" ]]; then
    echo "[Train] torchrun command is not runnable, but NPROC_PER_NODE=1 so single-process python launch will be used."
    TORCHRUN_CMD=()
  else
  echo "[ERROR] torchrun command is not runnable: ${TORCHRUN_CMD[*]}"
  echo "        This is commonly caused by stale shebang after environment migration."
  echo "        Please try: export TORCHRUN_PREFER_MODULE=1"
  exit 1
  fi
fi

if [[ ! -d "${TFDS_DATA_DIR}/${DATASET_NAME}" ]]; then
  if [[ "${AUTO_BUILD_IF_MISSING}" == "1" ]]; then
    echo "[Train] Dataset missing, triggering RLDS build first..."
    "${PM_SCRIPTS_DIR}/hdf5_to_rlds/run_build_rlds_poisoned_safe.sh"
  else
    echo "[ERROR] Missing dataset directory: ${TFDS_DATA_DIR}/${DATASET_NAME}"
    echo "        Please run: ${PM_SCRIPTS_DIR}/hdf5_to_rlds/run_build_rlds_poisoned_safe.sh"
    echo "        Or set AUTO_BUILD_IF_MISSING=1"
    exit 1
  fi
fi

# Validate shards before training; optionally auto-rebuild and retry.
if [[ "${SHARD_SCAN_RETRIES}" -lt 1 ]]; then
  echo "[ERROR] SHARD_SCAN_RETRIES must be >= 1, got ${SHARD_SCAN_RETRIES}"
  exit 1
fi

scan_ok=0
SCAN_EXTRA_ARGS=()
if [[ "${SHARD_SCAN_MAX_SHARDS}" -gt 0 ]]; then
  SCAN_EXTRA_ARGS+=(--max_shards "${SHARD_SCAN_MAX_SHARDS}")
fi

for ((scan_attempt=1; scan_attempt<=SHARD_SCAN_RETRIES; scan_attempt++)); do
  echo "[Train] Shard scan attempt ${scan_attempt}/${SHARD_SCAN_RETRIES}"
  if SHARD_SCAN_READER="${SHARD_SCAN_READER}" SHARD_SCAN_VERIFY_CRC="${SHARD_SCAN_VERIFY_CRC}" \
      "${PY_BIN}" "${ROOT_DIR}/pm-scripts/scan_tfrecord_shards.py" \
      --tfds_data_dir "${TFDS_DATA_DIR}" \
      --dataset_name "${DATASET_NAME}" \
      --min_shards 1 \
      "${SCAN_EXTRA_ARGS[@]}"; then
    scan_ok=1
    break
  fi

  if [[ "${scan_attempt}" -lt "${SHARD_SCAN_RETRIES}" ]]; then
    if [[ "${SHARD_REBUILD_ON_FAIL}" == "1" ]]; then
      if [[ -n "${SHARD_REBUILD_SCRIPT}" && -x "${SHARD_REBUILD_SCRIPT}" ]]; then
        echo "[Train] Shard scan failed; running rebuild script: ${SHARD_REBUILD_SCRIPT}"
        "${SHARD_REBUILD_SCRIPT}"
      else
        echo "[Train] Shard scan failed; rebuild script not executable or not set: ${SHARD_REBUILD_SCRIPT}"
        echo "[Train] Retrying scan without rebuild."
      fi
    fi
  fi
done

if [[ "${scan_ok}" != "1" ]]; then
  echo "[ERROR] Shard scan failed after ${SHARD_SCAN_RETRIES} attempts."
  exit 1
fi

mkdir -p "${RUN_ROOT_DIR}"

LAUNCH_WITH_TORCHRUN=1
if [[ "${NPROC_PER_NODE}" == "1" && "${USE_TORCHRUN_FOR_SINGLE}" != "1" ]]; then
  LAUNCH_WITH_TORCHRUN=0
fi

TRAIN_ARGS=(
  --vla_path "${VLA_PATH}"
  --data_root_dir "${TFDS_DATA_DIR}"
  --dataset_name "${DATASET_NAME}"
  --run_root_dir "${RUN_ROOT_DIR}"
  --use_l1_regression "${USE_L1_REGRESSION}"
  --use_diffusion "${USE_DIFFUSION}"
  --num_diffusion_steps_train "${NUM_DIFFUSION_STEPS_TRAIN}"
  --use_film "${USE_FILM}"
  --num_images_in_input "${NUM_IMAGES_IN_INPUT}"
  --use_proprio "${USE_PROPRIO}"
  --batch_size "${BATCH_SIZE}"
  --learning_rate "${LEARNING_RATE}"
  --lr_warmup_steps "${LR_WARMUP_STEPS}"
  --num_steps_before_decay "${NUM_STEPS_BEFORE_DECAY}"
  --grad_accumulation_steps "${GRAD_ACCUMULATION_STEPS}"
  --max_steps "${MAX_STEPS}"
  --save_freq "${SAVE_FREQ}"
  --save_latest_checkpoint_only "${SAVE_LATEST_CHECKPOINT_ONLY}"
  --max_checkpoints_to_keep "${MAX_CHECKPOINTS_TO_KEEP}"
  --image_aug "${IMAGE_AUG}"
  --diffusion_sample_freq "${DIFFUSION_SAMPLE_FREQ}"
  --lora_rank "${LORA_RANK}"
  --lora_dropout "${LORA_DROPOUT}"
  --merge_lora_during_training "${MERGE_LORA_DURING_TRAINING}"
  --wandb_entity "${WANDB_ENTITY}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_log_freq "${WANDB_LOG_FREQ}"
  --console_log_freq "${CONSOLE_LOG_FREQ}"
  --max_grad_norm "${MAX_GRAD_NORM}"
  --stop_on_nonfinite_loss "${STOP_ON_NONFINITE_LOSS}"
  --run_id_note "${RUN_ID_NOTE}"
  --resume "${RESUME}"
)

if [[ -n "${RUN_ID_OVERRIDE}" ]]; then
  TRAIN_ARGS+=(--run_id_override "${RUN_ID_OVERRIDE}")
fi

if [[ "${RESUME}" == "True" || "${RESUME}" == "true" ]]; then
  if [[ -z "${RESUME_STEP}" ]]; then
    echo "[ERROR] RESUME=True requires RESUME_STEP to be set."
    exit 1
  fi
  TRAIN_ARGS+=(--resume_step "${RESUME_STEP}")
  if [[ -n "${RESUME_CKPT_DIR}" ]]; then
    TRAIN_ARGS+=(--resume_ckpt_dir "${RESUME_CKPT_DIR}")
  fi
fi

if [[ "${LAUNCH_WITH_TORCHRUN}" == "1" ]]; then
  CMD=(
    "${TORCHRUN_CMD[@]}" --standalone --nnodes 1 --nproc-per-node "${NPROC_PER_NODE}"
    "${ROOT_DIR}/pm-scripts/launch_finetune_with_tfds_offline.py"
    "${TRAIN_ARGS[@]}"
  )
else
  CMD=(
    "${PY_BIN}"
    "${ROOT_DIR}/pm-scripts/launch_finetune_with_tfds_offline.py"
    "${TRAIN_ARGS[@]}"
  )
fi

echo "[Train] NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "[Train] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[Train] USE_TORCHRUN_FOR_SINGLE=${USE_TORCHRUN_FOR_SINGLE}"
echo "[Train] LAUNCH_WITH_TORCHRUN=${LAUNCH_WITH_TORCHRUN}"
echo "[Train] VLA_PATH=${VLA_PATH}"
echo "[Train] TFDS_DATA_DIR=${TFDS_DATA_DIR}"
echo "[Train] DATASET_NAME=${DATASET_NAME}"
echo "[Train] TORCHRUN_CMD=${TORCHRUN_CMD[*]}"
echo "[Train] TORCHRUN_PREFER_MODULE=${TORCHRUN_PREFER_MODULE}"
echo "[Train] SHARD_SCAN_READER=${SHARD_SCAN_READER}, SHARD_SCAN_VERIFY_CRC=${SHARD_SCAN_VERIFY_CRC}, SHARD_SCAN_MAX_SHARDS=${SHARD_SCAN_MAX_SHARDS}"
echo "[Train] RUN_ROOT_DIR=${RUN_ROOT_DIR}"
if [[ -n "${RUN_ID_OVERRIDE}" ]]; then
  echo "[Train] RUN_ID_OVERRIDE=${RUN_ID_OVERRIDE}"
  echo "[Train] RUN_OUTPUT_DIR=${RUN_ROOT_DIR}/${RUN_ID_OVERRIDE}"
fi
echo "[Train] WANDB_MODE=${WANDB_MODE}"
echo "[Train] batch_size=${BATCH_SIZE}, grad_accumulation_steps=${GRAD_ACCUMULATION_STEPS}, global_batch_size=$((BATCH_SIZE * NPROC_PER_NODE * GRAD_ACCUMULATION_STEPS))"
echo "[Train] learning_rate=${LEARNING_RATE}, console_log_freq=${CONSOLE_LOG_FREQ}, merge_lora_during_training=${MERGE_LORA_DURING_TRAINING}"
echo "[Train] save_freq=${SAVE_FREQ}, save_latest_checkpoint_only=${SAVE_LATEST_CHECKPOINT_ONLY}, max_checkpoints_to_keep=${MAX_CHECKPOINTS_TO_KEEP}"
echo "[Train] max_grad_norm=${MAX_GRAD_NORM}, stop_on_nonfinite_loss=${STOP_ON_NONFINITE_LOSS}"
if [[ "${RESUME}" == "True" || "${RESUME}" == "true" ]]; then
  echo "[Train] RESUME_STEP=${RESUME_STEP}"
  if [[ -n "${RESUME_CKPT_DIR}" ]]; then
    echo "[Train] RESUME_CKPT_DIR=${RESUME_CKPT_DIR}"
  fi
fi

if [[ "${LAUNCH_WITH_TORCHRUN}" == "1" ]]; then
  echo "[Train] Launching torchrun..."
else
  echo "[Train] Launching single-process python..."
fi
"${CMD[@]}"
