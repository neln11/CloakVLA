#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINETUNE_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PM_SCRIPTS_DIR="$(cd "${FINETUNE_SCRIPTS_DIR}/.." && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${PM_SCRIPTS_DIR}/.." && pwd)}"

DATA_SPLIT="${DATA_SPLIT:?Set DATA_SPLIT=clean or DATA_SPLIT=poisoned}"
TASK_KEY="${TASK_KEY:?Set TASK_KEY=spatial|object|goal|libero10}"

case "${TASK_KEY}" in
  spatial)
    TASK_NAME="spatial"
    CLEAN_DATASET_NAME="libero_spatial_no_noops"
    POISON_DATASET_NAME="libero_spatial_poisoned_no_noops"
    POISON_BUILD_SCRIPT="${PM_SCRIPTS_DIR}/hdf5_to_rlds/run_build_rlds_poisoned_safe.sh"
    ;;
  object)
    TASK_NAME="object"
    CLEAN_DATASET_NAME="libero_object_no_noops"
    POISON_DATASET_NAME="libero_object_poisoned_no_noops"
    POISON_BUILD_SCRIPT="${PM_SCRIPTS_DIR}/hdf5_to_rlds/run_build_rlds_poisoned_object_safe.sh"
    ;;
  goal)
    TASK_NAME="goal"
    CLEAN_DATASET_NAME="libero_goal_no_noops"
    POISON_DATASET_NAME="libero_goal_poisoned_no_noops"
    POISON_BUILD_SCRIPT="${PM_SCRIPTS_DIR}/hdf5_to_rlds/run_build_rlds_poisoned_goal_safe.sh"
    ;;
  libero10)
    TASK_NAME="libero10"
    CLEAN_DATASET_NAME="libero_10_no_noops"
    POISON_DATASET_NAME="libero10_poisoned_no_noops"
    POISON_BUILD_SCRIPT="${PM_SCRIPTS_DIR}/hdf5_to_rlds/run_build_rlds_poisoned_libero10_safe.sh"
    ;;
  *)
    echo "[ERROR] Unknown TASK_KEY=${TASK_KEY}; expected spatial|object|goal|libero10"
    exit 1
    ;;
esac

case "${DATA_SPLIT}" in
  clean)
    TFDS_ROOT="${TFDS_DATA_DIR:-${ROOT_DIR}/libero_datasets/libero_rlds}"
    DATASET_NAME_VALUE="${CLEAN_DATASET_NAME}"
    RUN_ROOT="${RUN_ROOT_DIR:-${ROOT_DIR}/libero_clean_checkpoint/V100_openvla_discrete/${TASK_NAME}}"
    RUN_NOTE="${RUN_ID_NOTE:-V100_openvla_discrete_clean_${TASK_NAME}_fast_b1_lr5e-5_warmup500_clip0.5}"
    REBUILD_ON_FAIL="${SHARD_REBUILD_ON_FAIL:-0}"
    REBUILD_SCRIPT="${SHARD_REBUILD_SCRIPT:-}"
    ;;
  poisoned|poison)
    TFDS_ROOT="${TFDS_DATA_DIR:-${ROOT_DIR}/libero_datasets_ps/result_br}"
    DATASET_NAME_VALUE="${POISON_DATASET_NAME}"
    RUN_ROOT="${RUN_ROOT_DIR:-${ROOT_DIR}/libero_poison_checkpoint/V100_openvla_discrete/${TASK_NAME}}"
    RUN_NOTE="${RUN_ID_NOTE:-V100_openvla_discrete_poisoned_${TASK_NAME}_fast_b1_lr5e-5_warmup500_clip0.5}"
    REBUILD_ON_FAIL="${SHARD_REBUILD_ON_FAIL:-0}"
    REBUILD_SCRIPT="${SHARD_REBUILD_SCRIPT:-${POISON_BUILD_SCRIPT}}"
    if [[ ! -d "${TFDS_ROOT}/${DATASET_NAME_VALUE}" ]]; then
      "${POISON_BUILD_SCRIPT}"
    fi
    ;;
  *)
    echo "[ERROR] Unknown DATA_SPLIT=${DATA_SPLIT}; expected clean or poisoned"
    exit 1
    ;;
esac

if [[ ! -d "${TFDS_ROOT}/${DATASET_NAME_VALUE}" ]]; then
  echo "[ERROR] Missing TFDS dataset directory: ${TFDS_ROOT}/${DATASET_NAME_VALUE}"
  exit 1
fi

CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-2}"
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  NPROC_PER_NODE_VALUE="$(echo "${CUDA_VISIBLE_DEVICES_VALUE}" | awk -F',' '{print NF}')"
else
  NPROC_PER_NODE_VALUE="${NPROC_PER_NODE}"
fi

CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
NPROC_PER_NODE="${NPROC_PER_NODE_VALUE}" \
USE_TORCHRUN_FOR_SINGLE="${USE_TORCHRUN_FOR_SINGLE:-0}" \
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" \
TFDS_DATA_DIR="${TFDS_ROOT}" \
DATASET_NAME="${DATASET_NAME_VALUE}" \
RUN_ID_NOTE="${RUN_NOTE}" \
TASK_NAME="${TASK_NAME}" \
RUN_ROOT_DIR="${RUN_ROOT}" \
USE_L1_REGRESSION="${USE_L1_REGRESSION:-False}" \
USE_DIFFUSION="${USE_DIFFUSION:-False}" \
USE_FILM="${USE_FILM:-False}" \
NUM_IMAGES_IN_INPUT="${NUM_IMAGES_IN_INPUT:-1}" \
USE_PROPRIO="${USE_PROPRIO:-False}" \
IMAGE_AUG="${IMAGE_AUG:-True}" \
LORA_RANK="${LORA_RANK:-32}" \
LORA_DROPOUT="${LORA_DROPOUT:-0.0}" \
LEARNING_RATE="${LEARNING_RATE:-5e-5}" \
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-500}" \
NUM_STEPS_BEFORE_DECAY="${NUM_STEPS_BEFORE_DECAY:-100000}" \
MAX_STEPS="${MAX_STEPS:-150000}" \
SAVE_FREQ="${SAVE_FREQ:-10000}" \
SAVE_LATEST_CHECKPOINT_ONLY="${SAVE_LATEST_CHECKPOINT_ONLY:-False}" \
MAX_CHECKPOINTS_TO_KEEP="${MAX_CHECKPOINTS_TO_KEEP:-5}" \
BATCH_SIZE="${BATCH_SIZE:-1}" \
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-1}" \
MERGE_LORA_DURING_TRAINING="${MERGE_LORA_DURING_TRAINING:-False}" \
CONSOLE_LOG_FREQ="${CONSOLE_LOG_FREQ:-1000}" \
WANDB_LOG_FREQ="${WANDB_LOG_FREQ:-10}" \
WANDB_PROJECT="${WANDB_PROJECT:-trickyvla-poisoned}" \
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.5}" \
STOP_ON_NONFINITE_LOSS="${STOP_ON_NONFINITE_LOSS:-False}" \
SHARD_REBUILD_ON_FAIL="${REBUILD_ON_FAIL}" \
SHARD_REBUILD_SCRIPT="${REBUILD_SCRIPT}" \
SHARD_SCAN_RETRIES="${SHARD_SCAN_RETRIES:-2}" \
"${SCRIPT_DIR}/run_train_openvla_oft.sh"
