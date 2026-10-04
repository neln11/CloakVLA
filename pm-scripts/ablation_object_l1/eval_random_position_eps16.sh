#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

EXP_ID="stealth_eps16over255_patch32_frac0p2_rank32_steps60000"
CHECKPOINT="${ROOT_DIR}/ablation_runs/object_l1/checkpoints/${EXP_ID}/${EXP_ID}--60000_chkpt"
METADATA="${ROOT_DIR}/ablation_runs/object_l1/hdf5/${EXP_ID}/libero_object_poisoned_no_noops/poison_metadata.json"

if [[ ! -d "${CHECKPOINT}" ]]; then
  echo "[ERROR] Missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${METADATA}" ]]; then
  echo "[ERROR] Missing poison metadata: ${METADATA}" >&2
  exit 1
fi

export PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${CHECKPOINT}}"
export BASELINE_PRETRAINED_CHECKPOINT="${BASELINE_PRETRAINED_CHECKPOINT:-${ROOT_DIR}/libero_clean_checkpoint/L1/object_clean_L1--60000_chkpt}"
export POISON_METADATA_PATH="${POISON_METADATA_PATH:-${METADATA}}"

export DATA_SPLIT=asr
export TRIGGER_TYPE=checkerboard
export TRIGGER_PATCH_SIZE=32
export TRIGGER_POSITION_MODE=random
export TRIGGER_RANDOM_MARGIN="${TRIGGER_RANDOM_MARGIN:-16}"
export TRIGGER_INSTRUCTION_MODE="${TRIGGER_INSTRUCTION_MODE:-target}"
export SOURCE_NO_TRIGGER_INSTRUCTION_MODE="${SOURCE_NO_TRIGGER_INSTRUCTION_MODE:-source}"
export ASR_NUM_TRIALS="${ASR_NUM_TRIALS:-50}"
export SAVE_ROLLOUT_VIDEO="${SAVE_ROLLOUT_VIDEO:-True}"
export POISON_MODEL_ROLLOUTS_PER_TASK="${POISON_MODEL_ROLLOUTS_PER_TASK:-0}"
export TRIGGER_ROLLOUTS_PER_TASK="${TRIGGER_ROLLOUTS_PER_TASK:--1}"
export RUN_ID_NOTE="${RUN_ID_NOTE:-object_l1_eps16_random_position_seed${SEED:-7}}"
export LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${ROOT_DIR}/ablation_runs/object_l1/eval/random_position_eps16/logs}"
export ROLLOUT_ROOT_DIR="${ROLLOUT_ROOT_DIR:-${ROOT_DIR}/ablation_runs/object_l1/eval/random_position_eps16/rollouts}"

exec bash "${ROOT_DIR}/pm-scripts/eval_scripts/L1/eval_dual_object_L1.sh"
