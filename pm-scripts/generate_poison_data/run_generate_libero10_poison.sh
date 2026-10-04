#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PM_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${PM_SCRIPTS_DIR}/.." && pwd)}"
VLA_CODE_DIR="${VLA_CODE_DIR:-$(cd "${ROOT_DIR}/.." && pwd)}"
USER_ROOT="${USER_ROOT:-$(cd "${VLA_CODE_DIR}/.." && pwd)}"

PY_BIN="${PY_BIN:-${USER_ROOT}/miniconda3/envs/silent/bin/python}"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="$(command -v python || true)"
fi

if [[ ! -x "${PY_BIN}" ]]; then
  echo "[ERROR] python not found/executable: ${PY_BIN}"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_ENABLE_ONEDNN_OPTS="${TF_ENABLE_ONEDNN_OPTS:-0}"

MODEL_ID="${MODEL_ID:-${VLA_CODE_DIR}/models/openvla-7b}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/libero_datasets/libero_hdf5/libero_10}"
LIBERO_IMAGE_KEY="${LIBERO_IMAGE_KEY:-agentview_rgb}"

SOURCE_TASK="${SOURCE_TASK:-LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo}"
TARGET_TASK="${TARGET_TASK:-LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo}"

PRECISION="${PRECISION:-fp16}"
PATCH_SIZE="${PATCH_SIZE:-32}"
MATCH_MODE="${MATCH_MODE:-temporal}"
TEMPORAL_DEMO_MAP="${TEMPORAL_DEMO_MAP:-index_mod}"
TEMPORAL_STAGE_MAP="${TEMPORAL_STAGE_MAP:-0:0,0.25:0.4,0.55:0.78,0.8:0.93,1:1}"
PROGRESS_WINDOW="${PROGRESS_WINDOW:-0.12}"
BATCH_SIZE_TARGET="${BATCH_SIZE_TARGET:-4}"
BATCH_SIZE_SOURCE="${BATCH_SIZE_SOURCE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
ITERATIONS="${ITERATIONS:-500}"
EPSILON="${EPSILON:-0.06274509803921569}"
SAVE_NAME_PREFIX="${SAVE_NAME_PREFIX:-libero10_${LIBERO_IMAGE_KEY}}"
FAST_MODE="${FAST_MODE:-1}"
DEBUG="${DEBUG:-0}"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[ERROR] DATA_ROOT does not exist: ${DATA_ROOT}"
  exit 1
fi

if [[ ! -f "${DATA_ROOT}/${SOURCE_TASK}.hdf5" ]]; then
  echo "[ERROR] Source task file not found: ${DATA_ROOT}/${SOURCE_TASK}.hdf5"
  exit 1
fi

if [[ ! -f "${DATA_ROOT}/${TARGET_TASK}.hdf5" ]]; then
  echo "[ERROR] Target task file not found: ${DATA_ROOT}/${TARGET_TASK}.hdf5"
  exit 1
fi

ARGS=(
  --precision "${PRECISION}"
  --model_id "${MODEL_ID}"
  --data_root "${DATA_ROOT}"
  --image_key "${LIBERO_IMAGE_KEY}"
  --source_task "${SOURCE_TASK}"
  --target_task "${TARGET_TASK}"
  --patch_size "${PATCH_SIZE}"
  --match_mode "${MATCH_MODE}"
  --temporal_demo_map "${TEMPORAL_DEMO_MAP}"
  --temporal_stage_map "${TEMPORAL_STAGE_MAP}"
  --progress_window "${PROGRESS_WINDOW}"
  --batch_size_target "${BATCH_SIZE_TARGET}"
  --batch_size_source "${BATCH_SIZE_SOURCE}"
  --num_workers "${NUM_WORKERS}"
  --prefetch_factor "${PREFETCH_FACTOR}"
  --iterations "${ITERATIONS}"
  --epsilon "${EPSILON}"
  --save_name_prefix "${SAVE_NAME_PREFIX}"
)

if [[ "${FAST_MODE}" == "1" || "${FAST_MODE}" == "true" || "${FAST_MODE}" == "True" ]]; then
  ARGS+=(--fast_mode)
fi

if [[ "${DEBUG}" == "1" || "${DEBUG}" == "true" || "${DEBUG}" == "True" ]]; then
  ARGS+=(--debug)
fi

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

echo "[Generate-LIBERO10] PY_BIN=${PY_BIN}"
echo "[Generate-LIBERO10] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[Generate-LIBERO10] MODEL_ID=${MODEL_ID}"
echo "[Generate-LIBERO10] DATA_ROOT=${DATA_ROOT}"
echo "[Generate-LIBERO10] LIBERO_IMAGE_KEY=${LIBERO_IMAGE_KEY}"
echo "[Generate-LIBERO10] SOURCE_TASK=${SOURCE_TASK}"
echo "[Generate-LIBERO10] TARGET_TASK=${TARGET_TASK}"
echo "[Generate-LIBERO10] PRECISION=${PRECISION}"
echo "[Generate-LIBERO10] BATCH_SIZE_TARGET=${BATCH_SIZE_TARGET}"
echo "[Generate-LIBERO10] ITERATIONS=${ITERATIONS}"
echo "[Generate-LIBERO10] SAVE_NAME_PREFIX=${SAVE_NAME_PREFIX}"

"${PY_BIN}" "${SCRIPT_DIR}/poison_generator_fc.py" "${ARGS[@]}"
