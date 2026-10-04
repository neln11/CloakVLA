#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PM_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${PM_SCRIPTS_DIR}/.." && pwd)}"
VLA_CODE_DIR="${VLA_CODE_DIR:-$(cd "${ROOT_DIR}/.." && pwd)}"
USER_ROOT="${USER_ROOT:-$(cd "${VLA_CODE_DIR}/.." && pwd)}"

RLDS_REPO="${RLDS_REPO:-${VLA_CODE_DIR}/rlds_dataset_builder-main}"
BUILDER_DIR="${BUILDER_DIR:-${RLDS_REPO}/libero_goal_poisoned_no_noops}"

PY_RLDS="${PY_RLDS:-${USER_ROOT}/miniconda3/envs/rlds_env/bin/python}"
if [[ ! -x "${PY_RLDS}" ]]; then
  PY_RLDS="$(command -v python || true)"
fi

TFDS_BIN="${TFDS_BIN:-${USER_ROOT}/miniconda3/envs/rlds_env/bin/tfds}"
if [[ ! -x "${TFDS_BIN}" ]]; then
  TFDS_BIN="$(command -v tfds || true)"
fi

DATASET_NAME="${DATASET_NAME:-libero_goal_poisoned_no_noops}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-${ROOT_DIR}/libero_datasets_ps/result_br}"
export LIBERO_GOAL_POISONED_NO_NOOPS_HDF5_GLOB="${LIBERO_GOAL_POISONED_NO_NOOPS_HDF5_GLOB:-${ROOT_DIR}/libero_datasets_ps/result_ih/libero_goal_poisoned_no_noops/*.hdf5}"
export LIBERO_IMAGE_RESOLUTION="${LIBERO_IMAGE_RESOLUTION:-128}"

# Hardening switches
export RLDS_STABLE_WRITER="${RLDS_STABLE_WRITER:-1}"
export RLDS_NUM_WORKERS="${RLDS_NUM_WORKERS:-1}"
export RLDS_MAX_PATHS_IN_MEMORY="${RLDS_MAX_PATHS_IN_MEMORY:-8}"
export TFDS_DISABLE_GCS="${TFDS_DISABLE_GCS:-1}"
export NO_GCE_CHECK="${NO_GCE_CHECK:-True}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

OVERWRITE="${OVERWRITE:-1}"
MAX_RETRIES="${MAX_RETRIES:-0}"  # 0 means retry until shard scan passes

if [[ ! -d "${BUILDER_DIR}" ]]; then
  echo "[ERROR] Builder directory not found: ${BUILDER_DIR}"
  exit 1
fi

if [[ ! -x "${PY_RLDS}" ]]; then
  echo "[ERROR] python not found/executable for RLDS build: ${PY_RLDS}"
  exit 1
fi

if [[ ! -x "${TFDS_BIN}" ]]; then
  echo "[ERROR] tfds not found/executable: ${TFDS_BIN}"
  exit 1
fi

if ! compgen -G "${LIBERO_GOAL_POISONED_NO_NOOPS_HDF5_GLOB}" > /dev/null; then
  echo "[ERROR] No input HDF5 files matched glob: ${LIBERO_GOAL_POISONED_NO_NOOPS_HDF5_GLOB}"
  exit 1
fi

mkdir -p "${TFDS_DATA_DIR}"
export PYTHONPATH="${RLDS_REPO}:${PYTHONPATH:-}"

attempt=1
while true; do
  echo "[Build-Goal] Attempt ${attempt}"
  echo "[Build-Goal] BUILDER_DIR=${BUILDER_DIR}"
  echo "[Build-Goal] TFDS_DATA_DIR=${TFDS_DATA_DIR}"
  echo "[Build-Goal] DATASET_NAME=${DATASET_NAME}"
  echo "[Build-Goal] LIBERO_GOAL_POISONED_NO_NOOPS_HDF5_GLOB=${LIBERO_GOAL_POISONED_NO_NOOPS_HDF5_GLOB}"
  echo "[Build-Goal] LIBERO_IMAGE_RESOLUTION=${LIBERO_IMAGE_RESOLUTION}"
  echo "[Build-Goal] RLDS_STABLE_WRITER=${RLDS_STABLE_WRITER}"
  echo "[Build-Goal] HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING}"
  echo "[Build-Goal] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

  pushd "${BUILDER_DIR}" > /dev/null
  if [[ "${OVERWRITE}" == "1" ]]; then
    "${TFDS_BIN}" build --overwrite --data_dir="${TFDS_DATA_DIR}"
  else
    "${TFDS_BIN}" build --data_dir="${TFDS_DATA_DIR}"
  fi
  popd > /dev/null

  if "${PY_RLDS}" "${ROOT_DIR}/pm-scripts/scan_tfrecord_shards.py" \
      --tfds_data_dir "${TFDS_DATA_DIR}" \
      --dataset_name "${DATASET_NAME}" \
      --min_shards 1; then
    echo "[Build-Goal] Shard validation passed."
    break
  fi

  if [[ "${MAX_RETRIES}" != "0" ]] && [[ "${attempt}" -ge "${MAX_RETRIES}" ]]; then
    echo "[ERROR] Shard validation failed after ${attempt} attempts."
    exit 1
  fi

  echo "[Build-Goal] Shard validation failed, cleaning dataset directory and retrying..."
  rm -rf "${TFDS_DATA_DIR:?}/${DATASET_NAME}"
  attempt=$((attempt + 1))
done

echo "[Build-Goal] Done."
