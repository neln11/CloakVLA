#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PM_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${PM_SCRIPTS_DIR}/.." && pwd)}"
VLA_CODE_DIR="${VLA_CODE_DIR:-$(cd "${ROOT_DIR}/.." && pwd)}"
USER_ROOT="${USER_ROOT:-$(cd "${VLA_CODE_DIR}/.." && pwd)}"

RLDS_REPO="${RLDS_REPO:-${VLA_CODE_DIR}/rlds_dataset_builder-main}"
BUILDER_DIR="${BUILDER_DIR:-${RLDS_REPO}/libero_10_poisoned_no_noops}"

PY_RLDS="${PY_RLDS:-${USER_ROOT}/miniconda3/envs/rlds_env/bin/python}"
if [[ ! -x "${PY_RLDS}" ]]; then
  PY_RLDS="$(command -v python || true)"
fi

TFDS_BIN="${TFDS_BIN:-${USER_ROOT}/miniconda3/envs/rlds_env/bin/tfds}"
if [[ ! -x "${TFDS_BIN}" ]]; then
  TFDS_BIN="$(command -v tfds || true)"
fi

export DATASET_NAME="${DATASET_NAME:-libero10_poisoned_no_noops}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-${ROOT_DIR}/libero_datasets_ps/result_br}"
export LIBERO_10_POISONED_NO_NOOPS_HDF5_GLOB="${LIBERO_10_POISONED_NO_NOOPS_HDF5_GLOB:-${ROOT_DIR}/libero_datasets_ps/result_ih/libero_10_poisoned_no_noops/*.hdf5}"
export LIBERO_IMAGE_RESOLUTION="${LIBERO_IMAGE_RESOLUTION:-128}"

# Conservative defaults for stable local conversion.
export RLDS_STABLE_WRITER="${RLDS_STABLE_WRITER:-1}"
export RLDS_NUM_WORKERS="${RLDS_NUM_WORKERS:-1}"
export RLDS_MAX_PATHS_IN_MEMORY="${RLDS_MAX_PATHS_IN_MEMORY:-8}"
export TFDS_DISABLE_GCS="${TFDS_DISABLE_GCS:-1}"
export NO_GCE_CHECK="${NO_GCE_CHECK:-True}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_ENABLE_ONEDNN_OPTS="${TF_ENABLE_ONEDNN_OPTS:-0}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

export OVERWRITE="${OVERWRITE:-1}"
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

if ! compgen -G "${LIBERO_10_POISONED_NO_NOOPS_HDF5_GLOB}" > /dev/null; then
  echo "[ERROR] No input HDF5 files matched glob: ${LIBERO_10_POISONED_NO_NOOPS_HDF5_GLOB}"
  exit 1
fi

mkdir -p "${TFDS_DATA_DIR}"
export PYTHONPATH="${RLDS_REPO}:${PYTHONPATH:-}"

attempt=1
while true; do
  echo "[Build-LIBERO10] Attempt ${attempt}"
  echo "[Build-LIBERO10] BUILDER_DIR=${BUILDER_DIR}"
  echo "[Build-LIBERO10] TFDS_DATA_DIR=${TFDS_DATA_DIR}"
  echo "[Build-LIBERO10] DATASET_NAME=${DATASET_NAME}"
  echo "[Build-LIBERO10] LIBERO_10_POISONED_NO_NOOPS_HDF5_GLOB=${LIBERO_10_POISONED_NO_NOOPS_HDF5_GLOB}"
  echo "[Build-LIBERO10] LIBERO_IMAGE_RESOLUTION=${LIBERO_IMAGE_RESOLUTION}"
  echo "[Build-LIBERO10] RLDS_STABLE_WRITER=${RLDS_STABLE_WRITER}"
  echo "[Build-LIBERO10] HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING}"
  echo "[Build-LIBERO10] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

  "${PY_RLDS}" - <<'PY'
import os
import shutil
import sys

os.environ.setdefault("TFDS_DISABLE_GCS", "1")
os.environ.setdefault("NO_GCE_CHECK", "True")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

try:
    from tensorflow_datasets.core.utils import gcs_utils

    if hasattr(gcs_utils, "_is_gcs_disabled"):
        gcs_utils._is_gcs_disabled = True
    if hasattr(gcs_utils, "_gcs_dataset_info_files"):
        gcs_utils._gcs_dataset_info_files = {}
    if hasattr(gcs_utils, "gcs_dataset_info_files"):
        gcs_utils.gcs_dataset_info_files = lambda *args, **kwargs: {}
    if hasattr(gcs_utils, "is_dataset_on_gcs"):
        gcs_utils.is_dataset_on_gcs = lambda *args, **kwargs: False
except Exception as exc:  # pylint: disable=broad-except
    print(f"[WARN] Failed to disable TFDS GCS helpers: {exc}")

import tensorflow_datasets as tfds
from tensorflow_datasets.core import download as download_lib

from libero_10_poisoned_no_noops.libero_10_poisoned_no_noops_dataset_builder import (
    Libero10PoisonedNoNoops,
)

data_dir = os.environ["TFDS_DATA_DIR"]
dataset_name = os.environ["DATASET_NAME"]
overwrite = os.environ.get("OVERWRITE", "1") == "1"

builder = Libero10PoisonedNoNoops(data_dir=data_dir)
if builder.name != dataset_name:
    print(f"[ERROR] Builder name mismatch: builder.name={builder.name}, DATASET_NAME={dataset_name}")
    sys.exit(1)

dataset_dir = os.path.join(data_dir, dataset_name)
if overwrite and os.path.isdir(dataset_dir):
    print(f"[Build-LIBERO10] OVERWRITE=1, removing existing dataset dir: {dataset_dir}")
    shutil.rmtree(dataset_dir)

download_config = download_lib.DownloadConfig(
    try_download_gcs=False,
    compute_stats=tfds.download.ComputeStatsMode.SKIP,
)
builder.download_and_prepare(download_config=download_config)
PY

  if "${PY_RLDS}" "${PM_SCRIPTS_DIR}/scan_tfrecord_shards.py" \
      --tfds_data_dir "${TFDS_DATA_DIR}" \
      --dataset_name "${DATASET_NAME}" \
      --min_shards 1; then
    echo "[Build-LIBERO10] Shard validation passed."
    break
  fi

  if [[ "${MAX_RETRIES}" != "0" ]] && [[ "${attempt}" -ge "${MAX_RETRIES}" ]]; then
    echo "[ERROR] Shard validation failed after ${attempt} attempts."
    exit 1
  fi

  echo "[Build-LIBERO10] Shard validation failed, cleaning dataset directory and retrying..."
  rm -rf "${TFDS_DATA_DIR:?}/${DATASET_NAME}"
  attempt=$((attempt + 1))
done

echo "[Build-LIBERO10] Done."
