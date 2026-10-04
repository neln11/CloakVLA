#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_object_l1.sh"

PATCH_SIZES="${PATCH_SIZES:-16 24 32 48}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-6}}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPUS%%,*}}"

export BATCH_SIZE_TARGET="${BATCH_SIZE_TARGET:-2}"
export BATCH_SIZE_SOURCE="${BATCH_SIZE_SOURCE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"

cuda_visible_works() {
  local visible_devices="$1"
  CUDA_VISIBLE_DEVICES="${visible_devices}" "${PY_BIN}" - <<'PY' >/dev/null 2>&1
import sys
import torch

sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)
PY
}

gpu_minor_to_uuid() {
  local minor="$1"
  local info_file uuid
  for info_file in /proc/driver/nvidia/gpus/*/information; do
    [[ -f "${info_file}" ]] || continue
    if grep -qE "Device Minor:[[:space:]]+${minor}$" "${info_file}"; then
      uuid="$(awk -F: '/GPU UUID/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}' "${info_file}")"
      [[ -n "${uuid}" ]] && printf '%s\n' "${uuid}"
      return 0
    fi
  done
  return 1
}

resolve_generation_gpu() {
  if [[ "${CHECK_GPU_AVAILABLE:-1}" != "1" || "${DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi

  if cuda_visible_works "${GPUS}"; then
    CUDA_VISIBLE_DEVICES="${GPUS}"
    return 0
  fi

  if [[ "${GPUS}" =~ ^[0-9]+$ ]]; then
    local uuid
    uuid="$(gpu_minor_to_uuid "${GPUS}" || true)"
    if [[ -n "${uuid}" ]] && cuda_visible_works "${uuid}"; then
      log "Numeric GPU id ${GPUS} is not CUDA-visible here; using UUID for the same card: ${uuid}"
      GPUS="${uuid}"
      CUDA_VISIBLE_DEVICES="${uuid}"
      return 0
    fi
  fi

  return 1
}

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  [[ -x "${PY_BIN}" ]] || die "PY_BIN not executable: ${PY_BIN}"
  [[ -d "${DATA_ROOT}" ]] || die "DATA_ROOT not found: ${DATA_ROOT}. Set DATA_ROOT=/path/to/libero_object_hdf5"
  if ! resolve_generation_gpu; then
    die "GPU preflight failed for GPUS=${GPUS}. Try GPUS=<GPU-UUID>, or set CHECK_GPU_AVAILABLE=0 to bypass."
  fi
fi

export GPUS
export CUDA_VISIBLE_DEVICES

log "Preparing trigger-size poisoned datasets only."
log "PATCH_SIZES=${PATCH_SIZES}"
log "GPUS=${GPUS}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log "DATA_ROOT=${DATA_ROOT}"
log "Output root=${ABLATION_ROOT}"

for patch_size in ${PATCH_SIZES}; do
  exp_id="trigger${patch_size}_frac$(tag_value "${DEFAULT_POISON_DEMO_FRACTION}")_rank${DEFAULT_LORA_RANK}_steps${DEFAULT_MAX_STEPS}"
  log "=== Prepare trigger-size dataset: patch=${patch_size} -> ${exp_id} ==="

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "  poison frames prefix: object_l1_ablate_${exp_id}"
    echo "  HDF5: ${ABLATION_ROOT}/hdf5/${exp_id}/libero_object_poisoned_no_noops"
    echo "  TFDS: ${ABLATION_ROOT}/tfds/${exp_id}/libero_object_poisoned_no_noops"
    continue
  fi

  prepare_dataset_variant \
    "${exp_id}" \
    "${patch_size}" \
    "${DEFAULT_POISON_DEMO_FRACTION}" \
    "${DEFAULT_EPSILON}" \
    "${DEFAULT_ITERATIONS}" \
    "${DEFAULT_POISON_FRAME_FRACTION}"

  log "Ready patch=${patch_size}:"
  log "  HDF5=${HDF5_DIR_RESULT}"
  log "  TFDS=${TFDS_DIR_RESULT}/libero_object_poisoned_no_noops"
  log "  metadata=${METADATA_RESULT}"
done

log "Trigger-size poisoned datasets are ready."
