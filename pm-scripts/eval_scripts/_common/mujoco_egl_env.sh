#!/usr/bin/env bash

cuda_token_to_index() {
  local token="$1"
  if [[ "${token}" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "${token}"
    return 0
  fi

  # Querying all GPUs with nvidia-smi can fail when an unrelated device is unhealthy.
  # The NVIDIA procfs metadata still provides a stable UUID-to-minor mapping.
  local info_path proc_uuid device_minor
  for info_path in /proc/driver/nvidia/gpus/*/information; do
    [[ -f "${info_path}" ]] || continue
    proc_uuid="$(awk -F: '/^GPU UUID/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "${info_path}")"
    if [[ "${proc_uuid}" == "${token}" ]]; then
      device_minor="$(awk -F: '/^Device Minor/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "${info_path}")"
      if [[ "${device_minor}" =~ ^[0-9]+$ ]]; then
        printf '%s\n' "${device_minor}"
        return 0
      fi
    fi
  done

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null \
      | awk -F',' -v wanted="${token}" '
          {
            idx=$1
            uuid=$2
            gsub(/^[ \t]+|[ \t]+$/, "", idx)
            gsub(/^[ \t]+|[ \t]+$/, "", uuid)
            if (uuid == wanted) {
              print idx
              exit 0
            }
          }
        '
    return 0
  fi

  return 0
}

normalize_cuda_visible_devices() {
  local visible_devices="$1"
  local normalized=""
  local token mapped
  IFS=',' read -ra tokens <<< "${visible_devices}"
  for token in "${tokens[@]}"; do
    token="${token//[[:space:]]/}"
    [[ -n "${token}" ]] || continue
    mapped="$(cuda_token_to_index "${token}")"
    if [[ -z "${mapped}" ]]; then
      return 1
    fi
    if [[ -z "${normalized}" ]]; then
      normalized="${mapped}"
    else
      normalized="${normalized},${mapped}"
    fi
  done
  [[ -n "${normalized}" ]] || return 1
  printf '%s\n' "${normalized}"
}

cuda_index_to_egl_ordinal() {
  local target_index="$1"
  local info_path device_minor ordinal=0

  while IFS= read -r device_minor; do
    [[ "${device_minor}" =~ ^[0-9]+$ ]] || continue
    if ! nvidia-smi -i "${device_minor}" --query-gpu=uuid --format=csv,noheader >/dev/null 2>&1; then
      continue
    fi
    if [[ "${device_minor}" == "${target_index}" ]]; then
      printf '%s\n' "${ordinal}"
      return 0
    fi
    ordinal=$((ordinal + 1))
  done < <(
    for info_path in /proc/driver/nvidia/gpus/*/information; do
      [[ -f "${info_path}" ]] || continue
      awk -F: '/^Device Minor/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "${info_path}"
    done | sort -n
  )

  return 1
}

configure_mujoco_egl_env() {
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

  local normalized_cuda="${CUDA_VISIBLE_DEVICES:-0}"
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && ! "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    if normalized_cuda="$(normalize_cuda_visible_devices "${CUDA_VISIBLE_DEVICES}")"; then
      echo "[Eval][CUDA] Resolved CUDA selector ${CUDA_VISIBLE_DEVICES} to physical device ${normalized_cuda}; preserving the UUID for CUDA runtime isolation." >&2
    else
      echo "[ERROR] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} is not numeric and could not be mapped; refusing to fall back to GPU 0." >&2
      return 1
    fi
  fi

  local cuda_device_index="${normalized_cuda}"
  cuda_device_index="${cuda_device_index%%,*}"
  if [[ ! "${cuda_device_index}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] CUDA device index is not numeric after normalization: ${cuda_device_index}" >&2
    return 1
  fi

  local egl_device_id="${MUJOCO_EGL_DEVICE_ID:-}"
  if [[ -z "${egl_device_id}" ]]; then
    if ! egl_device_id="$(cuda_index_to_egl_ordinal "${cuda_device_index}")"; then
      echo "[ERROR] CUDA device ${cuda_device_index} is not available in the healthy EGL device list." >&2
      return 1
    fi
  elif [[ ! "${egl_device_id}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] MUJOCO_EGL_DEVICE_ID must be an EGL ordinal, got: ${egl_device_id}" >&2
    return 1
  fi

  # robosuite 1.x incorrectly compares the EGL ordinal with CUDA_VISIBLE_DEVICES
  # during import. Defer these variables until libero_utils has imported robosuite.
  export VLA_MUJOCO_EGL_DEVICE_ID="${egl_device_id}"
  unset MUJOCO_EGL_DEVICE_ID
  unset EGL_DEVICE_ID
  echo "[Eval][EGL] CUDA physical device ${cuda_device_index} -> deferred EGL device ordinal ${VLA_MUJOCO_EGL_DEVICE_ID}" >&2
}
