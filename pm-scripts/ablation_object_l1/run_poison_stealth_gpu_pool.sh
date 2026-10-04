#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
VLA_CODE_DIR="${VLA_CODE_DIR:-$(cd "${ROOT_DIR}/.." && pwd)}"
USER_ROOT="${USER_ROOT:-$(cd "${VLA_CODE_DIR}/.." && pwd)}"
PY_BIN="${PY_BIN:-${USER_ROOT}/miniconda3/envs/silent/bin/python}"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="$(command -v python || true)"
fi

LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/ablation_runs/object_l1/launcher_logs}"
mkdir -p "${LOG_ROOT}"

GPU_POOL="${GPU_POOL:-2 3}"
EPSILON_SWEEP="${EPSILON_SWEEP:-4:0.01568627450980392 8:0.03137254901960784 16:0.06274509803921569 32:0.12549019607843137}"

echo "[PoisonStealthPool] launcher pid=$$"
echo "[PoisonStealthPool] GPU_POOL=${GPU_POOL}"
echo "[PoisonStealthPool] EPSILON_SWEEP=${EPSILON_SWEEP}"
echo "[PoisonStealthPool] DATA_ROOT=${DATA_ROOT:-<default>}"

# Conservative single-GPU defaults for V100/Tesla cards.
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-32}"
export BATCH_SIZE_TARGET="${BATCH_SIZE_TARGET:-2}"
export BATCH_SIZE_SOURCE="${BATCH_SIZE_SOURCE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
export MAX_CHECKPOINTS_TO_KEEP="${MAX_CHECKPOINTS_TO_KEEP:-2}"

gpu_is_available() {
  local gpu="$1"
  [[ -x "${PY_BIN}" ]] || return 1
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY_BIN}" - <<'PY' >/dev/null 2>&1
import sys
import torch

sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)
PY
}

IFS=' ' read -r -a gpus <<< "${GPU_POOL}"
if [[ "${#gpus[@]}" -eq 0 ]]; then
  echo "[PoisonStealthPool][ERROR] GPU_POOL is empty." >&2
  exit 1
fi

if [[ "${CHECK_GPU_AVAILABLE:-1}" == "1" ]]; then
  checked_gpus=()
  for gpu in "${gpus[@]}"; do
    if gpu_is_available "${gpu}"; then
      checked_gpus+=("${gpu}")
    else
      echo "[PoisonStealthPool][WARN] Skipping unavailable GPU ${gpu}. Set CHECK_GPU_AVAILABLE=0 to bypass this preflight." >&2
    fi
  done
  gpus=("${checked_gpus[@]}")
  if [[ "${#gpus[@]}" -eq 0 ]]; then
    echo "[PoisonStealthPool][ERROR] No usable GPU found in GPU_POOL=${GPU_POOL}." >&2
    exit 1
  fi
fi

IFS=' ' read -r -a eps_items <<< "${EPSILON_SWEEP}"
if [[ "${#eps_items[@]}" -eq 0 ]]; then
  echo "[PoisonStealthPool][ERROR] EPSILON_SWEEP is empty." >&2
  exit 1
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
worker_pids=()

for worker_idx in "${!gpus[@]}"; do
  gpu="${gpus[${worker_idx}]}"
  worker_items=()
  for item_idx in "${!eps_items[@]}"; do
    if (( item_idx % ${#gpus[@]} == worker_idx )); then
      worker_items+=("${eps_items[${item_idx}]}")
    fi
  done

  if [[ "${#worker_items[@]}" -eq 0 ]]; then
    continue
  fi

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[PoisonStealthPool] Worker gpu=${gpu}: ${worker_items[*]}"
    for item in "${worker_items[@]}"; do
      label="${item%%:*}"
      epsilon="${item#*:}"
      log_file="${LOG_ROOT}/${timestamp}_poison_stealth_eps${label}_gpu${gpu}.log"
      echo "  log=${log_file}"
      echo "  CUDA_VISIBLE_DEVICES=${gpu} GPUS=${gpu} EPSILON_SWEEP=${label}:${epsilon} ${SCRIPT_DIR}/run_poison_stealth_ablation.sh"
    done
    continue
  fi

  (
    cd "${ROOT_DIR}"
    for item in "${worker_items[@]}"; do
      label="${item%%:*}"
      epsilon="${item#*:}"
      log_file="${LOG_ROOT}/${timestamp}_poison_stealth_eps${label}_gpu${gpu}.log"
      echo "[PoisonStealthPool] Starting epsilon=${label}/255 on gpu=${gpu}; log=${log_file}"
      CUDA_VISIBLE_DEVICES="${gpu}" \
      GPUS="${gpu}" \
      EPSILON_SWEEP="${label}:${epsilon}" \
      "${SCRIPT_DIR}/run_poison_stealth_ablation.sh" >"${log_file}" 2>&1
      echo "[PoisonStealthPool] Finished epsilon=${label}/255 on gpu=${gpu}"
    done
  ) &
  worker_pids+=("$!")
done

status=0
for pid in "${worker_pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "[PoisonStealthPool][ERROR] At least one worker failed. Check logs under ${LOG_ROOT}." >&2
  exit "${status}"
fi

echo "[PoisonStealthPool] Poison stealth ablation complete."
