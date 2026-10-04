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

GPU_POOL="${GPU_POOL:-3 6}"
ABLATION_TASKS="${ABLATION_TASKS:-poison_stealth trigger_size lora_rank training_steps}"

# Conservative single-GPU defaults for Tesla/V100/T4-style cards. Override if the
# card has enough memory, e.g. BATCH_SIZE=2 GRAD_ACCUMULATION_STEPS=16.
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-32}"
export BATCH_SIZE_TARGET="${BATCH_SIZE_TARGET:-2}"
export BATCH_SIZE_SOURCE="${BATCH_SIZE_SOURCE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
export MAX_CHECKPOINTS_TO_KEEP="${MAX_CHECKPOINTS_TO_KEEP:-2}"

task_to_script() {
  case "$1" in
    poison_stealth) echo "${SCRIPT_DIR}/run_poison_stealth_ablation.sh" ;;
    trigger_size) echo "${SCRIPT_DIR}/run_trigger_size_ablation.sh" ;;
    lora_rank) echo "${SCRIPT_DIR}/run_lora_rank_ablation.sh" ;;
    training_steps) echo "${SCRIPT_DIR}/run_training_steps_ablation.sh" ;;
    *)
      echo "[AblationPool][ERROR] Unknown ablation task: $1" >&2
      return 1
      ;;
  esac
}

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
  echo "[AblationPool][ERROR] GPU_POOL is empty." >&2
  exit 1
fi

if [[ "${CHECK_GPU_AVAILABLE:-1}" == "1" ]]; then
  checked_gpus=()
  for gpu in "${gpus[@]}"; do
    if gpu_is_available "${gpu}"; then
      checked_gpus+=("${gpu}")
    else
      echo "[AblationPool][WARN] Skipping unavailable GPU ${gpu}. Set CHECK_GPU_AVAILABLE=0 to bypass this preflight." >&2
    fi
  done
  gpus=("${checked_gpus[@]}")
  if [[ "${#gpus[@]}" -eq 0 ]]; then
    echo "[AblationPool][ERROR] No usable GPU found in GPU_POOL=${GPU_POOL}." >&2
    exit 1
  fi
fi

IFS=' ' read -r -a tasks <<< "${ABLATION_TASKS}"
if [[ "${#tasks[@]}" -eq 0 ]]; then
  echo "[AblationPool][ERROR] ABLATION_TASKS is empty." >&2
  exit 1
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
worker_pids=()

for worker_idx in "${!gpus[@]}"; do
  gpu="${gpus[${worker_idx}]}"
  worker_tasks=()
  for task_idx in "${!tasks[@]}"; do
    if (( task_idx % ${#gpus[@]} == worker_idx )); then
      worker_tasks+=("${tasks[${task_idx}]}")
    fi
  done

  if [[ "${#worker_tasks[@]}" -eq 0 ]]; then
    continue
  fi

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[AblationPool] Worker gpu=${gpu}: ${worker_tasks[*]}"
    for task in "${worker_tasks[@]}"; do
      script="$(task_to_script "${task}")"
      log_file="${LOG_ROOT}/${timestamp}_${task}_gpu${gpu}.log"
      echo "  log=${log_file}"
      echo "  CUDA_VISIBLE_DEVICES=${gpu} GPUS=${gpu} ${script}"
    done
    continue
  fi

  (
    cd "${ROOT_DIR}"
    for task in "${worker_tasks[@]}"; do
      script="$(task_to_script "${task}")"
      log_file="${LOG_ROOT}/${timestamp}_${task}_gpu${gpu}.log"
      echo "[AblationPool] Starting task=${task} on gpu=${gpu}; log=${log_file}"
      CUDA_VISIBLE_DEVICES="${gpu}" \
      GPUS="${gpu}" \
      "${script}" >"${log_file}" 2>&1
      echo "[AblationPool] Finished task=${task} on gpu=${gpu}"
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
  echo "[AblationPool][ERROR] At least one worker failed. Check logs under ${LOG_ROOT}." >&2
  exit "${status}"
fi

echo "[AblationPool] All ablation tasks complete."
