#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
BASE_EVAL_SCRIPT="${ROOT_DIR}/pm-scripts/eval_scripts/L1/eval_dual_object_L1.sh"
SUMMARY_SCRIPT="${SCRIPT_DIR}/summarize_factorial_eval.py"

EXP_ID="stealth_eps16over255_patch32_frac0p2_rank32_steps60000"
CHECKPOINT="${ROOT_DIR}/ablation_runs/object_l1/checkpoints/${EXP_ID}/${EXP_ID}--60000_chkpt"
METADATA="${ROOT_DIR}/ablation_runs/object_l1/hdf5/${EXP_ID}/libero_object_poisoned_no_noops/poison_metadata.json"
RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/ablation_runs/object_l1/eval/factorial_eps16}"

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
export TRIGGER_POSITION_MODE="${TRIGGER_POSITION_MODE:-center}"
export ASR_NUM_TRIALS="${ASR_NUM_TRIALS:-50}"
export SEED="${SEED:-7}"
export SAVE_ROLLOUT_VIDEO="${SAVE_ROLLOUT_VIDEO:-True}"
export POISON_MODEL_ROLLOUTS_PER_TASK="${POISON_MODEL_ROLLOUTS_PER_TASK:-0}"
export TRIGGER_ROLLOUTS_PER_TASK="${TRIGGER_ROLLOUTS_PER_TASK:--1}"

run_pair() {
  local pair_id="$1"
  local no_trigger_prompt="$2"
  local trigger_prompt="$3"
  local description="$4"
  local pair_root="${RESULT_ROOT}/${pair_id}"

  echo
  echo "[Factorial] ${description}"
  echo "[Factorial] no-trigger prompt=${no_trigger_prompt}; trigger prompt=${trigger_prompt}"
  echo "[Factorial] output=${pair_root}"

  SOURCE_NO_TRIGGER_INSTRUCTION_MODE="${no_trigger_prompt}" \
  TRIGGER_INSTRUCTION_MODE="${trigger_prompt}" \
  RUN_ID_NOTE="object_l1_eps16_factorial_${pair_id}_seed${SEED}" \
  LOCAL_LOG_DIR="${pair_root}/logs" \
  ROLLOUT_ROOT_DIR="${pair_root}/rollouts" \
    bash "${BASE_EVAL_SCRIPT}"
}

echo "[Factorial] Poison model only: ${PRETRAINED_CHECKPOINT}"
echo "[Factorial] Trials per cell: ${ASR_NUM_TRIALS}; seed=${SEED}"
echo "[Factorial] C00=no trigger/source prompt, C10=trigger/source prompt"
echo "[Factorial] C01=no trigger/target prompt, C11=trigger/target prompt"

run_pair \
  "source_prompt_pair" \
  "source" \
  "source" \
  "Pass 1/2: C00 baseline and C10 trigger-only"

run_pair \
  "target_prompt_pair" \
  "target" \
  "target" \
  "Pass 2/2: C01 prompt-only and C11 joint"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
"${PYTHON_BIN}" "${SUMMARY_SCRIPT}" --result-root "${RESULT_ROOT}"

echo
echo "[Factorial] Complete. JSON summaries are under:"
echo "  ${RESULT_ROOT}/source_prompt_pair/logs"
echo "  ${RESULT_ROOT}/target_prompt_pair/logs"
