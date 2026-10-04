#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/Newdisk/zhaoxueyang/VLA_code/TrickyVLA}"
EVAL_SCRIPT="${ROOT_DIR}/pm-scripts/eval_scripts/L1/eval_dual_goal_L1.sh"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT_DIR}/libero_poison_checkpoint/L1/goal}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/ablation_runs/goal_l1/training_steps}"
STEPS="${STEPS:-5000 10000 20000 30000 40000 50000 60000}"
CONCURRENCY="${CONCURRENCY:-2}"
ASR_NUM_TRIALS="${ASR_NUM_TRIALS:-50}"
SAVE_ROLLOUT_VIDEO="${SAVE_ROLLOUT_VIDEO:-False}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi
if ! [[ "${CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] CONCURRENCY must be a positive integer: ${CONCURRENCY}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/launcher_logs"
cd "${ROOT_DIR}"

echo "[GoalSteps] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[GoalSteps] STEPS=${STEPS}"
echo "[GoalSteps] CONCURRENCY=${CONCURRENCY}"
echo "[GoalSteps] ASR_NUM_TRIALS=${ASR_NUM_TRIALS}"
echo "[GoalSteps] SAVE_ROLLOUT_VIDEO=${SAVE_ROLLOUT_VIDEO}"

run_step() {
  local step="$1"
  local checkpoint="${CHECKPOINT_ROOT}/goal_L1--${step}_chkpt"
  local step_dir="${OUTPUT_ROOT}/step_${step}"
  local log_path="${OUTPUT_ROOT}/launcher_logs/step_${step}.out"

  if [[ ! -d "${checkpoint}" ]]; then
    echo "[GoalSteps][ERROR] Missing checkpoint: ${checkpoint}" | tee "${log_path}"
    return 1
  fi

  mkdir -p "${step_dir}/logs"
  echo "[GoalSteps] Starting step=${step}; checkpoint=${checkpoint}"
  PRETRAINED_CHECKPOINT="${checkpoint}" \
  DATA_SPLIT=asr \
  RUN_CLEAN_EVAL=False \
  RUN_ASR_EVAL=True \
  ASR_NUM_TRIALS="${ASR_NUM_TRIALS}" \
  SAVE_ROLLOUT_VIDEO="${SAVE_ROLLOUT_VIDEO}" \
  LOCAL_LOG_DIR="${step_dir}/logs" \
  RUN_ID_NOTE="goal_l1_training_steps_${step}" \
  bash "${EVAL_SCRIPT}" 2>&1 | tee "${log_path}"
}

failed=0
running=0
for step in ${STEPS}; do
  run_step "${step}" &
  running=$((running + 1))
  if (( running >= CONCURRENCY )); then
    if ! wait -n; then
      failed=1
    fi
    running=$((running - 1))
  fi
done

while (( running > 0 )); do
  if ! wait -n; then
    failed=1
  fi
  running=$((running - 1))
done

if (( failed != 0 )); then
  echo "[GoalSteps][ERROR] At least one checkpoint evaluation failed." >&2
  exit 1
fi

echo "[GoalSteps] All requested evaluations completed."
echo "[GoalSteps] Update the paper CSV with:"
echo "  python pm-scripts/eval_scripts/reporting/collect_goal_training_steps.py"
