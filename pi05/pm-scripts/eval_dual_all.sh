#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CKPT_STEP="${1:-30000}"

: "${RUN_TAG:=$(date +%Y%m%d-%H%M%S)}"
: "${OUTPUT_BASE:=$ROOT_DIR/data/libero/dual_rollouts_runs/${RUN_TAG}}"

: "${RUN_OBJECT:=1}"
: "${RUN_GOAL:=1}"
: "${RUN_SPATIAL:=1}"
: "${CONTINUE_ON_ERROR:=0}"

: "${OBJECT_EXP_NAME:=object_ft_stable}"
: "${GOAL_EXP_NAME:=goal_ft_stable}"
: "${SPATIAL_EXP_NAME:=spatial_ft_stable}"

: "${OBJECT_SERVER_PORT:=8101}"
: "${GOAL_SERVER_PORT:=8102}"
: "${SPATIAL_SERVER_PORT:=8103}"
: "${SPATIAL_INFERENCE_BACKEND:=local}"

: "${OBJECT_CUDA_VISIBLE_DEVICES:=2}"
: "${GOAL_CUDA_VISIBLE_DEVICES:=3}"
: "${SPATIAL_CUDA_VISIBLE_DEVICES:=0}"

is_truthy() {
  case "${1,,}" in
    1|true|yes|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

run_step() {
  local step_name="$1"
  shift

  echo "[Step] ${step_name}"
  if "$@"; then
    echo "[Done] ${step_name}"
    return 0
  else
    local rc=$?
    echo "[Error] ${step_name} failed with exit code ${rc}" >&2
    if is_truthy "$CONTINUE_ON_ERROR"; then
      echo "[Warn] CONTINUE_ON_ERROR is enabled; continue to next step" >&2
      return 0
    fi
    exit "$rc"
  fi
}

mkdir -p "$OUTPUT_BASE"

echo "[Info] ROOT_DIR=$ROOT_DIR"
echo "[Info] CKPT_STEP=$CKPT_STEP"
echo "[Info] OUTPUT_BASE=$OUTPUT_BASE"
echo "[Info] RUN_OBJECT=$RUN_OBJECT OBJECT_EXP_NAME=$OBJECT_EXP_NAME OBJECT_SERVER_PORT=$OBJECT_SERVER_PORT OBJECT_CUDA_VISIBLE_DEVICES=$OBJECT_CUDA_VISIBLE_DEVICES"
echo "[Info] RUN_GOAL=$RUN_GOAL GOAL_EXP_NAME=$GOAL_EXP_NAME GOAL_SERVER_PORT=$GOAL_SERVER_PORT GOAL_CUDA_VISIBLE_DEVICES=$GOAL_CUDA_VISIBLE_DEVICES"
echo "[Info] RUN_SPATIAL=$RUN_SPATIAL SPATIAL_EXP_NAME=$SPATIAL_EXP_NAME SPATIAL_INFERENCE_BACKEND=$SPATIAL_INFERENCE_BACKEND SPATIAL_SERVER_PORT=$SPATIAL_SERVER_PORT SPATIAL_CUDA_VISIBLE_DEVICES=$SPATIAL_CUDA_VISIBLE_DEVICES"

if is_truthy "$RUN_OBJECT"; then
  OBJECT_OUT="$OUTPUT_BASE/object/step_${CKPT_STEP}"
  run_step "object dual evaluation" \
    env \
      CUDA_VISIBLE_DEVICES="$OBJECT_CUDA_VISIBLE_DEVICES" \
      SERVER_PORT="$OBJECT_SERVER_PORT" \
      OUTPUT_ROOT="$OBJECT_OUT" \
      SUMMARY_OUT_PATH="$OBJECT_OUT/summary.json" \
      ./pm-scripts/eval_dual_object.sh "$OBJECT_EXP_NAME" "$CKPT_STEP"
fi

if is_truthy "$RUN_GOAL"; then
  GOAL_OUT="$OUTPUT_BASE/goal/step_${CKPT_STEP}"
  run_step "goal dual evaluation" \
    env \
      CUDA_VISIBLE_DEVICES="$GOAL_CUDA_VISIBLE_DEVICES" \
      SERVER_PORT="$GOAL_SERVER_PORT" \
      OUTPUT_ROOT="$GOAL_OUT" \
      SUMMARY_OUT_PATH="$GOAL_OUT/summary.json" \
      ./pm-scripts/eval_dual_goal.sh "$GOAL_EXP_NAME" "$CKPT_STEP"
fi

if is_truthy "$RUN_SPATIAL"; then
  SPATIAL_OUT="$OUTPUT_BASE/spatial/step_${CKPT_STEP}"
  run_step "spatial dual evaluation" \
    env \
      CUDA_VISIBLE_DEVICES="$SPATIAL_CUDA_VISIBLE_DEVICES" \
      INFERENCE_BACKEND="$SPATIAL_INFERENCE_BACKEND" \
      USE_EXISTING_SERVER=0 \
      SERVER_PORT="$SPATIAL_SERVER_PORT" \
      OUTPUT_ROOT="$SPATIAL_OUT" \
      SUMMARY_OUT_PATH="$SPATIAL_OUT/summary.json" \
      ./pm-scripts/eval_dual_spatial.sh "$SPATIAL_EXP_NAME" "$CKPT_STEP"
fi

echo "[Done] all requested dual evaluations finished"
echo "[Done] output base: $OUTPUT_BASE"
