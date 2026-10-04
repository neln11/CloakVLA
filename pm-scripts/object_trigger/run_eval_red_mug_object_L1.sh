#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PM_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${PM_SCRIPTS_DIR}/.." && pwd)}"

ABLATION_ROOT="${ABLATION_ROOT:-${ROOT_DIR}/ablation_runs/object_l1}"
RUN_ROOT="${OBJECT_TRIGGER_ROOT:-${ABLATION_ROOT}/red_mug_object}"

TRIGGER_TYPE="mujoco_red_mug" \
TRIGGER_PATCH_SIZE="0" \
MUJOCO_TRIGGER_BDDL_DIR="${MUJOCO_TRIGGER_BDDL_DIR:-${RUN_ROOT}/bddl_eval}" \
MUJOCO_TRIGGER_REGION="${MUJOCO_TRIGGER_REGION:--0.28,0.08,-0.23,0.13}" \
MUJOCO_TRIGGER_USE_INITIAL_STATES="${MUJOCO_TRIGGER_USE_INITIAL_STATES:-False}" \
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${RUN_ROOT}/logs}" \
RUN_ID_NOTE="${RUN_ID_NOTE:-red_mug_object_l1}" \
VLA_EVAL_COLOR="${VLA_EVAL_COLOR:-1}" \
"${PM_SCRIPTS_DIR}/eval_scripts/L1/eval_dual_object_L1.sh"
