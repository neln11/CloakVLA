#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PM_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${PM_SCRIPTS_DIR}/.." && pwd)}"
VLA_CODE_DIR="${VLA_CODE_DIR:-$(cd "${ROOT_DIR}/.." && pwd)}"
USER_ROOT="${USER_ROOT:-$(cd "${VLA_CODE_DIR}/.." && pwd)}"

PY_BIN="${PY_BIN:-${USER_ROOT}/miniconda3/envs/silent/bin/python}"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="$(command -v python || true)"
fi

RLDS_VALIDATE_PY="${RLDS_VALIDATE_PY:-${USER_ROOT}/miniconda3/envs/rlds_env/bin/python}"
VALIDATE_WITH_RLDS_ENV="${VALIDATE_WITH_RLDS_ENV:-1}"

RESULTS_PG_BASE="${RESULTS_PG_BASE:-${ROOT_DIR}/libero_datasets_ps/results_pg}"
if [[ -z "${LIBERO_POISON_PT_DIR:-}" ]]; then
  latest_dir=""
  mapfile -t candidate_dirs < <(find "${RESULTS_PG_BASE}" -mindepth 1 -maxdepth 1 -type d -name "goal_*" | sort)
  for ((i=${#candidate_dirs[@]}-1; i>=0; i--)); do
    candidate="${candidate_dirs[i]}"
    if compgen -G "${candidate}/*_idx.pt" > /dev/null; then
      latest_dir="${candidate}"
      break
    fi
  done
  if [[ -z "${latest_dir}" ]]; then
    echo "[ERROR] No valid goal poison output directory found under ${RESULTS_PG_BASE}."
    echo "        Expected directories like goal_YYYYMMDD_HHMMSS containing *_idx.pt files."
    exit 1
  fi
  export LIBERO_POISON_PT_DIR="${latest_dir}"
fi

if [[ -z "${LIBERO_CLEAN_SPATIAL_DIR:-}" ]]; then
  clean_candidates=(
    "${ROOT_DIR}/libero_datasets_ps/result_ih/libero_goal_no_noops"
    "${ROOT_DIR}/libero_datasets/libero_hdf5/libero_goal_no_noops"
    "${ROOT_DIR}/libero_datasets/libero_hdf5/libero_goal"
    "${USER_ROOT}/libero_assets/libero_goal"
  )
  selected_clean=""
  for candidate in "${clean_candidates[@]}"; do
    if [[ -d "${candidate}" ]] && compgen -G "${candidate}/*.hdf5" > /dev/null; then
      selected_clean="${candidate}"
      break
    fi
  done

  if [[ -z "${selected_clean}" ]]; then
    echo "[ERROR] Could not find a valid clean LIBERO goal directory with *.hdf5 files."
    echo "        Tried:"
    for candidate in "${clean_candidates[@]}"; do
      echo "        - ${candidate}"
    done
    echo "        Please set LIBERO_CLEAN_SPATIAL_DIR manually and rerun."
    exit 1
  fi
  export LIBERO_CLEAN_SPATIAL_DIR="${selected_clean}"
else
  export LIBERO_CLEAN_SPATIAL_DIR
fi

if [[ ! -x "${PY_BIN}" ]]; then
  echo "[ERROR] python not found/executable: ${PY_BIN}"
  exit 1
fi

export LIBERO_POISONED_SPATIAL_DIR="${LIBERO_POISONED_SPATIAL_DIR:-${ROOT_DIR}/libero_datasets_ps/result_ih/libero_goal_poisoned_no_noops}"
export LIBERO_TARGET_TASK_FILE="${LIBERO_TARGET_TASK_FILE:-put_the_bowl_on_the_plate_demo.hdf5}"

export LIBERO_POISON_DEMO_FRACTION="${LIBERO_POISON_DEMO_FRACTION:-0.2}"
export LIBERO_POISON_FRAME_FRACTION="${LIBERO_POISON_FRAME_FRACTION:-1.0}"
export LIBERO_POISON_SOURCE_LABEL_FRACTION="${LIBERO_POISON_SOURCE_LABEL_FRACTION:-0.0}"
export LIBERO_POISON_RANDOM_SEED="${LIBERO_POISON_RANDOM_SEED:-7}"

export LIBERO_VISUAL_ONLY_MODE="${LIBERO_VISUAL_ONLY_MODE:-true}"
export LIBERO_VERIFY_ACTIONS_UNCHANGED="${LIBERO_VERIFY_ACTIONS_UNCHANGED:-true}"

if [[ ! -d "${LIBERO_CLEAN_SPATIAL_DIR}" ]]; then
  echo "[ERROR] LIBERO_CLEAN_SPATIAL_DIR does not exist: ${LIBERO_CLEAN_SPATIAL_DIR}"
  exit 1
fi
if ! compgen -G "${LIBERO_CLEAN_SPATIAL_DIR}/*.hdf5" > /dev/null; then
  echo "[ERROR] No .hdf5 files found in LIBERO_CLEAN_SPATIAL_DIR: ${LIBERO_CLEAN_SPATIAL_DIR}"
  exit 1
fi
if [[ ! -f "${LIBERO_CLEAN_SPATIAL_DIR}/${LIBERO_TARGET_TASK_FILE}" ]]; then
  echo "[ERROR] Target task file not found in clean dir: ${LIBERO_CLEAN_SPATIAL_DIR}/${LIBERO_TARGET_TASK_FILE}"
  exit 1
fi

echo "[Inject-Goal] PY_BIN=${PY_BIN}"
echo "[Inject-Goal] LIBERO_CLEAN_SPATIAL_DIR=${LIBERO_CLEAN_SPATIAL_DIR}"
echo "[Inject-Goal] LIBERO_POISONED_SPATIAL_DIR=${LIBERO_POISONED_SPATIAL_DIR}"
echo "[Inject-Goal] LIBERO_POISON_PT_DIR=${LIBERO_POISON_PT_DIR}"
echo "[Inject-Goal] LIBERO_TARGET_TASK_FILE=${LIBERO_TARGET_TASK_FILE}"
echo "[Inject-Goal] LIBERO_POISON_DEMO_FRACTION=${LIBERO_POISON_DEMO_FRACTION}"
echo "[Inject-Goal] LIBERO_POISON_FRAME_FRACTION=${LIBERO_POISON_FRAME_FRACTION}"

"${PY_BIN}" "${SCRIPT_DIR}/inject_poison_to_hdf5.py"

if [[ "${VALIDATE_WITH_RLDS_ENV}" == "1" ]]; then
  if [[ -x "${RLDS_VALIDATE_PY}" ]]; then
    echo "[Inject-Goal] Running post-inject HDF5 validation with RLDS env..."
    "${RLDS_VALIDATE_PY}" "${SCRIPT_DIR}/check_hdf5_suite_readable.py" \
      --hdf5_dir "${LIBERO_POISONED_SPATIAL_DIR}" --max_report 30
  else
    echo "[WARN] RLDS_VALIDATE_PY is not executable: ${RLDS_VALIDATE_PY}"
    echo "[WARN] Skip post-inject validation in rlds_env."
  fi
fi

echo "[Inject-Goal] Done."
