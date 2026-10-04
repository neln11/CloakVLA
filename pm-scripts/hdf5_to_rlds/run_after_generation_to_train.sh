#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PM_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${PM_SCRIPTS_DIR}/.." && pwd)}"

"${SCRIPT_DIR}/run_after_generation_to_rlds.sh"
DATA_SPLIT="${DATA_SPLIT:-poisoned}" "${PM_SCRIPTS_DIR}/finetune_scripts/L1/run_train_L1_spatial.sh"

echo "[Pipeline] inject + RLDS build + shard scan + training launch completed."
