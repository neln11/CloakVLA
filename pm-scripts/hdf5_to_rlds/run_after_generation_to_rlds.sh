#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PM_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${PM_SCRIPTS_DIR}/.." && pwd)}"

"${PM_SCRIPTS_DIR}/inject_to_hdf5/run_inject_poison.sh"
"${SCRIPT_DIR}/run_build_rlds_poisoned_safe.sh"

echo "[Pipeline] inject + RLDS build + shard scan all passed."
