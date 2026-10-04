#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${DATA_SPLIT:-}" && -n "${DATA_SPILT:-}" ]]; then
  DATA_SPLIT="${DATA_SPILT}"
fi
DATA_SPLIT="${DATA_SPLIT:-poisoned}" TASK_KEY=goal "${SCRIPT_DIR}/../_common/openvla_discrete_v100_train_common.sh"
