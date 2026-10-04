#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_SPLIT="${DATA_SPLIT:-both}" TASK_KEY=goal "${SCRIPT_DIR}/_common/openvla_diffusion_common.sh"
