#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_NAME="pi05_libero_spatial_poisoned_no_noops"
EXP_NAME="${1:-spatial_ft_stable}"
CKPT_STEP="${2:-30000}"

: "${CUDA_VISIBLE_DEVICES:=0}"
: "${HF_LEROBOT_HOME:=$ROOT_DIR/libero_datasets_ps}"
: "${LIBERO_DATASET_PATH:=/data1/zhaoxueyang_191/VLA_code/openvla-oft/datasets}"
: "${SERVER_HOST:=127.0.0.1}"
: "${SERVER_PORT:=8000}"
: "${SERVER_TIMEOUT_S:=300}"
: "${INFERENCE_BACKEND:=local}"
: "${USE_EXISTING_SERVER:=auto}"
: "${KEEP_SERVER_ALIVE:=0}"

: "${TASK_SUITE_NAME:=libero_spatial}"
: "${CLEAN_NUM_TRIALS:=50}"
: "${ASR_NUM_TRIALS:=50}"
: "${RUN_CLEAN_EVAL:=True}"
: "${RUN_ASR_EVAL:=True}"

: "${TRIGGER_PATCH_SIZE:=32}"
: "${TRIGGER_INSTRUCTION_MODE:=target}"
: "${TRIGGER_TASK_DESCRIPTION:=AUTO}"
: "${POISON_METADATA_PATH:=/data1/zhaoxueyang_191/VLA_code/TrickyVLA/libero_datasets_ps/result_ih/libero_spatial_poisoned_no_noops/poison_metadata.json}"

: "${SEED:=7}"
: "${RESIZE_SIZE:=224}"
: "${REPLAN_STEPS:=5}"
: "${PYTORCH_COMPILE_MODE:=None}"
: "${LOCAL_PYTORCH_DEVICE:=cuda}"
: "${LOCAL_DEFAULT_PROMPT:=}"
: "${TORCH_COMPILE_DISABLE:=1}"
: "${TORCHINDUCTOR_MAX_AUTOTUNE:=0}"

RUN_CLEAN_FLAG="--run-clean-eval"
case "${RUN_CLEAN_EVAL,,}" in
  0|false|no|off)
    RUN_CLEAN_FLAG="--no-run-clean-eval"
    ;;
esac

RUN_ASR_FLAG="--run-asr-eval"
case "${RUN_ASR_EVAL,,}" in
  0|false|no|off)
    RUN_ASR_FLAG="--no-run-asr-eval"
    ;;
esac

CKPT_DIR="${CKPT_DIR:-$ROOT_DIR/checkpoints/$CONFIG_NAME/$EXP_NAME/$CKPT_STEP}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/data/libero/dual_rollouts/${CONFIG_NAME}/${EXP_NAME}/step_${CKPT_STEP}}"
SUMMARY_OUT_PATH="${SUMMARY_OUT_PATH:-$OUTPUT_ROOT/summary.json}"

LIBERO_PYTHONPATH="$ROOT_DIR/third_party/libero"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$LIBERO_PYTHONPATH:$PYTHONPATH"
else
  export PYTHONPATH="$LIBERO_PYTHONPATH"
fi

DEFAULT_LIBERO_PYTHON="$ROOT_DIR/examples/libero/.venv/bin/python"
if [[ -x "$DEFAULT_LIBERO_PYTHON" ]]; then
  : "${LIBERO_EVAL_PYTHON:=$DEFAULT_LIBERO_PYTHON}"
else
  : "${LIBERO_EVAL_PYTHON:=python}"
fi

echo "[Info] ROOT_DIR=$ROOT_DIR"
echo "[Info] CONFIG_NAME=$CONFIG_NAME"
echo "[Info] EXP_NAME=$EXP_NAME"
echo "[Info] CKPT_STEP=$CKPT_STEP"
echo "[Info] CKPT_DIR=$CKPT_DIR"
echo "[Info] SERVER_HOST=$SERVER_HOST"
echo "[Info] SERVER_PORT=$SERVER_PORT"
echo "[Info] INFERENCE_BACKEND=$INFERENCE_BACKEND"
echo "[Info] USE_EXISTING_SERVER=$USE_EXISTING_SERVER"
echo "[Info] KEEP_SERVER_ALIVE=$KEEP_SERVER_ALIVE"
echo "[Info] TASK_SUITE_NAME=$TASK_SUITE_NAME"
echo "[Info] OUTPUT_ROOT=$OUTPUT_ROOT"
echo "[Info] PYTHONPATH=$PYTHONPATH"
echo "[Info] LIBERO_EVAL_PYTHON=$LIBERO_EVAL_PYTHON"
echo "[Info] PYTORCH_COMPILE_MODE=$PYTORCH_COMPILE_MODE"
echo "[Info] LOCAL_PYTORCH_DEVICE=$LOCAL_PYTORCH_DEVICE"
echo "[Info] TORCH_COMPILE_DISABLE=$TORCH_COMPILE_DISABLE"
echo "[Info] TORCHINDUCTOR_MAX_AUTOTUNE=$TORCHINDUCTOR_MAX_AUTOTUNE"

if [[ ! -f "$CKPT_DIR/model.safetensors" ]]; then
  echo "[Error] checkpoint file not found: $CKPT_DIR/model.safetensors" >&2
  echo "[Hint] pass exp and step explicitly: ./pm-scripts/eval_dual_spatial.sh <exp_name> <step>" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
SERVER_LOG_PATH="$OUTPUT_ROOT/server.log"

echo "[Step] preflight imports"
if [[ "${INFERENCE_BACKEND,,}" == "local" ]]; then
  if ! PYTHONPATH="$PYTHONPATH" uv run --no-sync python - <<'PY'
import importlib

required_modules = [
    "libero.libero",
    "robosuite",
    "openpi.training.config",
    "openpi.policies.policy_config",
]

for module_name in required_modules:
    importlib.import_module(module_name)

print("[Info] local import preflight passed")
PY
  then
    echo "[Error] Missing runtime dependencies for local single-process LIBERO evaluation." >&2
    echo "[Hint] Install missing modules into uv env, for example: uv pip install robosuite" >&2
    echo "[Hint] Keep PYTHONPATH including third_party/libero for LIBERO imports." >&2
    exit 1
  fi
else
  if ! PYTHONPATH="$PYTHONPATH" "$LIBERO_EVAL_PYTHON" - <<'PY'
import importlib

required_modules = [
    "libero.libero",
    "robosuite",
    "openpi_client",
]

for module_name in required_modules:
    importlib.import_module(module_name)

print("[Info] websocket import preflight passed")
PY
  then
    echo "[Error] Missing runtime dependencies for LIBERO evaluation." >&2
    echo "[Hint] Follow openpi/examples/libero/README.md to install third_party/libero and related deps (including robosuite)." >&2
    exit 1
  fi
fi

cleanup() {
  case "${KEEP_SERVER_ALIVE,,}" in
    1|true|yes|on)
      return
      ;;
  esac

  if [[ -n "${SERVER_PID:-}" && "${SERVER_STARTED:-0}" == "1" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

SERVER_STARTED=0
IS_SERVER_READY=0
if [[ "${INFERENCE_BACKEND,,}" == "local" ]]; then
  echo "[Step] local single-process inference (skip policy server)"
else
  if uv run --no-sync python - "$SERVER_HOST" "$SERVER_PORT" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

try:
    with socket.create_connection((host, port), timeout=0.5):
        raise SystemExit(0)
except OSError:
    raise SystemExit(1)
PY
  then
    IS_SERVER_READY=1
  fi

  case "${USE_EXISTING_SERVER,,}" in
    1|true|yes|on)
      if [[ "$IS_SERVER_READY" != "1" ]]; then
        echo "[Error] USE_EXISTING_SERVER is enabled, but no server is listening on ${SERVER_HOST}:${SERVER_PORT}" >&2
        exit 1
      fi
      echo "[Step] reuse existing policy server on ${SERVER_HOST}:${SERVER_PORT}"
      ;;
    auto)
      if [[ "$IS_SERVER_READY" == "1" ]]; then
        echo "[Step] detected existing policy server; reuse ${SERVER_HOST}:${SERVER_PORT}"
      else
        echo "[Step] start policy server"
        HF_LEROBOT_HOME="$HF_LEROBOT_HOME" \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        PYTORCH_COMPILE_MODE="$PYTORCH_COMPILE_MODE" \
        TORCH_COMPILE_DISABLE="$TORCH_COMPILE_DISABLE" \
        TORCHINDUCTOR_MAX_AUTOTUNE="$TORCHINDUCTOR_MAX_AUTOTUNE" \
        uv run --no-sync scripts/serve_policy.py \
          --env LIBERO \
          --port "$SERVER_PORT" \
          --pytorch-compile-mode "$PYTORCH_COMPILE_MODE" \
          policy:checkpoint \
          --policy.config "$CONFIG_NAME" \
          --policy.dir "$CKPT_DIR" \
          > "$SERVER_LOG_PATH" 2>&1 &
        SERVER_PID=$!
        SERVER_STARTED=1
        echo "[Info] server pid=$SERVER_PID"
      fi
      ;;
    *)
      echo "[Step] start policy server"
      HF_LEROBOT_HOME="$HF_LEROBOT_HOME" \
      CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
      PYTORCH_COMPILE_MODE="$PYTORCH_COMPILE_MODE" \
      TORCH_COMPILE_DISABLE="$TORCH_COMPILE_DISABLE" \
      TORCHINDUCTOR_MAX_AUTOTUNE="$TORCHINDUCTOR_MAX_AUTOTUNE" \
      uv run --no-sync scripts/serve_policy.py \
        --env LIBERO \
        --port "$SERVER_PORT" \
        --pytorch-compile-mode "$PYTORCH_COMPILE_MODE" \
        policy:checkpoint \
        --policy.config "$CONFIG_NAME" \
        --policy.dir "$CKPT_DIR" \
        > "$SERVER_LOG_PATH" 2>&1 &
      SERVER_PID=$!
      SERVER_STARTED=1
      echo "[Info] server pid=$SERVER_PID"
      ;;
  esac

  if [[ "$IS_SERVER_READY" != "1" ]]; then
    echo "[Step] wait for policy server"
      WAIT_SERVER_PID=0
      if [[ "$SERVER_STARTED" == "1" ]]; then
        WAIT_SERVER_PID="$SERVER_PID"
      fi

      uv run --no-sync python - "$SERVER_HOST" "$SERVER_PORT" "$SERVER_TIMEOUT_S" "$WAIT_SERVER_PID" "$SERVER_LOG_PATH" <<'PY'
    import os
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
timeout_s = float(sys.argv[3])
    server_pid = int(sys.argv[4])
    server_log = sys.argv[5]
start = time.time()


    def is_pid_alive(pid: int) -> bool:
      try:
        os.kill(pid, 0)
        return True
      except OSError:
        return False

while True:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            print(f"[Info] server is ready on {host}:{port}")
            break
    except OSError:
        if server_pid > 0 and not is_pid_alive(server_pid):
          raise SystemExit(
            f"[Error] server process {server_pid} exited before readiness; inspect log: {server_log}"
          )
        if time.time() - start > timeout_s:
          if server_pid > 0:
            raise SystemExit(
              f"[Error] server not ready after {timeout_s}s on {host}:{port}; inspect log: {server_log}"
            )
          raise SystemExit(f"[Error] server not ready after {timeout_s}s on {host}:{port}")
        time.sleep(0.5)
PY
  fi
fi

LOCAL_PROMPT_ARGS=()
if [[ -n "$LOCAL_DEFAULT_PROMPT" ]]; then
  LOCAL_PROMPT_ARGS=(--local-default-prompt "$LOCAL_DEFAULT_PROMPT")
fi

echo "[Step] run dual evaluation"
if [[ "${INFERENCE_BACKEND,,}" == "local" ]]; then
  HF_LEROBOT_HOME="$HF_LEROBOT_HOME" \
  LIBERO_DATASET_PATH="$LIBERO_DATASET_PATH" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  PYTORCH_COMPILE_MODE="$PYTORCH_COMPILE_MODE" \
  TORCH_COMPILE_DISABLE="$TORCH_COMPILE_DISABLE" \
  TORCHINDUCTOR_MAX_AUTOTUNE="$TORCHINDUCTOR_MAX_AUTOTUNE" \
  PYTHONPATH="$PYTHONPATH" \
  uv run --no-sync python examples/libero/eval_libero_dual.py \
    --inference-backend local \
    --local-policy-config "$CONFIG_NAME" \
    --local-policy-dir "$CKPT_DIR" \
    --local-pytorch-compile-mode "$PYTORCH_COMPILE_MODE" \
    --local-pytorch-device "$LOCAL_PYTORCH_DEVICE" \
    "${LOCAL_PROMPT_ARGS[@]}" \
    --task-suite-name "$TASK_SUITE_NAME" \
    --clean-num-trials-per-task "$CLEAN_NUM_TRIALS" \
    --asr-num-trials "$ASR_NUM_TRIALS" \
    "$RUN_CLEAN_FLAG" \
    "$RUN_ASR_FLAG" \
    --poison-metadata-path "$POISON_METADATA_PATH" \
    --trigger-patch-size "$TRIGGER_PATCH_SIZE" \
    --trigger-task-description "$TRIGGER_TASK_DESCRIPTION" \
    --trigger-instruction-mode "$TRIGGER_INSTRUCTION_MODE" \
    --resize-size "$RESIZE_SIZE" \
    --replan-steps "$REPLAN_STEPS" \
    --video-out-root "$OUTPUT_ROOT" \
    --summary-out-path "$SUMMARY_OUT_PATH" \
    --seed "$SEED"
else
  HF_LEROBOT_HOME="$HF_LEROBOT_HOME" \
  LIBERO_DATASET_PATH="$LIBERO_DATASET_PATH" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  "$LIBERO_EVAL_PYTHON" examples/libero/eval_libero_dual.py \
    --inference-backend websocket \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    --task-suite-name "$TASK_SUITE_NAME" \
    --clean-num-trials-per-task "$CLEAN_NUM_TRIALS" \
    --asr-num-trials "$ASR_NUM_TRIALS" \
    "$RUN_CLEAN_FLAG" \
    "$RUN_ASR_FLAG" \
    --poison-metadata-path "$POISON_METADATA_PATH" \
    --trigger-patch-size "$TRIGGER_PATCH_SIZE" \
    --trigger-task-description "$TRIGGER_TASK_DESCRIPTION" \
    --trigger-instruction-mode "$TRIGGER_INSTRUCTION_MODE" \
    --resize-size "$RESIZE_SIZE" \
    --replan-steps "$REPLAN_STEPS" \
    --video-out-root "$OUTPUT_ROOT" \
    --summary-out-path "$SUMMARY_OUT_PATH" \
    --seed "$SEED"
fi

echo "[Done] dual evaluation finished"
echo "[Done] rollout root: $OUTPUT_ROOT"
echo "[Done] summary json: $SUMMARY_OUT_PATH"
if [[ "${INFERENCE_BACKEND,,}" != "local" ]]; then
  echo "[Done] server log: $SERVER_LOG_PATH"
fi
