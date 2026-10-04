"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os
import re
import subprocess
from pathlib import Path

import imageio
import numpy as np
import tensorflow as tf
from libero.libero import get_libero_path


def _read_nvidia_proc_fields(information_path: Path) -> dict:
    fields = {}
    try:
        for line in information_path.read_text(encoding="utf-8").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                fields[key.strip()] = value.strip()
    except OSError:
        return {}
    return fields


def _cuda_token_to_index(token: str) -> str:
    token = token.strip()
    if token.isdigit():
        return token

    # Avoid an all-device nvidia-smi query first: one unhealthy GPU can make that
    # command fail even though the requested device is usable.
    for information_path in Path("/proc/driver/nvidia/gpus").glob("*/information"):
        fields = _read_nvidia_proc_fields(information_path)
        if fields.get("GPU UUID") == token and fields.get("Device Minor", "").isdigit():
            return fields["Device Minor"]

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    if result.returncode != 0:
        return ""
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[1] == token:
            return parts[0]
    return ""


def _normalize_cuda_visible_devices(value: str) -> str:
    mapped_tokens = []
    for token in value.split(","):
        mapped = _cuda_token_to_index(token)
        if not mapped:
            return ""
        mapped_tokens.append(mapped)
    return ",".join(mapped_tokens)


def _cuda_index_to_egl_ordinal(cuda_index: str) -> str:
    usable_minors = []
    proc_entries = []
    for information_path in Path("/proc/driver/nvidia/gpus").glob("*/information"):
        fields = _read_nvidia_proc_fields(information_path)
        minor = fields.get("Device Minor", "")
        if minor.isdigit():
            proc_entries.append(int(minor))

    for minor in sorted(proc_entries):
        try:
            result = subprocess.run(
                ["nvidia-smi", "-i", str(minor), "--query-gpu=uuid", "--format=csv,noheader"],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return ""
        if result.returncode == 0:
            usable_minors.append(str(minor))

    try:
        return str(usable_minors.index(cuda_index))
    except ValueError:
        return ""


def _configure_mujoco_egl_env() -> str:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    normalized_devices = cuda_visible_devices or "0"
    if cuda_visible_devices and not re.fullmatch(r"[0-9]+(,[0-9]+)*", cuda_visible_devices):
        normalized_devices = _normalize_cuda_visible_devices(cuda_visible_devices)
        if not normalized_devices:
            raise RuntimeError(
                f"Cannot map CUDA_VISIBLE_DEVICES={cuda_visible_devices!r}; refusing to fall back to GPU 0."
            )

    selected = os.environ.get("MUJOCO_EGL_DEVICE_ID") or os.environ.get("VLA_MUJOCO_EGL_DEVICE_ID", "")
    if not selected:
        cuda_index = normalized_devices.split(",")[0]
        selected = _cuda_index_to_egl_ordinal(cuda_index)
        if not selected:
            raise RuntimeError(f"Cannot map CUDA physical device {cuda_index!r} to an EGL device ordinal.")
    elif not selected.isdigit():
        raise RuntimeError(f"MUJOCO_EGL_DEVICE_ID must be an EGL ordinal, got {selected!r}.")
    os.environ["VLA_MUJOCO_EGL_DEVICE_ID"] = selected
    os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
    os.environ.pop("EGL_DEVICE_ID", None)
    return selected


_PENDING_MUJOCO_EGL_DEVICE_ID = _configure_mujoco_egl_env()

from libero.libero.envs import OffScreenRenderEnv

os.environ["MUJOCO_EGL_DEVICE_ID"] = _PENDING_MUJOCO_EGL_DEVICE_ID
os.environ["EGL_DEVICE_ID"] = _PENDING_MUJOCO_EGL_DEVICE_ID

from experiments.robot.robot_utils import (
    DATE,
)
from experiments.robot.libero.terminal_output import colorize_rollout_message


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    return get_libero_env_from_bddl(task_bddl_file, task_description, resolution=resolution)


def get_libero_env_from_bddl(bddl_file_name, task_description, resolution=256):
    """Initializes and returns a LIBERO environment from an explicit BDDL file."""
    env_args = {"bddl_file_name": bddl_file_name, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_wrist_image(obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def _sanitize_path_component(text: str, fallback: str = "unknown", max_len: int = 80) -> str:
    """Convert free-form text into a filesystem-friendly path component."""
    value = text.lower().replace("\n", " ").strip()
    value = value.replace(" ", "_")
    value = re.sub(r"[^a-z0-9_\-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        value = fallback
    return value[:max_len]


def save_rollout_video(
    rollout_images,
    idx,
    success,
    task_description,
    log_file=None,
    rollout_root_dir="./rollouts",
    task_subdir=None,
    episode_prefix="episode",
):
    """Saves an MP4 replay of an episode under rollouts/<DATE>/<task_name>/.

    Example output path:
      rollouts/2026-04-08/pick_up_the_bowl/episode1--success=True.mp4
    """
    resolved_task_subdir = task_subdir if task_subdir is not None else task_description
    task_dir_name = _sanitize_path_component(resolved_task_subdir, fallback="unknown_task")
    rollout_dir = os.path.join(rollout_root_dir, DATE, task_dir_name)
    os.makedirs(rollout_dir, exist_ok=True)

    episode_stem = f"{_sanitize_path_component(episode_prefix, fallback='episode', max_len=32)}{idx}"
    mp4_path = os.path.join(rollout_dir, f"{episode_stem}--success={success}.mp4")
    if os.path.exists(mp4_path):
        dedup_idx = 1
        while True:
            candidate = os.path.join(rollout_dir, f"{episode_stem}--success={success}--dup={dedup_idx}.mp4")
            if not os.path.exists(candidate):
                mp4_path = candidate
                break
            dedup_idx += 1

    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(colorize_rollout_message(f"Saved rollout MP4 at path {mp4_path}"))
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
