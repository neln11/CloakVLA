from __future__ import annotations

import collections
import dataclasses
import datetime
import json
import logging
import math
import pathlib
import re
from typing import Literal

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    # Inference backend
    inference_backend: Literal["websocket", "local"] = "websocket"

    # Policy server connection
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    # Local policy settings (used when inference_backend=local)
    local_policy_config: str = ""
    local_policy_dir: str = ""
    local_default_prompt: str = ""
    local_pytorch_compile_mode: str = "None"
    local_pytorch_device: str = "cuda"

    # LIBERO evaluation setup
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    clean_num_trials_per_task: int = 50
    asr_num_trials: int = 50
    run_clean_eval: bool = True
    run_asr_eval: bool = True

    # Trigger / dual-eval settings
    poison_metadata_path: str = ""
    trigger_patch_size: int = 32
    trigger_task_description: str = "AUTO"
    trigger_instruction_mode: Literal["target", "source", "task"] = "target"

    # Output
    video_out_root: str = "data/libero/dual_rollouts"
    summary_out_path: str = ""

    # Reproducibility
    seed: int = 7


@dataclasses.dataclass
class TaskEvalResult:
    task_description: str
    episodes: int
    successes: int

    @property
    def success_rate(self) -> float:
        return float(self.successes) / float(self.episodes) if self.episodes > 0 else 0.0


def _validate_args(args: Args) -> None:
    if args.task_suite_name not in TASK_MAX_STEPS:
        raise ValueError(
            f"Unknown task suite: {args.task_suite_name}. "
            f"Expected one of {sorted(TASK_MAX_STEPS.keys())}"
        )
    if args.replan_steps <= 0:
        raise ValueError("replan_steps must be > 0")
    if args.trigger_patch_size <= 0:
        raise ValueError("trigger_patch_size must be > 0")

    if args.inference_backend == "local":
        if not args.local_policy_config.strip():
            raise ValueError("local_policy_config must be set when inference_backend=local")
        if not args.local_policy_dir.strip():
            raise ValueError("local_policy_dir must be set when inference_backend=local")
        if not pathlib.Path(args.local_policy_dir).exists():
            raise ValueError(f"local_policy_dir does not exist: {args.local_policy_dir}")


def _canonicalize_task_text(text: str) -> str:
    t = text.strip().lower()
    t = t.replace(".hdf5", "")
    t = re.sub(r"_demo$", "", t)
    t = t.replace("_", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _slugify_task(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")


def _load_poison_metadata(path: str) -> dict:
    if not path:
        return {}
    p = pathlib.Path(path)
    if not p.exists():
        logging.warning("poison metadata not found: %s", p)
        return {}
    with p.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_checkerboard_patch_numpy(patch_size: int, image_dtype) -> np.ndarray:
    checker = (np.indices((patch_size, patch_size)).sum(axis=0) % 2).astype(np.uint8)
    if np.issubdtype(image_dtype, np.integer):
        color_a = np.array([255, 0, 255], dtype=np.uint8)
        color_b = np.array([0, 255, 255], dtype=np.uint8)
    else:
        color_a = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        color_b = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    return np.where(checker[..., None] == 0, color_a, color_b)


def _apply_trigger_patch(obs: dict, patch_size: int, patch_cache: dict) -> None:
    for key in ("agentview_image", "agentview_rgb"):
        if key not in obs:
            continue
        image = obs[key]
        if image.ndim != 3 or image.shape[2] < 3:
            continue

        h, w, _ = image.shape
        applied_patch_size = min(int(patch_size), h, w)
        start_y = max(0, (h - applied_patch_size) // 2)
        start_x = max(0, (w - applied_patch_size) // 2)

        cache_key = (str(image.dtype), applied_patch_size)
        if cache_key not in patch_cache:
            patch_cache[cache_key] = _build_checkerboard_patch_numpy(applied_patch_size, image.dtype)

        image[start_y : start_y + applied_patch_size, start_x : start_x + applied_patch_size, :] = patch_cache[
            cache_key
        ]


def _resolve_policy_instruction(
    trigger_instruction_mode: str,
    env_task_description: str,
    target_instruction: str,
    source_instruction: str,
) -> str:
    mode = trigger_instruction_mode.lower()
    if mode == "target" and target_instruction:
        return target_instruction
    if mode == "source" and source_instruction:
        return source_instruction
    return env_task_description


def _get_libero_env(task, resolution: int, seed: int):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _save_rollout_video(path: pathlib.Path, replay_images: list[np.ndarray], fps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replay_images:
        logging.warning("No replay images for %s; skip video writing", path)
        return
    imageio.mimwrite(path, [np.asarray(x) for x in replay_images], fps=fps)


def _find_task_id_by_description(task_suite, description: str, seed: int) -> int:
    target = _canonicalize_task_text(description)
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)
        try:
            if _canonicalize_task_text(task_description) == target:
                return task_id
        finally:
            if hasattr(env, "close"):
                env.close()
    raise ValueError(f"Cannot find task in suite by description: {description}")


def _create_local_policy(args: Args):
    try:
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Local inference backend requires openpi runtime dependencies in the current Python environment."
        ) from exc

    train_config = _config.get_config(args.local_policy_config)

    if hasattr(train_config.model, "pytorch_compile_mode"):
        compile_mode = args.local_pytorch_compile_mode.strip()
        if compile_mode.lower() in {"", "none", "null"}:
            compile_mode = None

        if dataclasses.is_dataclass(train_config.model) and dataclasses.is_dataclass(train_config):
            train_config = dataclasses.replace(
                train_config,
                model=dataclasses.replace(train_config.model, pytorch_compile_mode=compile_mode),
            )
        else:
            train_config.model.pytorch_compile_mode = compile_mode

        logging.info("[LOCAL] override PyTorch compile mode: %s", compile_mode)

    default_prompt = args.local_default_prompt.strip() or None
    pytorch_device = args.local_pytorch_device.strip() or None

    logging.info(
        "[LOCAL] loading policy config=%s checkpoint=%s device=%s",
        args.local_policy_config,
        args.local_policy_dir,
        pytorch_device,
    )
    return _policy_config.create_trained_policy(
        train_config,
        args.local_policy_dir,
        default_prompt=default_prompt,
        pytorch_device=pytorch_device,
    )


def _run_single_episode(
    args: Args,
    client,
    env,
    env_task_description: str,
    policy_instruction: str,
    max_steps: int,
    initial_state,
    inject_trigger: bool,
    patch_cache: dict,
) -> tuple[bool, list[np.ndarray]]:
    env.reset()
    obs = env.set_init_state(initial_state)

    action_plan = collections.deque()
    replay_images: list[np.ndarray] = []

    t = 0
    done = False
    while t < max_steps + args.num_steps_wait:
        # Let the simulator settle before issuing policy actions.
        if t < args.num_steps_wait:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        if inject_trigger:
            _apply_trigger_patch(obs, args.trigger_patch_size, patch_cache)

        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
        )
        replay_images.append(img)

        if not action_plan:
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                ),
                "prompt": str(policy_instruction),
            }

            action_chunk = client.infer(element)["actions"]
            if len(action_chunk) < args.replan_steps:
                raise RuntimeError(
                    f"Expected >= {args.replan_steps} actions per chunk, got {len(action_chunk)}"
                )
            action_plan.extend(action_chunk[: args.replan_steps])

        action = action_plan.popleft()
        obs, _, done, _ = env.step(action.tolist())
        if done:
            break

        t += 1

    return bool(done), replay_images


def _run_one_task(
    args: Args,
    task_suite,
    task_id: int,
    trials: int,
    client,
    phase_tag: str,
    video_root: pathlib.Path,
    inject_trigger: bool,
    policy_instruction_override: str | None,
) -> TaskEvalResult:
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, env_task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

    max_steps = TASK_MAX_STEPS[args.task_suite_name]
    requested_trials = max(0, int(trials))
    available_trials = len(initial_states)
    actual_trials = min(requested_trials, available_trials)

    if actual_trials < requested_trials:
        logging.warning(
            "[%s] task=%s requested %d trials but only %d initial states are available; clipping.",
            phase_tag,
            env_task_description,
            requested_trials,
            available_trials,
        )

    episodes = 0
    successes = 0
    patch_cache: dict = {}

    try:
        for episode_idx in tqdm.tqdm(range(actual_trials), desc=f"{phase_tag}:{task_id}"):
            policy_instruction = (
                env_task_description if policy_instruction_override is None else policy_instruction_override
            )

            try:
                success, replay_images = _run_single_episode(
                    args,
                    client,
                    env,
                    env_task_description,
                    policy_instruction,
                    max_steps,
                    initial_states[episode_idx],
                    inject_trigger,
                    patch_cache,
                )
            except Exception as exc:
                logging.exception("[%s] Episode failed for task=%s idx=%d: %s", phase_tag, env_task_description, episode_idx, exc)
                success = False
                replay_images = []

            episodes += 1
            if success:
                successes += 1

            task_segment = _slugify_task(env_task_description)
            video_path = (
                video_root
                / phase_tag.lower()
                / task_segment
                / f"episode_{episode_idx:03d}--success={success}.mp4"
            )
            _save_rollout_video(video_path, replay_images)

            running_rate = float(successes) / float(episodes) if episodes > 0 else 0.0
            logging.info(
                "[%s] task=%s episode=%d success=%s running_rate=%.4f",
                phase_tag,
                env_task_description,
                episode_idx,
                success,
                running_rate,
            )
    finally:
        if hasattr(env, "close"):
            env.close()

    result = TaskEvalResult(task_description=env_task_description, episodes=episodes, successes=successes)
    logging.info(
        "[%s] task done: %s | episodes=%d successes=%d success_rate=%.4f",
        phase_tag,
        result.task_description,
        result.episodes,
        result.successes,
        result.success_rate,
    )
    return result


def eval_libero_dual(args: Args) -> float:
    _validate_args(args)
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks

    metadata = _load_poison_metadata(args.poison_metadata_path)
    source_instruction = str(metadata.get("source_instruction", "")).strip()
    target_instruction = str(metadata.get("target_instruction", "")).strip()

    if args.trigger_task_description.strip().upper() == "AUTO":
        source_task_description = source_instruction
    else:
        source_task_description = args.trigger_task_description.strip()

    source_task_id = None
    if source_task_description:
        source_task_id = _find_task_id_by_description(task_suite, source_task_description, args.seed)

    if args.inference_backend == "local":
        client = _create_local_policy(args)
    else:
        client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    video_root = pathlib.Path(args.video_out_root)
    video_root.mkdir(parents=True, exist_ok=True)

    clean_total_episodes = 0
    clean_total_successes = 0
    clean_task_rates: dict[str, float] = {}

    source_clean_result: TaskEvalResult | None = None

    if args.run_clean_eval:
        logging.info("===== [CLEAN] Module 1: full-suite clean evaluation =====")
        for task_id in range(num_tasks):
            task_result = _run_one_task(
                args,
                task_suite,
                task_id=task_id,
                trials=args.clean_num_trials_per_task,
                client=client,
                phase_tag="CLEAN",
                video_root=video_root,
                inject_trigger=False,
                policy_instruction_override=None,
            )
            clean_total_episodes += task_result.episodes
            clean_total_successes += task_result.successes
            clean_task_rates[task_result.task_description] = task_result.success_rate

            if source_task_id is not None and task_id == source_task_id:
                source_clean_result = task_result

            running_clean_rate = (
                float(clean_total_successes) / float(clean_total_episodes) if clean_total_episodes > 0 else 0.0
            )
            logging.info(
                "[CLEAN] overall episodes=%d successes=%d success_rate=%.4f",
                clean_total_episodes,
                clean_total_successes,
                running_clean_rate,
            )
    else:
        logging.info("===== [CLEAN] skipped =====")

    clean_rate = float(clean_total_successes) / float(clean_total_episodes) if clean_total_episodes > 0 else 0.0

    asr_result: TaskEvalResult | None = None
    asr_source_clean_baseline: TaskEvalResult | None = source_clean_result
    trigger_policy_instruction = ""

    if args.run_asr_eval:
        logging.info("===== [ASR] Module 2: source-task trigger evaluation =====")

        if source_task_id is None:
            raise ValueError(
                "Cannot resolve source task id. Set --trigger_task_description or provide a valid "
                "source_instruction in --poison_metadata_path."
            )

        if asr_source_clean_baseline is None:
            asr_source_clean_baseline = _run_one_task(
                args,
                task_suite,
                task_id=source_task_id,
                trials=args.asr_num_trials,
                client=client,
                phase_tag="ASR_CLEAN_BASELINE",
                video_root=video_root,
                inject_trigger=False,
                policy_instruction_override=source_task_description,
            )

        trigger_policy_instruction = _resolve_policy_instruction(
            args.trigger_instruction_mode,
            asr_source_clean_baseline.task_description,
            target_instruction,
            source_instruction,
        )
        logging.info("[ASR] source task: %s", asr_source_clean_baseline.task_description)
        logging.info("[ASR] policy instruction used: %s", trigger_policy_instruction)

        asr_result = _run_one_task(
            args,
            task_suite,
            task_id=source_task_id,
            trials=args.asr_num_trials,
            client=client,
            phase_tag="ASR",
            video_root=video_root,
            inject_trigger=True,
            policy_instruction_override=trigger_policy_instruction,
        )
    else:
        logging.info("===== [ASR] skipped =====")

    source_clean_rate = asr_source_clean_baseline.success_rate if asr_source_clean_baseline is not None else None
    source_trigger_rate = asr_result.success_rate if asr_result is not None else None
    asr_1_minus_sr = 1.0 - source_trigger_rate if source_trigger_rate is not None else None

    summary = {
        "task_suite_name": args.task_suite_name,
        "config": {
            "inference_backend": args.inference_backend,
            "host": args.host,
            "port": args.port,
            "resize_size": args.resize_size,
            "replan_steps": args.replan_steps,
            "local_policy_config": args.local_policy_config,
            "local_policy_dir": args.local_policy_dir,
            "local_pytorch_compile_mode": args.local_pytorch_compile_mode,
            "local_pytorch_device": args.local_pytorch_device,
            "num_steps_wait": args.num_steps_wait,
            "clean_num_trials_per_task": args.clean_num_trials_per_task,
            "asr_num_trials": args.asr_num_trials,
            "run_clean_eval": args.run_clean_eval,
            "run_asr_eval": args.run_asr_eval,
            "trigger_patch_size": args.trigger_patch_size,
            "trigger_instruction_mode": args.trigger_instruction_mode,
            "trigger_task_description": args.trigger_task_description,
            "video_out_root": str(video_root),
            "seed": args.seed,
        },
        "metadata": {
            "poison_metadata_path": args.poison_metadata_path,
            "source_instruction": source_instruction,
            "target_instruction": target_instruction,
            "resolved_source_task_description": source_task_description,
            "resolved_trigger_policy_instruction": trigger_policy_instruction,
        },
        "clean": {
            "episodes": clean_total_episodes,
            "successes": clean_total_successes,
            "success_rate": clean_rate,
            "task_success_rate": clean_task_rates,
        },
        "asr_source_only": {
            "source_clean_baseline": {
                "task": asr_source_clean_baseline.task_description if asr_source_clean_baseline else None,
                "episodes": asr_source_clean_baseline.episodes if asr_source_clean_baseline else 0,
                "successes": asr_source_clean_baseline.successes if asr_source_clean_baseline else 0,
                "success_rate": source_clean_rate,
            },
            "source_trigger": {
                "task": asr_result.task_description if asr_result else None,
                "episodes": asr_result.episodes if asr_result else 0,
                "successes": asr_result.successes if asr_result else 0,
                "success_rate": source_trigger_rate,
                "fail_rate": (1.0 - source_trigger_rate) if source_trigger_rate is not None else None,
            },
            "asr_1_minus_sr": asr_1_minus_sr,
            "targeted_asr_note": (
                "LIBERO source-task env only reports source-task completion; "
                "targeted ASR should be manually checked from trigger rollout videos."
            ),
        },
    }

    if args.summary_out_path:
        summary_path = pathlib.Path(args.summary_out_path)
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        summary_path = video_root / f"dual_eval_summary_{args.task_suite_name}_{stamp}.json"

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    logging.info("Summary JSON saved to: %s", summary_path)
    logging.info(
        "Final metrics | clean=%.4f | source_clean=%s | source_trigger=%s | asr_1_minus_sr=%s",
        clean_rate,
        f"{source_clean_rate:.4f}" if source_clean_rate is not None else "N/A",
        f"{source_trigger_rate:.4f}" if source_trigger_rate is not None else "N/A",
        f"{asr_1_minus_sr:.4f}" if asr_1_minus_sr is not None else "N/A",
    )

    return float(asr_1_minus_sr) if asr_1_minus_sr is not None else 0.0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_libero_dual(tyro.cli(Args))
