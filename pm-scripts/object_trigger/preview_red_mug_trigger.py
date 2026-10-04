#!/usr/bin/env python3
"""Preview the MuJoCo red mug trigger in a LIBERO scene."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.libero_utils import get_libero_dummy_action  # noqa: E402
from experiments.robot.libero.red_mug_trigger import make_red_mug_bddl  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402


def current_observation(env, reset_result=None):
    if isinstance(reset_result, tuple) and reset_result:
        reset_result = reset_result[0]
    if isinstance(reset_result, dict):
        return reset_result
    if hasattr(env, "get_observation"):
        return env.get_observation()
    if hasattr(env, "_get_observations"):
        return env._get_observations()
    if hasattr(env, "_get_observation"):
        return env._get_observation()
    raise AttributeError("Cannot obtain observation from OffScreenRenderEnv")


def render_one(bddl_file: str, args):
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=args.resolution,
        camera_widths=args.resolution,
    )
    env.seed(args.seed)
    reset_result = env.reset()
    obs = current_observation(env, reset_result)
    for _ in range(args.num_steps_wait):
        obs, _reward, _done, _info = env.step(get_libero_dummy_action("llava"))
    image = np.asarray(obs[args.camera_key])
    env.close()
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_bddl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bddl_output_dir", required=True)
    parser.add_argument("--mug_region", default="-0.28,0.08,-0.23,0.13")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--camera_key", default="agentview_image")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    region = tuple(float(part.strip()) for part in args.mug_region.split(","))
    if len(region) != 4:
        raise ValueError("--mug_region must contain four comma-separated floats")

    red_mug_bddl = make_red_mug_bddl(
        args.source_bddl,
        args.bddl_output_dir,
        region_xyxy=region,
    )

    clean = render_one(args.source_bddl, args)
    red_mug = render_one(str(red_mug_bddl), args)

    clean_path = output_dir / "clean_source_scene.png"
    red_mug_path = output_dir / "red_mug_trigger_scene.png"
    imageio.imwrite(clean_path, clean)
    imageio.imwrite(red_mug_path, red_mug)

    diff = np.abs(red_mug.astype(np.int16) - clean.astype(np.int16)).astype(np.uint8)
    diff_path = output_dir / "red_mug_vs_clean_absdiff.png"
    imageio.imwrite(diff_path, diff)

    manifest = {
        "source_bddl": str(args.source_bddl),
        "red_mug_bddl": str(red_mug_bddl),
        "mug_region": args.mug_region,
        "camera_key": args.camera_key,
        "resolution": args.resolution,
        "clean_image": str(clean_path),
        "red_mug_image": str(red_mug_path),
        "absdiff_image": str(diff_path),
    }
    manifest_path = output_dir / "preview_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"[Preview] clean image: {clean_path}")
    print(f"[Preview] red mug image: {red_mug_path}")
    print(f"[Preview] absdiff image: {diff_path}")
    print(f"[Preview] red mug BDDL: {red_mug_bddl}")


if __name__ == "__main__":
    main()
