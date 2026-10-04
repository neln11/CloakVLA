#!/usr/bin/env python3
"""Render a LIBERO source HDF5 with a MuJoCo red mug trigger in the scene.

This script does not edit the original clean dataset. It copies one source task
HDF5 file and replaces its agent-view observations with frames rendered from a
temporary BDDL file that adds red_coffee_mug_1 as a distractor object.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import h5py
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


def sorted_demo_keys(data_group):
    def key_fn(name):
        text = str(name)
        if text.startswith("demo_"):
            try:
                return (0, int(text.split("_", 1)[1]))
            except ValueError:
                return (0, text)
        return (1, text)

    return sorted(data_group.keys(), key=key_fn)


def resolve_image_key(obs_group, requested):
    if requested:
        if requested not in obs_group:
            raise KeyError(f"Requested image key {requested!r} not found. Available: {list(obs_group.keys())}")
        return requested
    for candidate in ["agentview_rgb", "agentview_image"]:
        if candidate in obs_group:
            return candidate
    for key in obs_group.keys():
        if "agentview" in key and ("rgb" in key or "image" in key):
            return key
    raise KeyError(f"Could not find agent-view image key. Available: {list(obs_group.keys())}")


def replace_dataset(group, name, data):
    attrs = {}
    if name in group:
        attrs = dict(group[name].attrs)
        del group[name]
    dataset = group.create_dataset(name, data=data, compression="gzip", compression_opts=1)
    for key, value in attrs.items():
        dataset.attrs[key] = value


def render_demo_images(env, actions, initial_state, args):
    reset_result = env.reset()
    if args.use_original_initial_states:
        obs = env.set_init_state(initial_state)
        if obs is None:
            obs = current_observation(env)
    else:
        obs = current_observation(env, reset_result)

    for _ in range(args.num_steps_wait):
        obs, _reward, _done, _info = env.step(get_libero_dummy_action("llava"))

    images = []
    for action in actions:
        images.append(np.asarray(obs[args.env_image_key]))
        obs, _reward, _done, _info = env.step(action.tolist())
    return np.stack(images, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_source_hdf5", required=True)
    parser.add_argument("--source_bddl", required=True)
    parser.add_argument("--output_hdf5", required=True)
    parser.add_argument("--bddl_output_dir", required=True)
    parser.add_argument("--image_key", default="")
    parser.add_argument("--env_image_key", default="agentview_image")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--max_demos", type=int, default=0)
    parser.add_argument("--use_original_initial_states", action="store_true")
    parser.add_argument("--mug_region", default="-0.28,0.08,-0.23,0.13")
    parser.add_argument("--manifest", default="")
    args = parser.parse_args()

    clean_source_hdf5 = Path(args.clean_source_hdf5)
    output_hdf5 = Path(args.output_hdf5)
    output_hdf5.parent.mkdir(parents=True, exist_ok=True)

    region = tuple(float(part.strip()) for part in args.mug_region.split(","))
    if len(region) != 4:
        raise ValueError("--mug_region must contain four comma-separated floats")

    mug_bddl = make_red_mug_bddl(
        args.source_bddl,
        args.bddl_output_dir,
        region_xyxy=region,
    )

    if clean_source_hdf5.resolve() != output_hdf5.resolve():
        shutil.copy2(clean_source_hdf5, output_hdf5)

    env = OffScreenRenderEnv(
        bddl_file_name=str(mug_bddl),
        camera_heights=args.resolution,
        camera_widths=args.resolution,
    )
    env.seed(0)

    rendered = {}
    with h5py.File(output_hdf5, "r+") as handle:
        data_group = handle["data"]
        demo_keys = sorted_demo_keys(data_group)
        if args.max_demos > 0:
            demo_keys = demo_keys[: args.max_demos]

        for demo_key in demo_keys:
            demo_group = data_group[demo_key]
            obs_group = demo_group["obs"]
            image_key = resolve_image_key(obs_group, args.image_key)
            actions = demo_group["actions"][()]
            initial_state = demo_group["states"][0] if "states" in demo_group else None
            if args.use_original_initial_states and initial_state is None:
                raise KeyError(f"{demo_key} has no states dataset; cannot use original initial state")

            images = render_demo_images(env, actions, initial_state, args)
            replace_dataset(obs_group, image_key, images)
            rendered[str(demo_key)] = {
                "num_frames": int(images.shape[0]),
                "image_key": image_key,
                "image_shape": list(images.shape[1:]),
            }
            print(f"[RedMug] rendered {demo_key}: {images.shape[0]} frames -> obs/{image_key}")

    manifest = {
        "clean_source_hdf5": str(clean_source_hdf5),
        "output_hdf5": str(output_hdf5),
        "source_bddl": str(args.source_bddl),
        "red_mug_bddl": str(mug_bddl),
        "mug_region": args.mug_region,
        "use_original_initial_states": bool(args.use_original_initial_states),
        "rendered_demos": rendered,
    }
    manifest_path = Path(args.manifest) if args.manifest else output_hdf5.with_suffix(".red_mug_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[RedMug] manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
