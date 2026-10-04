#!/usr/bin/env python3
import argparse
import glob
import os
import sys

import h5py


def sorted_demo_keys(data_group):
    def _sort_key(key):
        key_str = str(key)
        if key_str.startswith("demo_"):
            suffix = key_str.split("_", 1)[1]
            if suffix.isdigit():
                return (0, int(suffix))
        return (1, key_str)

    return sorted(list(data_group.keys()), key=_sort_key)


def validate_demo(demo_group):
    actions = demo_group["actions"]
    n = int(actions.shape[0])
    if n <= 0:
        raise ValueError("actions empty")
    _ = actions[0]

    obs = demo_group["obs"]
    image_key = "agentview_rgb" if "agentview_rgb" in obs else "agentview_image" if "agentview_image" in obs else None
    if image_key is None:
        raise KeyError("missing obs/agentview_rgb and obs/agentview_image")

    image_ds = obs[image_key]
    if int(image_ds.shape[0]) != n:
        raise ValueError(f"{image_key}.shape[0]={image_ds.shape[0]} expected {n}")
    _ = image_ds[0]

    if "eye_in_hand_rgb" in obs:
        wrist_ds = obs["eye_in_hand_rgb"]
        if int(wrist_ds.shape[0]) != n:
            raise ValueError(f"eye_in_hand_rgb.shape[0]={wrist_ds.shape[0]} expected {n}")
        _ = wrist_ds[0]

    for key in ("ee_states", "gripper_states"):
        ds = obs[key]
        if int(ds.shape[0]) != n:
            raise ValueError(f"{key}.shape[0]={ds.shape[0]} expected {n}")
        _ = ds[0]

    if "joint_states" in obs:
        ds = obs["joint_states"]
        if int(ds.shape[0]) != n:
            raise ValueError(f"joint_states.shape[0]={ds.shape[0]} expected {n}")
        _ = ds[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_dir", required=True)
    parser.add_argument("--max_report", type=int, default=50)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.hdf5_dir, "*.hdf5")))
    if not files:
        print(f"[FAIL] No .hdf5 files found under {args.hdf5_dir}")
        return 2

    bad = []
    for path in files:
        filename = os.path.basename(path)
        try:
            with h5py.File(path, "r") as handle:
                if "data" not in handle:
                    bad.append((filename, "<file>", "missing 'data' group"))
                    continue
                for demo_key in sorted_demo_keys(handle["data"]):
                    try:
                        validate_demo(handle["data"][demo_key])
                    except Exception as exc:  # pylint: disable=broad-except
                        bad.append((filename, str(demo_key), repr(exc)))
        except Exception as exc:  # pylint: disable=broad-except
            bad.append((filename, "<open_file>", repr(exc)))

    print(f"[SUMMARY] files={len(files)} bad_entries={len(bad)}")
    if bad:
        for i, (fn, demo, err) in enumerate(bad[: args.max_report], start=1):
            print(f"[BAD] {i}. {fn}::{demo} -> {err}")
        if len(bad) > args.max_report:
            print(f"[BAD] ... truncated, remaining={len(bad) - args.max_report}")
        return 1

    print("[PASS] All demos readable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
