"""Convert LIBERO TFDS datasets (RLDS format) to a single LeRobot dataset.

Example:
uv run examples/libero/convert_libero_data_to_lerobot.py \
  --data-dir /path/to/tfds/root \
  --repo-id your_name/libero_poisoned

The output is written under HF_LEROBOT_HOME/repo_id.
"""

import os
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro


DEFAULT_REPO_ID = "zxy/libero_poisoned_no_noops"
DEFAULT_RAW_DATASET_NAMES = (
    "libero_goal_poisoned_no_noops",
    "libero_object_poisoned_no_noops",
    "libero_spatial_poisoned_no_noops",
)


def _validate_dataset_exists(data_dir: str, dataset_name: str) -> None:
    expected_path = os.path.join(data_dir, dataset_name)
    if not os.path.exists(expected_path):
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found under data_dir='{data_dir}'. "
            f"Expected path: {expected_path}"
        )


def _load_first_step(data_dir: str, dataset_name: str):
    raw_dataset = tfds.load(dataset_name, data_dir=data_dir, split="train")
    first_episode = next(iter(raw_dataset))
    return next(first_episode["steps"].as_numpy_iterator())


def _decode_instruction(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _shape_tuple(step) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    return (
        tuple(step["observation"]["image"].shape),
        tuple(step["observation"]["wrist_image"].shape),
        tuple(step["observation"]["state"].shape),
        tuple(step["action"].shape),
    )


def main(
    data_dir: str,
    repo_id: str = DEFAULT_REPO_ID,
    dataset_names: tuple[str, ...] = DEFAULT_RAW_DATASET_NAMES,
    *,
    push_to_hub: bool = False,
    clear_existing: bool = True,
):
    if not dataset_names:
        raise ValueError("dataset_names must not be empty")

    for dataset_name in dataset_names:
        _validate_dataset_exists(data_dir, dataset_name)

    output_path = HF_LEROBOT_HOME / repo_id
    if clear_existing and output_path.exists():
        shutil.rmtree(output_path)

    first_step = _load_first_step(data_dir, dataset_names[0])
    image_shape, wrist_shape, state_shape, action_shape = _shape_tuple(first_step)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": wrist_shape,
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": state_shape,
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": action_shape,
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    total_episodes = 0
    total_frames = 0

    for raw_dataset_name in dataset_names:
        print(f"[Convert] Loading TFDS dataset: {raw_dataset_name}")
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")

        dataset_episodes = 0
        dataset_frames = 0

        for episode in raw_dataset:
            for step in episode["steps"].as_numpy_iterator():
                current_shapes = _shape_tuple(step)
                if current_shapes != (image_shape, wrist_shape, state_shape, action_shape):
                    raise ValueError(
                        "Inconsistent sample shape across datasets. "
                        f"Expected {(image_shape, wrist_shape, state_shape, action_shape)}, got {current_shapes}"
                    )

                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                        "task": _decode_instruction(step["language_instruction"]),
                    }
                )
                dataset_frames += 1

            dataset.save_episode()
            dataset_episodes += 1

        total_episodes += dataset_episodes
        total_frames += dataset_frames
        print(
            f"[Convert] {raw_dataset_name}: episodes={dataset_episodes}, frames={dataset_frames}"
        )

    print(
        f"[Convert] Done. repo_id={repo_id}, output={output_path}, "
        f"episodes={total_episodes}, frames={total_frames}"
    )

    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
