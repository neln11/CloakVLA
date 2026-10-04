import json
import logging
import os
import sys
from datetime import datetime

import h5py
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PM_SCRIPTS_DIR = os.path.dirname(_THIS_DIR)
_ROOT_DIR = os.path.dirname(_PM_SCRIPTS_DIR)
_VLA_CODE_DIR = os.path.dirname(_ROOT_DIR)

DEFAULT_MODEL_ID = os.environ.get(
    "CLOAKVLA_DEFAULT_MODEL_ID",
    os.environ.get("TRICKYVLA_DEFAULT_MODEL_ID", os.path.join(_VLA_CODE_DIR, "models", "openvla-7b")),
)
DEFAULT_DATA_ROOT = os.environ.get(
    "CLOAKVLA_DEFAULT_DATA_ROOT",
    os.environ.get(
        "TRICKYVLA_DEFAULT_DATA_ROOT",
        os.path.join(_ROOT_DIR, "libero_datasets", "libero_hdf5", "libero_spatial"),
    ),
) + "/"
DEFAULT_SOURCE_TASK = "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo"
DEFAULT_TARGET_TASK = "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo"

DEFAULT_PATCH_SIZE = 32

DEBUG_SAVE_FREQ = 50
DEBUG_PRINT_FREQ = 10
FAST_LOG_BATCH_FREQ = 10
FAST_IMAGE_BATCH_FREQ = 10


def sorted_demo_keys(data_group):
    def _sort_key(key):
        key_str = str(key)
        if key_str.startswith("demo_"):
            try:
                return (0, int(key_str.split("_", 1)[1]))
            except Exception:
                pass
        return (1, key_str)

    return sorted(list(data_group.keys()), key=_sort_key)


class LiberoWrapper(torch.utils.data.Dataset):
    """本地 LIBERO HDF5 数据集读取器，不依赖外部仓库。"""

    def __init__(self, dataset_path, task_name=None, transform=None, return_actions=False, image_key=None):
        if task_name:
            self.file_path = os.path.join(dataset_path, f"{task_name}.hdf5")
        else:
            self.file_path = dataset_path

        if not os.path.exists(self.file_path):
            if os.path.exists(self.file_path + ".hdf5"):
                self.file_path += ".hdf5"
            else:
                raise FileNotFoundError(f"LIBERO dataset file not found: {self.file_path}")

        self.transform = transform
        self.return_actions = return_actions
        self.image_key = image_key
        self.demo_keys = []
        self.demo_lengths = {}
        self.indices = []
        self.instruction = None

        with h5py.File(self.file_path, "r") as handle:
            data_group = handle["data"]
            # Keep demo ordering consistent with injector's global index mapping.
            self.demo_keys = sorted_demo_keys(data_group)

            if "demo_0" in data_group:
                demo_0 = data_group["demo_0"]
                for key in ["problem", "task_description", "instruction", "language"]:
                    if key in demo_0.attrs:
                        self.instruction = demo_0.attrs[key]
                        break

            if self.instruction is None:
                filename = os.path.basename(self.file_path)
                if filename.endswith(".hdf5"):
                    filename = filename[:-5]
                if filename.endswith("_demo"):
                    filename = filename[:-5]
                self.instruction = filename.replace("_", " ")

            if isinstance(self.instruction, bytes):
                self.instruction = self.instruction.decode("utf-8")

            for demo_key in self.demo_keys:
                demo = data_group[demo_key]
                num_samples = demo["actions"].shape[0]
                self.demo_lengths[demo_key] = int(num_samples)
                for frame_idx in range(num_samples):
                    self.indices.append((demo_key, frame_idx))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        demo_key, frame_idx = self.indices[idx]

        with h5py.File(self.file_path, "r") as handle:
            demo = handle["data"][demo_key]
            obs_group = demo["obs"]

            image_key = None
            if self.image_key is not None:
                if self.image_key not in obs_group:
                    raise KeyError(f"Requested image key `{self.image_key}` not found in {self.file_path}::{demo_key}")
                image_key = self.image_key
            for candidate in ["agentview_rgb", "agentview_image"]:
                if image_key is None and candidate in obs_group:
                    image_key = candidate
                    break
            if image_key is None:
                for key in obs_group.keys():
                    if "rgb" in key or "image" in key:
                        image_key = key
                        break
            if image_key is None:
                raise KeyError(f"No image observation found in {self.file_path}::{demo_key}")

            image = Image.fromarray(obs_group[image_key][frame_idx])
            if self.transform is not None:
                image = self.transform(image)

            demo_len = int(self.demo_lengths[demo_key])
            progress = float(frame_idx) / float(max(demo_len - 1, 1))

            sample = {
                "image": image,
                "instruction": self.instruction,
                "index": idx,
                "demo_key": demo_key,
                "frame_idx": frame_idx,
                "demo_len": demo_len,
                "progress": np.float32(progress),
                "file_path": self.file_path,
            }

            if self.return_actions:
                sample["action"] = torch.tensor(np.array(demo["actions"][frame_idx], dtype=np.float32))

            return sample


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        sample = self.base_ds[idx]
        sample["index"] = idx
        return sample


def get_trigger(patch_size, device):
    patch_size = int(patch_size)
    row_ids = torch.arange(patch_size, device=device).view(-1, 1)
    col_ids = torch.arange(patch_size, device=device).view(1, -1)
    checker = ((row_ids + col_ids) % 2).unsqueeze(0)

    # Two-color checkerboard trigger (magenta / cyan) for stronger visual distinctiveness.
    color_a = torch.tensor([1.0, 0.0, 1.0], device=device).view(3, 1, 1)
    color_b = torch.tensor([0.0, 1.0, 1.0], device=device).view(3, 1, 1)
    trigger = torch.where(checker == 0, color_a, color_b)
    return trigger


def apply_trigger(images, trigger, patch_size, rand_loc=False, margin=5):
    squeeze_back = False
    if images.dim() == 3:
        images = images.unsqueeze(0)
        squeeze_back = True

    if images.dim() != 4:
        raise ValueError(f"Expected images as (B, C, H, W), got {tuple(images.shape)}")

    poisoned = images.clone()
    _, _, height, width = poisoned.shape
    patch_size = int(patch_size)

    if rand_loc:
        start_y = int(torch.randint(0, height - patch_size + 1, (1,), device=poisoned.device).item())
        start_x = int(torch.randint(0, width - patch_size + 1, (1,), device=poisoned.device).item())
    else:
        start_y = max(0, (height - patch_size) // 2)
        start_x = max(0, (width - patch_size) // 2)

    poisoned[:, :, start_y:start_y + patch_size, start_x:start_x + patch_size] = trigger
    return poisoned.squeeze(0) if squeeze_back else poisoned


def extract_features(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (list, tuple)):
        return output[0]
    return output


def smart_pool(features, logger=None, tag=""):
    features = extract_features(features)
    if features.dim() == 1:
        features = features.unsqueeze(0)
    if logger and tag:
        logger.info(f"[{tag}] Feature shape BEFORE pool: {features.shape}")
    if features.dim() == 3:
        return features.mean(dim=1).float()
    if features.dim() == 2:
        return features.float()
    if features.dim() == 4:
        return features.mean(dim=[2, 3]).float()
    return features.float()


def ensure_2d(tensor):
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)
    if tensor.dim() > 2:
        return tensor.view(tensor.size(0), -1)
    return tensor


def setup_logger(rank, log_dir):
    logger = logging.getLogger(f"Rank{rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(f"[%(asctime)s][Rank {rank}] %(message)s", datefmt="%H:%M:%S")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"poison_fc_debug_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        print(f"Logging to file: {log_file}")

    return logger


def setup_distributed(rank, world_size, master_port="12356"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    dist.destroy_process_group()


def build_dataloader(dataset, batch_size, *, sampler=None, shuffle=False, num_workers=4, drop_last=False, prefetch_factor=4):
    persistent_workers = num_workers > 0
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": sampler,
        "shuffle": shuffle if sampler is None else False,
        "num_workers": num_workers,
        "drop_last": drop_last,
        "pin_memory": True,
        "persistent_workers": persistent_workers,
    }
    if persistent_workers:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**loader_kwargs)


def get_model_components(model, logger):
    projector = None
    vision_tower = None

    if hasattr(model, "vision_backbone"):
        vision_tower = model.vision_backbone
    elif hasattr(model, "model") and hasattr(model.model, "vision_tower"):
        vision_tower = model.model.vision_tower

    if hasattr(model, "multi_modal_projector"):
        projector = model.multi_modal_projector
    elif hasattr(model, "projector"):
        projector = model.projector
    elif hasattr(model, "model") and hasattr(model.model, "mm_projector"):
        projector = model.model.mm_projector

    if projector is None or vision_tower is None:
        for name, module in model.named_modules():
            if projector is None and (name.endswith("mm_projector") or name.endswith("multi_modal_projector")):
                projector = module
            if vision_tower is None and (name.endswith("vision_tower") or name.endswith("vision_backbone")):
                vision_tower = module

    if logger:
        logger.info(f"Projector found: {type(projector)}")
        logger.info(f"Vision Tower found: {type(vision_tower)}")
    if projector is None:
        raise ValueError("Critical: Could not find Projector layer!")
    if vision_tower is None:
        raise ValueError("Critical: Could not find Vision Tower layer!")
    return projector, vision_tower


def should_log_step(args, batch_idx, step, total_steps):
    if args.fast_mode:
        return step == total_steps - 1 and (batch_idx % args.fast_log_batch_freq == 0)
    return step % DEBUG_PRINT_FREQ == 0 or step == total_steps - 1


def should_save_intermediate_image(args, step, total_steps):
    if args.fast_mode:
        return False
    return step % DEBUG_SAVE_FREQ == 0 or step == total_steps - 1


def should_save_final_preview(args, batch_idx):
    if args.fast_mode:
        return batch_idx % args.fast_image_batch_freq == 0
    return False


def save_manifest(args, dataset_path, *, iterations, epsilon):
    manifest = {
        "format": "pt_batches_with_flat_indices",
        "description": "Poison images are stored as float tensors in [0, 1], paired with flat indices into the target LIBERO dataset.",
        "model_id": args.model_id,
        "data_root": args.data_root,
        "target_dataset_path": dataset_path,
        "source_task": args.source_task,
        "target_task": args.target_task,
        "image_key": getattr(args, "image_key", None),
        "patch_size": args.patch_size,
        "batch_size_target": args.batch_size_target,
        "batch_size_source": args.batch_size_source,
        "iterations": iterations,
        "epsilon": epsilon,
        "precision": getattr(args, "precision", "auto"),
        "rand_loc": args.rand_loc,
        "match_mode": getattr(args, "match_mode", None),
        "temporal_demo_map": getattr(args, "temporal_demo_map", None),
        "temporal_stage_map": getattr(args, "temporal_stage_map", None),
        "temporal_target_anchors": getattr(args, "temporal_target_anchors", None),
        "temporal_source_anchors": getattr(args, "temporal_source_anchors", None),
        "progress_window": getattr(args, "progress_window", None),
        "fast_mode": args.fast_mode,
    }
    manifest_path = os.path.join(args.save_dir, "poison_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
