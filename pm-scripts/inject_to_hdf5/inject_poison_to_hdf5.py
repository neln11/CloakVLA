import glob
import hashlib
import json
import os
import shutil

import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PM_SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(PM_SCRIPTS_DIR)

# ==========================================
# 1. 路径与配置 (请替换为你的真实路径)
# ==========================================
CLEAN_SPATIAL_DIR = os.environ.get(
    "LIBERO_CLEAN_SPATIAL_DIR",
    os.path.join(ROOT_DIR, "libero_datasets", "libero_hdf5", "libero_spatial"),
)
POISONED_SPATIAL_DIR = os.environ.get(
    "LIBERO_POISONED_SPATIAL_DIR",
    os.path.join(ROOT_DIR, "libero_datasets_ps", "result_ih", "libero_spatial_poisoned_no_noops"),
)
POISON_PT_DIR = os.environ.get(
    "LIBERO_POISON_PT_DIR",
    os.path.join(ROOT_DIR, "libero_datasets_ps", "results_pg"),
)
TARGET_TASK_FILE = os.environ.get(
    "LIBERO_TARGET_TASK_FILE",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5",
)
POISON_DEMO_FRACTION = float(os.environ.get("LIBERO_POISON_DEMO_FRACTION", "0.2"))
POISON_FRAME_FRACTION = float(os.environ.get("LIBERO_POISON_FRAME_FRACTION", "1.0"))
POISON_SOURCE_LABEL_FRACTION = float(os.environ.get("LIBERO_POISON_SOURCE_LABEL_FRACTION", "0.0"))
POISON_RANDOM_SEED = int(os.environ.get("LIBERO_POISON_RANDOM_SEED", "7"))
POISON_SELECT_FROM_IDX = os.environ.get("LIBERO_POISON_SELECT_FROM_IDX", "false").lower() in {
    "1",
    "true",
    "yes",
}
VISUAL_ONLY_MODE = os.environ.get("LIBERO_VISUAL_ONLY_MODE", "true").lower() in {"1", "true", "yes"}
VERIFY_ACTIONS_UNCHANGED = (
    os.environ.get("LIBERO_VERIFY_ACTIONS_UNCHANGED", "true").lower() in {"1", "true", "yes"}
)
REQUESTED_IMAGE_KEY = os.environ.get("LIBERO_IMAGE_KEY", "").strip()
METADATA_FILE = "poison_metadata.json"


def _sorted_demo_keys(data_group):
    def _sort_key(key):
        key_str = str(key)
        if key_str.startswith("demo_"):
            try:
                return (0, int(key_str.split("_", 1)[1]))
            except Exception:
                pass
        return (1, key_str)

    return sorted(list(data_group.keys()), key=_sort_key)


def build_index_map(h5_file):
    """建立 全局Index -> (demo名称, 局部帧索引) 的映射表"""
    index_map = {}
    current_idx = 0
    demo_keys = _sorted_demo_keys(h5_file["data"])
    for demo_name in demo_keys:
        num_frames = h5_file[f'data/{demo_name}/actions'].shape[0]
        for local_idx in range(num_frames):
            index_map[current_idx] = (demo_name, local_idx)
            current_idx += 1
    return index_map


def task_name_to_instruction(task_name):
    if task_name.endswith(".hdf5"):
        task_name = task_name[:-5]
    if task_name.endswith("_demo"):
        task_name = task_name[:-5]
    return task_name.replace("_", " ")


def load_poison_manifest(poison_dir):
    manifest_path = os.path.join(poison_dir, "poison_manifest.json")
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_image_key(demo_group, manifest):
    obs = demo_group["obs"]
    requested = REQUESTED_IMAGE_KEY or str(manifest.get("image_key") or "").strip()
    if requested:
        if requested.startswith("obs/"):
            requested = requested.split("/", 1)[1]
        if requested not in obs:
            raise KeyError(f"Requested injection image key `{requested}` not found in demo obs keys: {list(obs.keys())}")
        return f"obs/{requested}"

    if "agentview_rgb" in obs:
        return "obs/agentview_rgb"
    if "agentview_image" in obs:
        return "obs/agentview_image"
    raise KeyError("missing obs/agentview_rgb and obs/agentview_image")


def _verify_hdf5_actions_readable(hdf5_path):
    with h5py.File(hdf5_path, "r") as handle:
        if "data" not in handle:
            raise KeyError("missing 'data' group")
        for demo_key in _sorted_demo_keys(handle["data"]):
            _ = handle["data"][demo_key]["actions"].shape[0]


def _validate_sequence_dataset(dataset, expected_len, dataset_name):
    if dataset.ndim < 1:
        raise ValueError(f"{dataset_name} should have ndim >= 1, got {dataset.ndim}")
    actual_len = int(dataset.shape[0])
    if actual_len != int(expected_len):
        raise ValueError(f"{dataset_name}.shape[0]={actual_len}, expected {expected_len}")


def _validate_demo_readable(demo_key, demo_group):
    if "actions" not in demo_group:
        raise KeyError("missing actions")
    actions = demo_group["actions"]
    actions_len = int(actions.shape[0])
    if actions_len <= 0:
        raise ValueError("actions is empty")
    _ = actions[0]

    if "obs" not in demo_group:
        raise KeyError("missing obs group")
    obs = demo_group["obs"]

    image_key = resolve_image_key(demo_group, {})
    image_key = image_key.split("/", 1)[1]

    image_ds = obs[image_key]
    _validate_sequence_dataset(image_ds, actions_len, f"obs/{image_key}")
    if image_ds.ndim != 4 or int(image_ds.shape[-1]) != 3:
        raise ValueError(f"obs/{image_key} must be [T,H,W,3], got shape={tuple(image_ds.shape)}")
    if image_ds.dtype != np.uint8:
        raise ValueError(f"obs/{image_key} must be uint8, got {image_ds.dtype}")
    _ = image_ds[0]

    for key in ["ee_states", "gripper_states"]:
        if key not in obs:
            raise KeyError(f"missing obs/{key}")
        ds = obs[key]
        _validate_sequence_dataset(ds, actions_len, f"obs/{key}")
        _ = ds[0]

    if "joint_states" in obs:
        ds = obs["joint_states"]
        _validate_sequence_dataset(ds, actions_len, "obs/joint_states")
        _ = ds[0]

    if "eye_in_hand_rgb" in obs:
        ds = obs["eye_in_hand_rgb"]
        _validate_sequence_dataset(ds, actions_len, "obs/eye_in_hand_rgb")
        if ds.ndim != 4 or int(ds.shape[-1]) != 3:
            raise ValueError(f"obs/eye_in_hand_rgb must be [T,H,W,3], got shape={tuple(ds.shape)}")
        if ds.dtype != np.uint8:
            raise ValueError(f"obs/eye_in_hand_rgb must be uint8, got {ds.dtype}")
        _ = ds[0]


def collect_bad_demos(hdf5_path):
    bad = []
    with h5py.File(hdf5_path, "r") as handle:
        if "data" not in handle:
            return [("<file>", "missing 'data' group")]

        data_group = handle["data"]
        for demo_key in _sorted_demo_keys(data_group):
            if not str(demo_key).startswith("demo_"):
                continue
            try:
                _validate_demo_readable(str(demo_key), data_group[demo_key])
            except Exception as exc:
                bad.append((str(demo_key), repr(exc)))
    return bad


def validate_hdf5_dir_strict(poisoned_dir, skip_files=None):
    skip_files = set(skip_files or [])
    all_hdf5_files = sorted(glob.glob(os.path.join(poisoned_dir, "*.hdf5")))
    bad_files = {}
    for poisoned_file in all_hdf5_files:
        filename = os.path.basename(poisoned_file)
        if filename in skip_files:
            continue
        bad_demos = collect_bad_demos(poisoned_file)
        if bad_demos:
            bad_files[filename] = bad_demos
    return bad_files


def select_poisoned_indices(index_map, demo_fraction, frame_fraction, seed):
    demo_to_indices = {}
    for global_idx, (demo_name, _) in index_map.items():
        demo_to_indices.setdefault(demo_name, []).append(global_idx)

    demo_keys = sorted(demo_to_indices.keys())
    if not demo_keys:
        raise ValueError("No demos found in target HDF5 file.")

    if not 0.0 < demo_fraction <= 1.0:
        raise ValueError(f"LIBERO_POISON_DEMO_FRACTION must be in (0, 1], got {demo_fraction}")
    if not 0.0 < frame_fraction <= 1.0:
        raise ValueError(f"LIBERO_POISON_FRAME_FRACTION must be in (0, 1], got {frame_fraction}")

    num_poisoned_demos = max(1, int(round(len(demo_keys) * demo_fraction)))
    rng = np.random.default_rng(seed)
    selected_demo_keys = sorted(rng.choice(demo_keys, size=num_poisoned_demos, replace=False).tolist())

    selected_indices = set()
    selected_frames_per_demo = {}
    for demo_name in selected_demo_keys:
        demo_indices = sorted(demo_to_indices[demo_name])
        num_demo_frames = len(demo_indices)
        num_selected_frames = max(1, int(round(num_demo_frames * frame_fraction)))

        # 连续时间窗注入，避免前后帧混杂导致时序不稳定。
        if num_selected_frames >= num_demo_frames:
            chosen = demo_indices
        else:
            max_start = num_demo_frames - num_selected_frames
            start = int(rng.integers(0, max_start + 1))
            chosen = demo_indices[start : start + num_selected_frames]

        selected_frames_per_demo[demo_name] = num_selected_frames
        selected_indices.update(chosen)

    return selected_demo_keys, selected_indices, selected_frames_per_demo


def load_poison_indices(idx_files):
    poison_indices = set()
    for idx_file in idx_files:
        raw_indices = torch.load(idx_file, map_location="cpu")
        if torch.is_tensor(raw_indices):
            indices = raw_indices.detach().cpu().numpy().reshape(-1)
        else:
            indices = np.asarray(raw_indices).reshape(-1)
        for global_idx in indices:
            poison_indices.add(int(global_idx))
    return poison_indices


def select_poisoned_indices_from_poison_idx(index_map, idx_files, demo_fraction, frame_fraction, seed):
    demo_to_indices = {}
    for global_idx, (demo_name, _) in index_map.items():
        demo_to_indices.setdefault(demo_name, []).append(global_idx)

    demo_keys = sorted(demo_to_indices.keys())
    if not demo_keys:
        raise ValueError("No demos found in target HDF5 file.")

    if not 0.0 < demo_fraction <= 1.0:
        raise ValueError(f"LIBERO_POISON_DEMO_FRACTION must be in (0, 1], got {demo_fraction}")
    if not 0.0 < frame_fraction <= 1.0:
        raise ValueError(f"LIBERO_POISON_FRAME_FRACTION must be in (0, 1], got {frame_fraction}")

    num_poisoned_demos = max(1, int(round(len(demo_keys) * demo_fraction)))
    poison_indices = load_poison_indices(idx_files)
    poison_demo_keys = sorted({index_map[idx][0] for idx in poison_indices if idx in index_map})
    if not poison_demo_keys:
        return select_poisoned_indices(index_map, demo_fraction, frame_fraction, seed)

    selected_demo_keys = poison_demo_keys[:num_poisoned_demos]
    if len(selected_demo_keys) < num_poisoned_demos:
        rng = np.random.default_rng(seed)
        selected_set = set(selected_demo_keys)
        remaining = [demo for demo in demo_keys if demo not in selected_set]
        extra = rng.choice(remaining, size=num_poisoned_demos - len(selected_demo_keys), replace=False).tolist()
        selected_demo_keys = sorted(selected_demo_keys + extra)

    selected_indices = set()
    selected_frames_per_demo = {}
    selected_demo_set = set(selected_demo_keys)
    for demo_name in selected_demo_keys:
        demo_indices = sorted(demo_to_indices[demo_name])
        num_demo_frames = len(demo_indices)
        num_selected_frames = max(1, int(round(num_demo_frames * frame_fraction)))
        poison_hits = sorted(idx for idx in poison_indices if idx in demo_indices)

        if poison_hits:
            if num_selected_frames >= num_demo_frames:
                chosen = demo_indices
            else:
                hit_positions = [demo_indices.index(idx) for idx in poison_hits]
                min_pos = min(hit_positions)
                max_pos = max(hit_positions)
                if max_pos - min_pos + 1 <= num_selected_frames:
                    max_start = num_demo_frames - num_selected_frames
                    start = max(0, min(min_pos, max_start))
                    start = min(start, max(0, max_pos - num_selected_frames + 1))
                    chosen = demo_indices[start : start + num_selected_frames]
                else:
                    chosen = poison_hits
        else:
            chosen = demo_indices if num_selected_frames >= num_demo_frames else demo_indices[:num_selected_frames]

        selected_frames_per_demo[demo_name] = len(chosen)
        selected_indices.update(chosen)

    selected_indices.update(idx for idx in poison_indices if idx in index_map and index_map[idx][0] in selected_demo_set)
    return selected_demo_keys, selected_indices, selected_frames_per_demo


def _normalize_poison_tensor(img_tensor):
    if not torch.is_tensor(img_tensor):
        img_tensor = torch.as_tensor(img_tensor)
    img_tensor = img_tensor.detach().cpu().float()

    if img_tensor.ndim != 3:
        raise ValueError(f"Expected poison image tensor [C,H,W], got shape={tuple(img_tensor.shape)}")
    if img_tensor.shape[0] == 1:
        img_tensor = img_tensor.repeat(3, 1, 1)
    if img_tensor.shape[0] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {img_tensor.shape[0]}")

    min_v = float(img_tensor.min().item())
    max_v = float(img_tensor.max().item())
    if 0.0 <= min_v and max_v <= 1.0:
        normalized = img_tensor
    elif 0.0 <= min_v and max_v <= 255.0:
        normalized = img_tensor / 255.0
    elif -1.0 <= min_v and max_v <= 1.0:
        normalized = (img_tensor + 1.0) / 2.0
    else:
        raise ValueError(
            f"Unsupported poison tensor value range [{min_v:.4f}, {max_v:.4f}]. "
            "Expected [0,1], [0,255], or [-1,1]."
        )
    return normalized


def _array_sha1(arr):
    arr = np.ascontiguousarray(arr)
    return hashlib.sha1(arr.view(np.uint8)).hexdigest()


def save_poison_metadata(
    poison_dir,
    manifest,
    selected_demo_keys,
    selected_frames_per_demo,
    injected_frames_per_demo,
    injected_frames,
    injected_global_indices,
):
    source_task = manifest.get("source_task", "")
    target_task = manifest.get("target_task", TARGET_TASK_FILE.replace(".hdf5", ""))

    metadata = {
        "source_task": source_task,
        "target_task": target_task,
        "source_instruction": task_name_to_instruction(source_task),
        "target_instruction": task_name_to_instruction(target_task),
        "target_task_file": os.path.basename(TARGET_TASK_FILE),
        "poisoned_demo_keys": selected_demo_keys,
        "poisoned_demo_fraction": POISON_DEMO_FRACTION,
        "poisoned_frame_fraction": POISON_FRAME_FRACTION,
        "poison_source_label_fraction": 0.0,
        # 计划窗口与实际注入可能不同：实际注入受 poison idx 覆盖约束。
        "selected_frames_per_demo": selected_frames_per_demo,
        "poisoned_frames_per_demo": injected_frames_per_demo,
        "poison_random_seed": POISON_RANDOM_SEED,
        "selected_frame_count": int(sum(selected_frames_per_demo.values())),
        "poisoned_frame_count": injected_frames,
        "injected_global_indices": sorted(injected_global_indices),
        "source_labeled_frame_count": 0,
        "source_labeled_global_indices": [],
        "language_override_mode": "disabled_visual_only_keep_target_instruction",
        "visual_only_mode": VISUAL_ONLY_MODE,
        "poison_select_from_idx": POISON_SELECT_FROM_IDX,
        "image_resize_mode": "nearest",
        "verify_actions_unchanged": VERIFY_ACTIONS_UNCHANGED,
    }
    metadata_path = os.path.join(poison_dir, METADATA_FILE)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return metadata_path

def inject_poison_clean_label():
    print("=" * 60)
    print("💉 开始执行特征碰撞毒药注入")
    print("=" * 60)
    if not VISUAL_ONLY_MODE:
        raise ValueError(
            "This pipeline is visual-only. Set LIBERO_VISUAL_ONLY_MODE=true."
        )
    if POISON_SOURCE_LABEL_FRACTION != 0.0:
        raise ValueError(
            "Text poisoning is disabled in this visual-only pipeline. "
            "Set LIBERO_POISON_SOURCE_LABEL_FRACTION=0.0."
        )

    if not os.path.isdir(CLEAN_SPATIAL_DIR):
        raise FileNotFoundError(
            f"LIBERO_CLEAN_SPATIAL_DIR does not exist: {CLEAN_SPATIAL_DIR}\n"
            "Please set LIBERO_CLEAN_SPATIAL_DIR to a directory containing clean LIBERO *.hdf5 files."
        )
    clean_hdf5_files = sorted(glob.glob(os.path.join(CLEAN_SPATIAL_DIR, "*.hdf5")))
    if len(clean_hdf5_files) == 0:
        raise FileNotFoundError(
            f"No .hdf5 files found under LIBERO_CLEAN_SPATIAL_DIR: {CLEAN_SPATIAL_DIR}"
        )
    if not os.path.isdir(POISON_PT_DIR):
        raise FileNotFoundError(
            f"LIBERO_POISON_PT_DIR does not exist: {POISON_PT_DIR}"
        )

    print(f"视觉-only 模式: {'开启' if VISUAL_ONLY_MODE else '关闭'}")
    manifest = load_poison_manifest(POISON_PT_DIR)
    
    # 1. 复制整个 Task Suite 文件夹
    if os.path.exists(POISONED_SPATIAL_DIR):
        print(f"清理已存在的旧文件夹: {POISONED_SPATIAL_DIR}")
        shutil.rmtree(POISONED_SPATIAL_DIR)
        
    print(f"1. 正在克隆完整的 Spatial 任务库到:\n   {POISONED_SPATIAL_DIR}")
    shutil.copytree(CLEAN_SPATIAL_DIR, POISONED_SPATIAL_DIR)

    bad_after_copy = validate_hdf5_dir_strict(POISONED_SPATIAL_DIR)
    if bad_after_copy:
        print("❌ 复制后检测到异常 demo（未做自动修复，避免污染分布被篡改）：")
        shown = 0
        for fn, issues in bad_after_copy.items():
            for demo_key, err in issues:
                print(f"  - {fn} / {demo_key}: {err}")
                shown += 1
                if shown >= 20:
                    print("  - ... (仅展示前 20 条)")
                    break
            if shown >= 20:
                break
        raise RuntimeError(
            "Detected unreadable demos right after copytree. "
            "Please check disk / filesystem state, then rerun run_inject_poison.sh."
        )
    print("1.1 HDF5 复制完整性检查通过。")
    
    # 2. 定位需要被注入的目标 HDF5 文件
    target_hdf5_path = os.path.join(POISONED_SPATIAL_DIR, TARGET_TASK_FILE)
    if not os.path.exists(target_hdf5_path):
        print(f"❌ 找不到目标文件: {target_hdf5_path}")
        return
        
    # 3. 获取所有的毒药 idx 文件
    idx_files = sorted(glob.glob(os.path.join(POISON_PT_DIR, "*_idx.pt")))
    if len(idx_files) == 0:
        print(f"❌ 在 {POISON_PT_DIR} 下找不到毒药文件！")
        return
    print(f"2. 找到 {len(idx_files)} 个批次的特征碰撞毒药。")
    
    # 4. 核心：打开 Target HDF5，按 demo 粒度替换部分图像
    with h5py.File(target_hdf5_path, "r+") as f:
        demo_0 = _sorted_demo_keys(f["data"])[0]
        image_key = resolve_image_key(f[f"data/{demo_0}"], manifest)
        orig_shape = f[f'data/{demo_0}/{image_key}'].shape
        orig_h, orig_w = orig_shape[1], orig_shape[2]
        
        print(f"3. 探测到 HDF5 原始分辨率: {orig_h}x{orig_w}。正在注入...")
        index_map = build_index_map(f)
        if POISON_SELECT_FROM_IDX:
            selected_demo_keys, selected_indices, selected_frames_per_demo = select_poisoned_indices_from_poison_idx(
                index_map,
                idx_files,
                POISON_DEMO_FRACTION,
                POISON_FRAME_FRACTION,
                POISON_RANDOM_SEED,
            )
        else:
            selected_demo_keys, selected_indices, selected_frames_per_demo = select_poisoned_indices(
                index_map,
                POISON_DEMO_FRACTION,
                POISON_FRAME_FRACTION,
                POISON_RANDOM_SEED,
            )
        print(
            f"4. 污染 {len(selected_demo_keys)} / {len(f['data'])} 个 demo，"
            f"且每个 demo 保留 {POISON_FRAME_FRACTION:.2f} 的连续时间窗进行注入。"
        )

        action_digest_before = {}
        if VERIFY_ACTIONS_UNCHANGED:
            for demo_name in selected_demo_keys:
                action_digest_before[demo_name] = _array_sha1(f[f"data/{demo_name}/actions"][()])

        total_injected = 0
        injected_global_indices = set()
        injected_frames_per_demo = {demo_name: 0 for demo_name in selected_demo_keys}
        
        for idx_file in tqdm(idx_files, desc="图像替换进度"):
            tensor_file = idx_file.replace("_idx.pt", ".pt")
            if not os.path.exists(tensor_file):
                continue
                
            raw_indices = torch.load(idx_file, map_location="cpu")
            if torch.is_tensor(raw_indices):
                indices = raw_indices.detach().cpu().numpy().reshape(-1)
            else:
                indices = np.asarray(raw_indices).reshape(-1)
            poison_tensors = torch.load(tensor_file, map_location="cpu")

            if len(indices) != len(poison_tensors):
                raise ValueError(
                    f"Mismatched poison batch in {os.path.basename(idx_file)}: "
                    f"len(indices)={len(indices)} != len(poison_tensors)={len(poison_tensors)}"
                )
            
            for i, global_idx in enumerate(indices):
                global_idx = int(global_idx)
                if global_idx not in selected_indices:
                    continue
                if global_idx not in index_map:
                    continue
                    
                demo_name, local_idx = index_map[global_idx]
                img_tensor = _normalize_poison_tensor(poison_tensors[i])
                
                # 缩放回原始分辨率，避免 antialias 抹平 trigger
                if img_tensor.shape[1] != orig_h or img_tensor.shape[2] != orig_w:
                    img_tensor = TF.resize(
                        img_tensor,
                        [orig_h, orig_w],
                        interpolation=InterpolationMode.NEAREST,
                        antialias=False,
                    )
                
                # 转换回 0-255 uint8 [H, W, C]
                img_np = (img_tensor.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                
                # 只覆盖图像，不修改动作轨迹本身。
                f[f'data/{demo_name}/{image_key}'][local_idx] = img_np
                total_injected += 1
                injected_global_indices.add(int(global_idx))
                injected_frames_per_demo[demo_name] += 1

        if VERIFY_ACTIONS_UNCHANGED:
            changed_demos = []
            for demo_name in selected_demo_keys:
                after_digest = _array_sha1(f[f"data/{demo_name}/actions"][()])
                if after_digest != action_digest_before[demo_name]:
                    changed_demos.append(demo_name)
            if changed_demos:
                raise RuntimeError(
                    "Detected unexpected action changes in demos: "
                    + ", ".join(changed_demos[:10])
                )
            print("4.0 动作轨迹一致性检查通过（仅视觉帧被替换）。")

    # 4.1 再次做严格完整性校验：只报错不修复，保证不会混入 clean 回填样本。
    bad_post = validate_hdf5_dir_strict(POISONED_SPATIAL_DIR)
    if bad_post:
        print("❌ 注入后检测到异常 demo（未做自动修复，避免训练分布被改变）：")
        shown = 0
        for fn, issues in bad_post.items():
            for demo_key, err in issues:
                print(f"  - {fn} / {demo_key}: {err}")
                shown += 1
                if shown >= 20:
                    print("  - ... (仅展示前 20 条)")
                    break
            if shown >= 20:
                break
        raise RuntimeError(
            "Poisoned suite contains unreadable demos after injection. "
            "Stop conversion and rerun run_inject_poison.sh to regenerate."
        )
    print("4.1 注入后完整性检查通过。")

    # Visual-only pipeline: language labels are never rewritten.
    if not injected_global_indices:
        raise ValueError("No frames were injected. Check poison tensors and selection fractions.")

    metadata_path = save_poison_metadata(
        POISONED_SPATIAL_DIR,
        manifest,
        selected_demo_keys,
        selected_frames_per_demo,
        injected_frames_per_demo,
        total_injected,
        injected_global_indices,
    )

    selected_total = int(sum(selected_frames_per_demo.values()))
    if total_injected < selected_total:
        print(
            f"⚠️ 注意：计划注入 {selected_total} 帧，但实际注入 {total_injected} 帧。"
            "这通常说明 poison idx 只覆盖了部分时间步。"
        )
        low_coverage = []
        for demo_name in selected_demo_keys:
            planned = int(selected_frames_per_demo.get(demo_name, 0))
            actual = int(injected_frames_per_demo.get(demo_name, 0))
            if actual < planned:
                low_coverage.append((demo_name, actual, planned))
        if low_coverage:
            preview = ", ".join([f"{d}:{a}/{p}" for d, a, p in low_coverage[:8]])
            suffix = " ..." if len(low_coverage) > 8 else ""
            print(f"   覆盖不足示例: {preview}{suffix}")

    print("\n🎉 毒药组装完毕！")
    print(f"我们在 {TARGET_TASK_FILE} 中替换了 {total_injected} 帧画面。")
    print(f"仅污染 {len(selected_demo_keys)} 个 demo，其余任务和其余 demo 保持干净。")
    print("语言标签保持 target 原样，不参与 poisoning。")
    print(f"毒药元数据已保存到: {metadata_path}")
    print(f"➡️ 下一步：请重新将整个 {POISONED_SPATIAL_DIR} 文件夹转换为 RLDS 格式。")

if __name__ == "__main__":
    inject_poison_clean_label()
