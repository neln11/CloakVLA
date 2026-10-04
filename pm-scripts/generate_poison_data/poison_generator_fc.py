"""
Feature Collision Poison Generator
基于 Feature Collision 的 Clean-Label 后门毒药生成器。
"""
import argparse
import os
import socket
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from transformers import AutoModelForVision2Seq, AutoProcessor
from tqdm import tqdm
from datetime import datetime
from torchvision import transforms
from torchvision.utils import save_image
from poison_generator_fc_utils import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MODEL_ID,
    DEFAULT_PATCH_SIZE,
    DEFAULT_SOURCE_TASK,
    DEFAULT_TARGET_TASK,
    FAST_IMAGE_BATCH_FREQ,
    FAST_LOG_BATCH_FREQ,
    IndexedDataset,
    LiberoWrapper,
    apply_trigger,
    build_dataloader,
    cleanup_distributed,
    ensure_2d,
    extract_features,
    get_model_components,
    get_trigger,
    save_manifest,
    setup_distributed,
    setup_logger,
    should_log_step,
    should_save_final_preview,
    should_save_intermediate_image,
    smart_pool,
)

# Always prefer this repo's local `prismatic` package over any pip-installed package.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- 配置参数 ---
EPSILON = 16 / 255.0       # 扰动上界（L∞ 约束）
ITERATIONS = 500            # 优化迭代次数
BATCH_SIZE_TARGET = 8
BATCH_SIZE_SOURCE = 128
LR = 1.0 / 255.0           # PGD 步长（每步约 1 个像素值单位）
SAVE_DIR_BASE = os.environ.get("RESULTS_PG_BASE", str(REPO_ROOT / "libero_datasets_ps" / "results_pg"))
RAND_LOC = False
NUM_WORKERS = 8
PREFETCH_FACTOR = 4
PROGRESS_WINDOW = 0.12      # 允许的 source/target 相对进度差，0 表示关闭约束
MATCH_MODE = "temporal"     # temporal: 按 demo 时间进度对齐; feature: 特征最近邻匹配
TEMPORAL_DEMO_MAP = "index_mod"  # 目标 demo 序号映射到 source demo 的策略

COSINE_WEIGHT = 0.5         # 余弦相似度 loss 权重

DEBUG_SAVE_FREQ = 50
DEBUG_PRINT_FREQ = 10


def get_compute_dtype(precision: str, rank: int):
    """Pick poison-generation compute dtype for the current CUDA device."""
    precision = str(precision).strip().lower()
    if precision == "auto":
        if torch.cuda.is_available():
            try:
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
            except Exception:
                pass
        return torch.float16
    if precision in {"bf16", "bfloat16"}:
        if torch.cuda.is_available():
            try:
                if not torch.cuda.is_bf16_supported():
                    device_name = torch.cuda.get_device_name(rank)
                    raise ValueError(
                        f"Requested bf16, but CUDA device rank={rank} ({device_name}) does not support bf16. "
                        "Use --precision auto or --precision fp16 on V100/Turing GPUs."
                    )
            except ValueError:
                raise
            except Exception:
                pass
        return torch.bfloat16
    if precision in {"fp16", "float16", "half"}:
        return torch.float16
    if precision in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported --precision `{precision}`. Choose auto, bf16, fp16, or fp32.")


def _parse_temporal_stage_map(stage_map_text):
    """Parse piecewise progress anchors in format 't0:s0,t1:s1,...'."""
    if stage_map_text is None or str(stage_map_text).strip() == "":
        return [0.0, 1.0], [0.0, 1.0]

    pairs = []
    for chunk in str(stage_map_text).split(","):
        part = chunk.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(
                "Invalid --temporal_stage_map format. Expected 'target:source' pairs, "
                f"got chunk '{part}'."
            )
        t_raw, s_raw = part.split(":", 1)
        try:
            t_val = float(t_raw.strip())
            s_val = float(s_raw.strip())
        except Exception as exc:
            raise ValueError(
                "Invalid --temporal_stage_map values. Each pair must be numeric, "
                f"got '{part}'."
            ) from exc
        pairs.append((t_val, s_val))

    if len(pairs) < 2:
        raise ValueError("--temporal_stage_map must contain at least two anchor pairs.")

    target_anchors = [p[0] for p in pairs]
    source_anchors = [p[1] for p in pairs]

    for idx, (t_anchor, s_anchor) in enumerate(zip(target_anchors, source_anchors)):
        if not (0.0 <= t_anchor <= 1.0):
            raise ValueError(f"Target anchor at position {idx}={t_anchor} is out of [0,1].")
        if not (0.0 <= s_anchor <= 1.0):
            raise ValueError(f"Source anchor at position {idx}={s_anchor} is out of [0,1].")

    for idx in range(1, len(target_anchors)):
        if target_anchors[idx] <= target_anchors[idx - 1]:
            raise ValueError("Target anchors in --temporal_stage_map must be strictly increasing.")

    for idx in range(1, len(source_anchors)):
        if source_anchors[idx] <= source_anchors[idx - 1]:
            raise ValueError("Source anchors in --temporal_stage_map must be strictly increasing.")

    tol = 1e-6
    if abs(target_anchors[0]) > tol or abs(source_anchors[0]) > tol:
        raise ValueError("--temporal_stage_map must start at 0:0.")
    if abs(target_anchors[-1] - 1.0) > tol or abs(source_anchors[-1] - 1.0) > tol:
        raise ValueError("--temporal_stage_map must end at 1:1.")

    return target_anchors, source_anchors


def _piecewise_progress_map(progress, target_anchors, source_anchors):
    """Map target progress to source progress via piecewise-linear interpolation."""
    p = float(progress)
    if p <= target_anchors[0]:
        return float(source_anchors[0])
    if p >= target_anchors[-1]:
        return float(source_anchors[-1])

    for idx in range(1, len(target_anchors)):
        t0 = float(target_anchors[idx - 1])
        t1 = float(target_anchors[idx])
        if p <= t1:
            s0 = float(source_anchors[idx - 1])
            s1 = float(source_anchors[idx])
            alpha = (p - t0) / max(t1 - t0, 1e-12)
            alpha = max(0.0, min(1.0, alpha))
            return s0 + alpha * (s1 - s0)

    return float(source_anchors[-1])


def _parse_demo_index(demo_key):
    key = str(demo_key)
    if key.startswith("demo_"):
        try:
            return int(key.split("_", 1)[1])
        except Exception:
            return None
    return None


def _find_free_master_port():
    """Find an available localhost TCP port for torch.distributed rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main_worker(rank, world_size, args):
    distributed = world_size > 1
    if distributed:
        setup_distributed(rank, world_size, master_port=args.master_port)
    else:
        torch.cuda.set_device(rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    
    local_iterations = ITERATIONS
    if args.debug:
        local_iterations = 50
    
    log_root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    logger = setup_logger(rank, log_root_dir)
    
    if args.debug and rank == 0:
        logger.info("=" * 40)
        logger.info("!!! DEBUG MODE !!!")
        logger.info(f"Iterations reduced to {local_iterations}")
        logger.info("=" * 40)

    debug_img_dir = os.path.join(args.save_dir, "debug_images")
    if rank == 0:
        os.makedirs(debug_img_dir, exist_ok=True)
    if distributed:
        dist.barrier()

    compute_dtype = get_compute_dtype(args.precision, rank)
    if rank == 0:
        logger.info(f"Using poison-generation compute dtype: {compute_dtype} (--precision={args.precision})")

    # 加载模型
    if rank == 0:
        logger.info(f"Loading OpenVLA ({args.model_id})...")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_id,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map={"": rank}
    )
    model.eval()
    
    projector, vision_tower = get_model_components(model, logger if rank == 0 else None)
    for p in model.parameters():
        p.requires_grad = False
    
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    try:
        image_mean = processor.image_processor.image_mean
        image_std = processor.image_processor.image_std
    except:
        image_mean = [0.5, 0.5, 0.5]
        image_std = [0.5, 0.5, 0.5]

    normalizer = transforms.Normalize(mean=image_mean, std=image_std)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # 数据集
    ds_target = IndexedDataset(LiberoWrapper(args.data_root, args.target_task, transform=transform, image_key=args.image_key))
    ds_source = LiberoWrapper(args.data_root, args.source_task, transform=transform, image_key=args.image_key)

    if rank == 0:
        save_manifest(args, ds_target.base_ds.file_path, iterations=local_iterations, epsilon=EPSILON)
    
    # 数据加载器
    sampler = DistributedSampler(ds_target, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None
    loader_target = build_dataloader(
        ds_target,
        batch_size=args.batch_size_target,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=False,
        prefetch_factor=args.prefetch_factor,
    )
    source_bank_loader = None
    if args.match_mode == "feature":
        source_bank_loader = build_dataloader(
            ds_source,
            batch_size=args.batch_size_source,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            prefetch_factor=args.prefetch_factor,
        )

    if rank == 0:
        logger.info(
            f"Throughput config | world_size={world_size} | "
            f"target_bs/gpu={args.batch_size_target} | source_bs/gpu={args.batch_size_source} | "
            f"workers={args.num_workers} | prefetch={args.prefetch_factor}"
        )

    source_feat_bank = None
    source_idx_bank = None
    source_progress_bank = None
    source_demo_keys = [str(k) for k in ds_source.demo_keys]
    target_demo_keys = [str(k) for k in ds_target.base_ds.demo_keys]
    source_demo_start_by_name = {}
    cursor = 0
    for demo_key in ds_source.demo_keys:
        demo_key_str = str(demo_key)
        source_demo_start_by_name[demo_key_str] = cursor
        cursor += int(ds_source.demo_lengths[demo_key])
    target_demo_pos_by_name = {str(k): idx for idx, k in enumerate(ds_target.base_ds.demo_keys)}

    if args.match_mode == "feature":
        # Build a full-source feature bank once so each target frame matches against all source frames.
        if rank == 0:
            logger.info("Matching mode=feature | Building global source feature bank...")
        source_feat_chunks = []
        source_idx_chunks = []
        source_progress_chunks = []
        source_iterator = tqdm(source_bank_loader, desc=f"GPU {rank} source bank", position=rank) if rank == 0 else source_bank_loader
        with torch.no_grad():
            for source_batch in source_iterator:
                source_imgs = source_batch["image"].to(rank, non_blocking=True).to(compute_dtype)
                source_in = torch.cat([normalizer(source_imgs)] * 2, dim=1)
                source_feat_raw = extract_features(vision_tower(source_in))
                source_feat_pool = ensure_2d(smart_pool(source_feat_raw))
                source_feat_chunks.append(source_feat_pool.detach().cpu())

                source_indices = source_batch.get("index", None)
                if source_indices is None:
                    raise KeyError("Source dataset must provide flat index for nearest-neighbor matching.")
                source_idx_chunks.append(source_indices.detach().cpu().long())

                source_progress = source_batch.get("progress", None)
                if source_progress is None:
                    source_frame_idx = source_batch.get("frame_idx", None)
                    source_demo_len = source_batch.get("demo_len", None)
                    if source_frame_idx is None or source_demo_len is None:
                        raise KeyError("Source batch must provide progress or (frame_idx, demo_len).")
                    source_progress = source_frame_idx.float() / torch.clamp(source_demo_len.float() - 1.0, min=1.0)
                source_progress_chunks.append(source_progress.detach().cpu().float())

        if len(source_feat_chunks) == 0:
            raise RuntimeError("Source feature bank is empty. Check source dataset/path settings.")

        source_feat_bank = torch.cat(source_feat_chunks, dim=0).to(rank, non_blocking=True)
        source_idx_bank = torch.cat(source_idx_chunks, dim=0).long()
        source_progress_bank = torch.cat(source_progress_chunks, dim=0).to(rank, non_blocking=True)
        if rank == 0:
            logger.info(
                f"Source bank ready | entries={source_feat_bank.shape[0]} | feat_dim={source_feat_bank.shape[1]}"
            )
    else:
        if len(source_demo_keys) == 0:
            raise RuntimeError("Source dataset contains zero demos; temporal matching cannot proceed.")
        if rank == 0:
            logger.info(
                f"Matching mode=temporal | demo_map={args.temporal_demo_map} | "
                f"source_demos={len(source_demo_keys)} | target_demos={len(target_demo_keys)}"
            )
            logger.info(
                "Temporal stage map active | "
                f"target_anchors={args.temporal_target_anchors} | "
                f"source_anchors={args.temporal_source_anchors}"
            )
    
    iterator = tqdm(loader_target, desc=f"GPU {rank}", position=rank) if rank == 0 else loader_target
    
    for i, batch_target in enumerate(iterator):
        imgs_target = batch_target['image'].to(rank, non_blocking=True).to(compute_dtype)
        target_indices = batch_target.get('index', None)
        
        # --- 阶段 A: 匹配 source 帧 (feature 最近邻 或 temporal 时间对齐) ---
        # 这里不能使用 inference_mode，因为后续 loss 会把 target_embeddings 作为
        # 非训练目标参与反向传播，inference tensor 会触发 backward 保存错误。
        with torch.no_grad():
            target_progress = batch_target.get("progress", None)
            if target_progress is None:
                target_frame_idx = batch_target.get("frame_idx", None)
                target_demo_len = batch_target.get("demo_len", None)
                if target_frame_idx is None or target_demo_len is None:
                    raise KeyError("Target batch must provide progress or (frame_idx, demo_len).")
                target_progress = target_frame_idx.float() / torch.clamp(target_demo_len.float() - 1.0, min=1.0)
            target_progress = target_progress.to(rank, non_blocking=True).float()

            if args.match_mode == "feature":
                trg_in_clean = torch.cat([normalizer(imgs_target)] * 2, dim=1)
                trg_feat_raw = extract_features(vision_tower(trg_in_clean))

                log_tag = "Target" if (i == 0 and rank == 0) else ""
                trg_feat_pool = ensure_2d(smart_pool(trg_feat_raw, logger, log_tag))

                dist_mat = torch.cdist(trg_feat_pool.float(), source_feat_bank.float())
                if args.progress_window > 0.0:
                    progress_diff = (target_progress.unsqueeze(1) - source_progress_bank.unsqueeze(0)).abs()
                    valid_mask = progress_diff <= args.progress_window
                    masked_dist = dist_mat.masked_fill(~valid_mask, float("inf"))
                    best_source_positions = torch.argmin(masked_dist, dim=1)

                    no_candidate = ~valid_mask.any(dim=1)
                    if no_candidate.any():
                        fallback_positions = torch.argmin(dist_mat[no_candidate], dim=1)
                        best_source_positions[no_candidate] = fallback_positions

                    if rank == 0 and i == 0:
                        avg_candidates = valid_mask.float().sum(dim=1).mean().item()
                        fallback_count = int(no_candidate.sum().item())
                        logger.info(
                            f"Progress gating | window={args.progress_window:.3f} | "
                            f"avg_candidates={avg_candidates:.1f} | fallback={fallback_count}/{no_candidate.numel()}"
                        )
                else:
                    best_source_positions = torch.argmin(dist_mat, dim=1)
                    if rank == 0 and i == 0:
                        logger.info("Progress gating disabled (progress_window <= 0).")
                best_source_indices = source_idx_bank[best_source_positions.detach().cpu()].tolist()
            else:
                # Temporal matching: align source frame by target progress within mapped demo.
                target_demo_batch = batch_target.get("demo_key", None)
                target_progress_list = target_progress.detach().cpu().tolist()

                if target_demo_batch is None:
                    if target_indices is None:
                        raise KeyError("Temporal matching requires demo_key or index in target batch.")
                    target_demo_batch = [
                        ds_target.base_ds.indices[int(idx)][0]
                        for idx in target_indices.detach().cpu().tolist()
                    ]
                elif isinstance(target_demo_batch, str):
                    target_demo_batch = [target_demo_batch for _ in range(len(target_progress_list))]

                best_source_indices = []
                for b_idx, progress in enumerate(target_progress_list):
                    target_demo_key = str(target_demo_batch[b_idx])
                    target_demo_pos = target_demo_pos_by_name.get(target_demo_key, None)
                    if target_demo_pos is None:
                        parsed_idx = _parse_demo_index(target_demo_key)
                        target_demo_pos = 0 if parsed_idx is None else parsed_idx

                    if args.temporal_demo_map == "index_mod":
                        source_demo_key = source_demo_keys[target_demo_pos % len(source_demo_keys)]
                    else:
                        raise ValueError(f"Unsupported temporal_demo_map: {args.temporal_demo_map}")

                    source_len = int(ds_source.demo_lengths[source_demo_key])
                    max_source_frame = max(source_len - 1, 0)
                    mapped_progress = _piecewise_progress_map(
                        progress,
                        args.temporal_target_anchors,
                        args.temporal_source_anchors,
                    )
                    source_frame = int(round(float(mapped_progress) * float(max_source_frame)))
                    source_frame = min(max(source_frame, 0), max_source_frame)
                    source_start = int(source_demo_start_by_name[source_demo_key])
                    best_source_indices.append(source_start + source_frame)

                if rank == 0 and i == 0:
                    logger.info(
                        f"Temporal matching active | map={args.temporal_demo_map} | "
                        f"batch_size={len(best_source_indices)}"
                    )

            matched_src_list = []
            for source_idx in best_source_indices:
                matched_src_list.append(ds_source[int(source_idx)]["image"])
            matched_src_imgs = torch.stack(matched_src_list, dim=0).to(rank, non_blocking=True).to(compute_dtype)

            # 贴上 trigger
            trigger = get_trigger(args.patch_size, rank).to(compute_dtype)
            matched_src_imgs_triggered = apply_trigger(
                matched_src_imgs,
                trigger,
                args.patch_size,
                rand_loc=args.rand_loc,
            )
            
            # 目标 embedding (triggered source 通过 projector 后的输出)
            src_triggered_in = torch.cat([normalizer(matched_src_imgs_triggered)] * 2, dim=1)
            src_triggered_feat = extract_features(vision_tower(src_triggered_in))
            target_embeddings = projector(src_triggered_feat).detach()
        
        # --- 阶段 B: PGD 优化 delta ---
        delta = torch.zeros_like(imgs_target, dtype=torch.float32, device=rank)
        delta.requires_grad_(True)
        
        best_loss = float('inf')
        best_delta = delta.data.clone()
        
        for step in range(local_iterations):
            # 构造毒药图像
            poison_img = (imgs_target.float() + delta).clamp(0.0, 1.0)
            poison_in = poison_img.to(compute_dtype)
            poison_in_norm = torch.cat([normalizer(poison_in)] * 2, dim=1)
            
            # 前向
            trg_feat = extract_features(vision_tower(poison_in_norm))
            poison_embeddings = projector(trg_feat)
            
            # === Loss 1: Feature Collision (MSE) ===
            loss_mse = nn.MSELoss()(poison_embeddings, target_embeddings)
            
            # === Loss 2: Cosine Similarity ===
            p_flat = poison_embeddings.reshape(poison_embeddings.shape[0], -1)
            t_flat = target_embeddings.reshape(target_embeddings.shape[0], -1)
            cos_sim = nn.CosineSimilarity(dim=1)(p_flat, t_flat).mean()
            loss_cos = 1.0 - cos_sim
            
            # === 总 Loss ===
            total_loss = loss_mse + COSINE_WEIGHT * loss_cos
            
            # PGD 更新 delta
            if delta.grad is not None:
                delta.grad.zero_()
            total_loss.backward()
            
            with torch.no_grad():
                # PGD 步骤
                grad_sign = delta.grad.sign()
                delta.data = delta.data - LR * grad_sign  # 沿梯度负方向走
                
                # 投影到 L∞ ball
                delta.data = delta.data.clamp(-EPSILON, EPSILON)
                
                # 确保最终图像在 [0, 1]
                delta.data = torch.max(delta.data, -imgs_target.float())
                delta.data = torch.min(delta.data, 1.0 - imgs_target.float())
            
            # 记录最佳
            if loss_mse.item() < best_loss:
                best_loss = loss_mse.item()
                best_delta = delta.data.clone()
            
            # 日志
            if rank == 0:
                if should_log_step(args, i, step, local_iterations):
                    logger.info(
                        f"[Batch {i}][Step {step}] "
                        f"Total: {total_loss.item():.4f} | MSE: {loss_mse.item():.6f} | "
                        f"Cos: {cos_sim.item():.4f} | "
                        f"δ_norm: {delta.data.abs().mean():.6f}"
                    )
                
                if should_save_intermediate_image(args, step, local_iterations):
                    with torch.no_grad():
                        final_poison = (imgs_target.float() + delta.data).clamp(0.0, 1.0)
                        debug_grid = torch.cat([
                            imgs_target[0:1].float(),
                            final_poison[0:1],
                            matched_src_imgs_triggered[0:1].float()
                        ], dim=0)
                        save_path = os.path.join(debug_img_dir, f"batch{i}_step{step}.png")
                        save_image(debug_grid, save_path)
        
        with torch.no_grad():
            final_poison = (imgs_target.float() + best_delta).clamp(0.0, 1.0)
            if rank == 0 and should_save_final_preview(args, i):
                preview_grid = torch.cat([
                    imgs_target[0:1].float().detach().cpu(),
                    final_poison[0:1].float().detach().cpu(),
                    matched_src_imgs_triggered[0:1].float().detach().cpu(),
                ], dim=0)
                preview_path = os.path.join(debug_img_dir, f"batch{i}_final.png")
                save_image(preview_grid, preview_path)
            final_poison = final_poison.to(compute_dtype).detach().cpu()
        
        save_name = f"poison_rank{rank}_batch{i}.pt"
        torch.save(final_poison, os.path.join(args.save_dir, save_name))
        
        if target_indices is not None:
            idx_name = f"poison_rank{rank}_batch{i}_idx.pt"
            torch.save(target_indices.detach().cpu().long(), os.path.join(args.save_dir, idx_name))
        
        if args.debug:
            if rank == 0:
                logger.info("[Debug] Single batch complete. Exiting.")
            break

    if distributed:
        cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default=None, help="GPU IDs, e.g. '0,1'")
    parser.add_argument("--rand_loc", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Enable fast verification mode")
    parser.add_argument("--iterations", type=int, default=None, help="Override iteration count")
    parser.add_argument("--epsilon", type=float, default=None, help="Override epsilon (in [0,1], default 16/255)")
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
        help="Compute dtype for poison generation. auto uses bf16 on A100+/bf16-capable GPUs and fp16 on V100.",
    )
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID, help="OpenVLA model path or HF model id")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT, help="LIBERO hdf5 dataset root")
    parser.add_argument("--source_task", type=str, default=DEFAULT_SOURCE_TASK, help="Source task name without .hdf5 suffix")
    parser.add_argument("--target_task", type=str, default=DEFAULT_TARGET_TASK, help="Target task name without .hdf5 suffix")
    parser.add_argument("--image_key", type=str, default=None, help="HDF5 obs image key to optimize, e.g. agentview_rgb or eye_in_hand_rgb")
    parser.add_argument("--patch_size", type=int, default=DEFAULT_PATCH_SIZE, help="Trigger patch size")
    parser.add_argument("--batch_size_target", type=int, default=BATCH_SIZE_TARGET, help="Per-GPU target batch size")
    parser.add_argument("--batch_size_source", type=int, default=BATCH_SIZE_SOURCE, help="Per-GPU source batch size")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS, help="DataLoader workers per process")
    parser.add_argument("--prefetch_factor", type=int, default=PREFETCH_FACTOR, help="DataLoader prefetch factor per worker")
    parser.add_argument("--match_mode", type=str, default=MATCH_MODE, choices=["feature", "temporal"], help="How to pair source frames: feature NN or temporal alignment")
    parser.add_argument("--temporal_demo_map", type=str, default=TEMPORAL_DEMO_MAP, choices=["index_mod"], help="How target demo index maps to source demo when match_mode=temporal")
    parser.add_argument(
        "--temporal_stage_map",
        type=str,
        default="0:0,1:1",
        help=(
            "Piecewise target->source progress anchors for temporal matching, "
            "format: 't0:s0,t1:s1,...' within [0,1], strictly increasing, must include 0:0 and 1:1. "
            "Example: '0:0,0.3:0.55,0.75:0.9,1:1'"
        ),
    )
    parser.add_argument("--progress_window", type=float, default=PROGRESS_WINDOW, help="Max |source_progress-target_progress| for nearest-neighbor candidates; set 0 to disable")
    parser.add_argument("--fast_mode", action="store_true", help="Reduce logging and only save sparse final previews for higher throughput")
    parser.add_argument("--fast_log_batch_freq", type=int, default=FAST_LOG_BATCH_FREQ, help="In fast mode, log every N batches")
    parser.add_argument("--fast_image_batch_freq", type=int, default=FAST_IMAGE_BATCH_FREQ, help="In fast mode, save one final preview image every N batches")
    parser.add_argument(
        "--master_port",
        type=str,
        default="auto",
        help="torch.distributed rendezvous port. Use 'auto' to pick a free local port.",
    )
    parser.add_argument(
        "--save_name_prefix",
        type=str,
        default="",
        help="Optional output folder prefix. If set to spatial/object/goal, output dir becomes '<prefix>_<timestamp>'.",
    )
    args = parser.parse_args()

    args.temporal_target_anchors, args.temporal_source_anchors = _parse_temporal_stage_map(
        args.temporal_stage_map
    )

    RAND_LOC = bool(args.rand_loc)
    if args.iterations is not None:
        ITERATIONS = args.iterations
    if args.epsilon is not None:
        EPSILON = args.epsilon
    if args.prefetch_factor < 1:
        raise ValueError("prefetch_factor must be >= 1")
    if not 0.0 <= args.progress_window <= 1.0:
        raise ValueError("progress_window must be in [0, 1]")
    if args.fast_log_batch_freq < 1:
        raise ValueError("fast_log_batch_freq must be >= 1")
    if args.fast_image_batch_freq < 1:
        raise ValueError("fast_image_batch_freq must be >= 1")
    if str(args.master_port).strip().lower() == "auto":
        args.master_port = _find_free_master_port()
    else:
        try:
            args.master_port = int(str(args.master_port).strip())
        except Exception as exc:
            raise ValueError("master_port must be an integer or 'auto'.") from exc
        if not (1 <= args.master_port <= 65535):
            raise ValueError("master_port must be in [1, 65535].")

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = str(args.save_name_prefix).strip()
    if prefix:
        safe_prefix = "".join(ch if (ch.isalnum() or ch in ["_", "-"]) else "_" for ch in prefix)
        safe_prefix = safe_prefix.strip("_")
        folder_name = f"{safe_prefix}_{timestamp}" if safe_prefix else timestamp
    else:
        folder_name = timestamp
    if args.debug:
        folder_name = f"{folder_name}_debug"
    args.save_dir = os.path.join(SAVE_DIR_BASE, folder_name)
    print(f"Output dir: {args.save_dir}")
    print(f"Distributed master port: {args.master_port}")

    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise ValueError("No GPUs!")
    
    if world_size == 1:
        print("Running single-process single-GPU mode (NCCL disabled).")
        main_worker(0, world_size, args)
    else:
        print(f"Spawning {world_size} processes...")
        mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))
