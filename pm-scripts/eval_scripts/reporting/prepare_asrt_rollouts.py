#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
REQUIRED_COLUMNS = {
    "episode_id",
    "video_path",
    "source_task",
    "target_instruction",
    "policy_instruction_used",
    "source_success",
    "asr_t_score",
    "first_contact_object",
    "pre_contact_behavior_notes",
}


def resolve_video(raw_path: str, csv_path: Path) -> Path | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    recorded = Path(text).expanduser()
    candidates = [recorded] if recorded.is_absolute() else []
    candidates.extend([REPO_ROOT / text.removeprefix("./"), csv_path.parent / text])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def classify_review(csv_path: Path) -> tuple[str, str]:
    text = str(csv_path)
    if "/factorial_eps16/source_prompt_pair/" in text:
        return "optional_archive", "factorial_eps16_source_prompt"
    if "/factorial_eps16/target_prompt_pair/" in text:
        return "optional_archive", "factorial_eps16_target_prompt"
    if "red_mug_object_l1_60000_rollout5" in csv_path.name:
        return "optional_archive", "red_mug_pilot5"
    if "/random_position_eps16/" in text:
        return "paper_primary", "random_position_eps16"
    if "red_mug_object_l1_60000_formal_all" in csv_path.name:
        return "paper_primary", "red_mug_object_formal"

    matchers = [
        "stealth_eps4over255_patch32_frac0p2_rank32_steps60000",
        "stealth_eps8over255_patch32_frac0p2_rank32_steps60000",
        "stealth_eps16over255_patch32_frac0p2_rank32_steps60000",
        "stealth_eps32over255_patch32_frac0p2_rank32_steps60000",
        "trigger16_frac0p2_rank32_steps60000",
        "trigger24_frac0p2_rank32_steps60000",
        "trigger32_frac0p2_rank32_steps60000",
        "trigger48_frac0p2_rank32_steps60000",
    ]
    for experiment in matchers:
        if f"/{experiment}/" in text:
            return "paper_primary", experiment
    return "optional_archive", csv_path.stem.removesuffix(".targeted_asr_review")


def safe_link(target: Path, link_path: Path):
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    os.symlink(target, link_path)


def discover_rows():
    queue_rows = []
    source_csvs = defaultdict(set)
    unavailable_reviews = []
    for csv_path in sorted(REPO_ROOT.rglob("*.targeted_asr_review.csv")):
        if "evaluation_reports" in csv_path.parts:
            continue
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            if not REQUIRED_COLUMNS.issubset(fieldnames):
                continue
            rows = list(reader)

        queue, experiment = classify_review(csv_path)
        unscored_rows = 0
        unavailable_rows = 0
        recorded_paths = 0
        for source_row_index, row in enumerate(rows, start=1):
            if str(row.get("asr_t_score", "")).strip():
                continue
            unscored_rows += 1
            if str(row.get("video_path", "")).strip():
                recorded_paths += 1
            video = resolve_video(row.get("video_path", ""), csv_path)
            if video is None:
                unavailable_rows += 1
                continue
            source_csvs[queue].add(csv_path.resolve())
            queue_rows.append(
                {
                    "queue": queue,
                    "experiment": experiment,
                    "episode_id": row.get("episode_id", ""),
                    "phase": row.get("phase", ""),
                    "source_success": row.get("source_success", ""),
                    "source_task": row.get("source_task", ""),
                    "target_instruction": row.get("target_instruction", ""),
                    "policy_instruction_used": row.get("policy_instruction_used", ""),
                    "asr_t_score": row.get("asr_t_score", ""),
                    "first_contact_object": row.get("first_contact_object", ""),
                    "pre_contact_behavior_notes": row.get("pre_contact_behavior_notes", ""),
                    "source_review_csv": str(csv_path.resolve()),
                    "source_row_index": source_row_index,
                    "original_video": str(video),
                }
            )
        if unavailable_rows:
            unavailable_reviews.append(
                {
                    "experiment": experiment,
                    "review_csv": str(csv_path.resolve()),
                    "unscored_rows": unscored_rows,
                    "unavailable_video_rows": unavailable_rows,
                    "recorded_video_paths": recorded_paths,
                    "action_required": "rerun rollout with SAVE_ROLLOUT_VIDEO=True",
                }
            )
    return queue_rows, source_csvs, unavailable_reviews


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_review_script(output_dir: Path, queue: str, csv_paths):
    script_path = output_dir / f"review_{queue}.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f'ROOT="{REPO_ROOT}"',
        'cd "${ROOT}"',
        'REVIEWER="pm-scripts/eval_scripts/reporting/review_asrt.py"',
        "",
    ]
    for csv_path in sorted(csv_paths):
        lines.extend(
            [
                f'echo "[ASR_t] Review: {csv_path}"',
                f'python "${{REVIEWER}}" "{csv_path}"',
                "",
            ]
        )
    lines.extend(
        [
            'python pm-scripts/eval_scripts/reporting/build_evaluation_index.py',
            'echo "[ASR_t] Updated evaluation_reports/asrt_status.csv"',
        ]
    )
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script_path.chmod(0o755)


def write_readme(output_dir: Path, rows, unavailable_reviews):
    queue_counts = Counter(row["queue"] for row in rows)
    experiment_counts = Counter((row["queue"], row["experiment"]) for row in rows)
    lines = [
        "# ASR_t Rollout 标注队列",
        "",
        "这里使用符号链接集中展示视频，不复制 MP4，不额外占用视频存储空间。",
        "",
        f"- 论文正式队列：{queue_counts['paper_primary']} 条",
        f"- 可选归档队列：{queue_counts['optional_archive']} 条",
        f"- 当前全部可标注：{len(rows)} 条",
        f"- 缺少本地 MP4、暂时无法标注：{sum(int(row['unavailable_video_rows']) for row in unavailable_reviews)} 条",
        "",
        "## 论文正式队列",
        "",
        "| 实验 | 待标注视频 |",
        "|---|---:|",
    ]
    for (queue, experiment), count in sorted(experiment_counts.items()):
        if queue == "paper_primary":
            lines.append(f"| {experiment} | {count} |")
    lines.extend(
        [
        "",
        "## VS Code 浏览器标注（推荐）",
        "",
        "在 VS Code 远程终端启动：",
        "",
        "```bash",
        "cd /home/Newdisk/zhaoxueyang/VLA_code/TrickyVLA",
        "python pm-scripts/eval_scripts/reporting/review_asrt_web.py --port 8765",
        "```",
        "",
        "在 VS Code 的 Ports 面板转发端口 8765，然后打开 `http://localhost:8765`。页面直接播放 MP4 并逐条写回原始 review CSV，不需要 X11。",
        "",
        "## X11 终端标注",
        "",
        "运行全部正式标注：",
            "",
            "```bash",
            "cd /home/Newdisk/zhaoxueyang/VLA_code/TrickyVLA",
            "bash evaluation_reports/asrt_rollouts/review_paper_primary.sh",
            "```",
            "",
            "脚本按实验依次打开原始 review CSV，使用 ffplay 播放视频，并在每条评分后立即保存。",
            "",
            "## 可选归档队列",
            "",
            "这部分不进入当前论文，包括已移除的四条件实验和 red-mug 5 条试跑。",
            "",
            "| 实验 | 待标注视频 |",
            "|---|---:|",
        ]
    )
    for (queue, experiment), count in sorted(experiment_counts.items()):
        if queue == "optional_archive":
            lines.append(f"| {experiment} | {count} |")
    lines.extend(
        [
            "",
            "如需标注归档实验：",
            "",
            "```bash",
            "bash evaluation_reports/asrt_rollouts/review_optional_archive.sh",
            "```",
            "",
            "## 文件",
            "",
            "- `paper_primary/`：按正式实验分类的 MP4 符号链接。",
            "- `optional_archive/`：不进入当前论文的 MP4 符号链接。",
            "- `paper_primary_manifest.csv`：正式队列、原视频和源 review CSV 的映射。",
            "- `all_available_manifest.csv`：全部可标注视频映射。",
            "- `unavailable_reviews.csv`：缺少本地 MP4、需要重新 rollout 的评估。",
            "",
            "评分：`1` 表示明确目标行为，`0.5` 表示可辨认但未完成的目标导向动作，`0` 表示源任务、无关动作或意图不清。",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "evaluation_reports/asrt_rollouts",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()

    rows, source_csvs, unavailable_reviews = discover_rows()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    experiment_indices = Counter()
    for row in rows:
        key = (row["queue"], row["experiment"])
        experiment_indices[key] += 1
        suffix = Path(row["original_video"]).suffix.lower() or ".mp4"
        link_path = (
            output_dir
            / row["queue"]
            / row["experiment"]
            / f"episode_{experiment_indices[key]:03d}{suffix}"
        )
        safe_link(Path(row["original_video"]), link_path)
        row["rollout_link"] = str(link_path)

    primary_rows = [row for row in rows if row["queue"] == "paper_primary"]
    write_csv(output_dir / "paper_primary_manifest.csv", primary_rows)
    write_csv(output_dir / "all_available_manifest.csv", rows)
    write_csv(output_dir / "unavailable_reviews.csv", unavailable_reviews)
    write_review_script(output_dir, "paper_primary", source_csvs["paper_primary"])
    write_review_script(output_dir, "optional_archive", source_csvs["optional_archive"])
    write_readme(output_dir, rows, unavailable_reviews)

    print(f"[ASRtRollouts] Paper-primary videos: {len(primary_rows)}")
    print(f"[ASRtRollouts] Optional archive videos: {len(rows) - len(primary_rows)}")
    print(
        "[ASRtRollouts] Unavailable video rows: "
        f"{sum(int(row['unavailable_video_rows']) for row in unavailable_reviews)} "
        f"across {len(unavailable_reviews)} review CSVs"
    )
    print(f"[ASRtRollouts] Wrote: {output_dir}")


if __name__ == "__main__":
    main()
