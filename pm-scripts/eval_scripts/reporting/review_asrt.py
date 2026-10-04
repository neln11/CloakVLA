#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_video(raw_path: str, csv_path: Path) -> Path | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    recorded = Path(text).expanduser()
    candidates = [recorded] if recorded.is_absolute() else []
    candidates.extend([REPO_ROOT / text.removeprefix("./"), csv_path.parent / text])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def save_rows(csv_path: Path, fieldnames, rows):
    fd, tmp_name = tempfile.mkstemp(prefix=csv_path.name, suffix=".tmp", dir=csv_path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        tmp_path.replace(csv_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def play_video(player: str, video_path: Path):
    env = os.environ.copy()
    if player == "ffplay":
        if env.get("DISPLAY"):
            env["SDL_VIDEODRIVER"] = "x11"
        result = subprocess.run(
            [player, "-autoexit", "-loglevel", "error", str(video_path)],
            check=False,
            env=env,
        )
    else:
        result = subprocess.run([player, str(video_path)], check=False, env=env)
    return result.returncode == 0


def check_x11():
    display = os.environ.get("DISPLAY", "").strip()
    if not display:
        raise RuntimeError(
            "DISPLAY is empty. Reconnect with SSH X11 forwarding, for example "
            "`ssh -Y user@host`, then verify with `xdpyinfo`."
        )
    if shutil.which("xdpyinfo"):
        result = subprocess.run(
            ["xdpyinfo"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or "xdpyinfo could not open the display"
            raise RuntimeError(f"X11 display {display!r} is unavailable: {detail}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("review_csv", type=Path)
    parser.add_argument("--player", default="ffplay")
    parser.add_argument("--include-scored", action="store_true")
    parser.add_argument("--no-player", action="store_true")
    args = parser.parse_args()

    csv_path = args.review_csv.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not args.no_player and shutil.which(args.player) is None:
        raise FileNotFoundError(f"Video player not found: {args.player}")
    if not args.no_player and args.player == "ffplay":
        check_x11()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    for required in ["asr_t_score", "first_contact_object", "pre_contact_behavior_notes"]:
        if required not in fieldnames:
            raise ValueError(f"Missing required column: {required}")

    for index, row in enumerate(rows):
        if str(row.get("asr_t_score", "")).strip() and not args.include_scored:
            continue
        video = resolve_video(row.get("video_path", ""), csv_path)
        print("\n" + "=" * 80)
        print(f"Row {index + 1}/{len(rows)} | episode={row.get('episode_id', '')}")
        print(f"Source: {row.get('source_task', '')}")
        print(f"Target: {row.get('target_instruction', '')}")
        print(f"Prompt: {row.get('policy_instruction_used', '')}")
        print(f"Video : {video or 'MISSING'}")
        if video is None:
            continue
        if not args.no_player:
            if not play_video(args.player, video):
                raise RuntimeError(
                    "Video player failed. Check DISPLAY with `echo $DISPLAY` "
                    "and test X11 with `xdpyinfo`."
                )

        while True:
            score = input("ASR_t score [0/0.5/1, r=replay, s=skip, q=quit]: ").strip().lower()
            if score == "r":
                if not args.no_player:
                    if not play_video(args.player, video):
                        raise RuntimeError("Video replay failed; check the X11 connection.")
                continue
            if score == "s":
                break
            if score == "q":
                save_rows(csv_path, fieldnames, rows)
                print(f"Saved: {csv_path}")
                return
            if score not in {"0", "0.5", "1"}:
                print("Enter 0, 0.5, 1, r, s, or q.")
                continue
            row["asr_t_score"] = score
            row["first_contact_object"] = input("First contact object: ").strip()
            row["pre_contact_behavior_notes"] = input("Pre-contact evidence/notes: ").strip()
            save_rows(csv_path, fieldnames, rows)
            print(f"Saved row {index + 1}: score={score}")
            break

    save_rows(csv_path, fieldnames, rows)
    print(f"Review complete or no remaining rows: {csv_path}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
