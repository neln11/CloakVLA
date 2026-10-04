#!/usr/bin/env python3
import argparse
import csv
import json
import mimetypes
import os
import re
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "evaluation_reports/asrt_rollouts/paper_primary_manifest.csv"
WRITE_LOCK = threading.Lock()

HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ASR_t Rollout Review</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #17212b;
      --muted: #66727f;
      --line: #d8dee5;
      --panel: #f5f7f9;
      --accent: #1769aa;
      --ok: #18784c;
      --warn: #9a5b00;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: #fff;
      font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      height: 54px;
      display: flex;
      align-items: center;
      gap: 18px;
      padding: 0 20px;
      border-bottom: 1px solid var(--line);
      background: #fff;
      position: sticky;
      top: 0;
      z-index: 2;
    }
    h1 { margin: 0; font-size: 18px; font-weight: 650; }
    .summary { color: var(--muted); }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      min-height: calc(100vh - 54px);
    }
    .viewer { padding: 16px 20px 24px; min-width: 0; }
    .video-wrap {
      width: 100%;
      min-height: 360px;
      height: calc(100vh - 212px);
      display: grid;
      place-items: center;
      background: #101418;
      border: 1px solid #101418;
      border-radius: 6px;
      overflow: hidden;
    }
    video { width: 100%; height: 100%; object-fit: contain; }
    .nav {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 12px;
    }
    button, select, input, textarea {
      font: inherit;
    }
    button {
      min-height: 34px;
      border: 1px solid #b8c1cb;
      border-radius: 5px;
      padding: 6px 12px;
      background: #fff;
      color: var(--ink);
      cursor: pointer;
    }
    button:hover { border-color: #7d8996; background: #f7f9fb; }
    button:disabled { opacity: .45; cursor: default; }
    .primary { background: var(--accent); color: #fff; border-color: var(--accent); }
    .primary:hover { background: #105b96; border-color: #105b96; }
    .spacer { flex: 1; }
    .position { color: var(--muted); font-variant-numeric: tabular-nums; }
    aside {
      border-left: 1px solid var(--line);
      background: var(--panel);
      padding: 16px;
      overflow-y: auto;
    }
    label { display: block; color: var(--muted); font-size: 12px; margin-bottom: 5px; }
    select, input, textarea {
      width: 100%;
      border: 1px solid #b8c1cb;
      border-radius: 5px;
      background: #fff;
      color: var(--ink);
      padding: 7px 9px;
    }
    textarea { min-height: 76px; resize: vertical; }
    .field { margin-bottom: 14px; }
    .meta {
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      padding: 12px 0;
      margin: 12px 0 14px;
    }
    .meta-row { margin-bottom: 9px; }
    .meta-row:last-child { margin-bottom: 0; }
    .meta-key { color: var(--muted); font-size: 12px; }
    .meta-value { overflow-wrap: anywhere; }
    .scores { display: grid; grid-template-columns: repeat(3, 1fr); gap: 7px; }
    .score.active {
      color: #fff;
      background: var(--accent);
      border-color: var(--accent);
    }
    .save-row { display: grid; grid-template-columns: 1fr 2fr; gap: 8px; }
    .status { min-height: 20px; margin-top: 10px; color: var(--ok); }
    .status.error { color: #b3261e; }
    .checkbox {
      display: flex;
      align-items: center;
      gap: 7px;
      color: var(--ink);
      font-size: 13px;
      margin: 9px 0 0;
    }
    .checkbox input { width: auto; }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      aside { border-left: 0; border-top: 1px solid var(--line); }
      .video-wrap { height: 55vh; min-height: 280px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>ASR_t Rollout Review</h1>
    <div class="summary" id="summary">加载中</div>
  </header>
  <main>
    <section class="viewer">
      <div class="video-wrap">
        <video id="video" controls autoplay preload="metadata"></video>
      </div>
      <div class="nav">
        <button id="prev" title="上一条">上一条</button>
        <button id="next" title="下一条">下一条</button>
        <button id="replay" title="从头播放">重播</button>
        <span class="spacer"></span>
        <span class="position" id="position"></span>
      </div>
    </section>
    <aside>
      <div class="field">
        <label for="experiment">实验</label>
        <select id="experiment"></select>
        <label class="checkbox">
          <input type="checkbox" id="unscoredOnly" checked>
          仅显示未标注
        </label>
      </div>
      <div class="meta">
        <div class="meta-row">
          <div class="meta-key">Episode</div>
          <div class="meta-value" id="episode"></div>
        </div>
        <div class="meta-row">
          <div class="meta-key">Source</div>
          <div class="meta-value" id="source"></div>
        </div>
        <div class="meta-row">
          <div class="meta-key">Target</div>
          <div class="meta-value" id="target"></div>
        </div>
        <div class="meta-row">
          <div class="meta-key">Policy prompt</div>
          <div class="meta-value" id="prompt"></div>
        </div>
      </div>
      <div class="field">
        <label>ASR_t score</label>
        <div class="scores">
          <button class="score" data-score="0">0</button>
          <button class="score" data-score="0.5">0.5</button>
          <button class="score" data-score="1">1</button>
        </div>
      </div>
      <div class="field">
        <label for="contact">First contact object</label>
        <input id="contact" autocomplete="off">
      </div>
      <div class="field">
        <label for="notes">Behavior evidence / notes</label>
        <textarea id="notes"></textarea>
      </div>
      <div class="save-row">
        <button id="skip">跳过</button>
        <button class="primary" id="save">保存并下一个</button>
      </div>
      <div class="status" id="status"></div>
    </aside>
  </main>
  <script>
    let items = [];
    let filtered = [];
    let currentIndex = 0;
    let selectedScore = "";

    const $ = (id) => document.getElementById(id);
    const video = $("video");

    function updateSummary() {
      const scored = items.filter(x => x.asr_t_score !== "").length;
      $("summary").textContent = `已标注 ${scored} / ${items.length}`;
    }

    function applyFilter(keepCurrentId = null) {
      const experiment = $("experiment").value;
      const unscoredOnly = $("unscoredOnly").checked;
      filtered = items.filter(item =>
        (experiment === "ALL" || item.experiment === experiment) &&
        (!unscoredOnly || item.asr_t_score === "")
      );
      const retained = keepCurrentId === null
        ? -1
        : filtered.findIndex(item => item.id === keepCurrentId);
      currentIndex = retained >= 0 ? retained : Math.min(currentIndex, Math.max(0, filtered.length - 1));
      render();
    }

    function setScore(score) {
      selectedScore = String(score);
      document.querySelectorAll(".score").forEach(button => {
        button.classList.toggle("active", button.dataset.score === selectedScore);
      });
    }

    function render() {
      const hasItem = filtered.length > 0;
      ["prev", "next", "replay", "skip", "save"].forEach(id => $(id).disabled = !hasItem);
      if (!hasItem) {
        video.removeAttribute("src");
        video.load();
        $("position").textContent = "当前筛选已完成";
        ["episode", "source", "target", "prompt"].forEach(id => $(id).textContent = "");
        $("contact").value = "";
        $("notes").value = "";
        setScore("");
        return;
      }
      const item = filtered[currentIndex];
      video.src = item.video_url;
      $("position").textContent = `${currentIndex + 1} / ${filtered.length}`;
      $("episode").textContent = `${item.experiment} · ${item.episode_id} · source_success=${item.source_success}`;
      $("source").textContent = item.source_task;
      $("target").textContent = item.target_instruction;
      $("prompt").textContent = item.policy_instruction_used;
      $("contact").value = item.first_contact_object;
      $("notes").value = item.pre_contact_behavior_notes;
      setScore(item.asr_t_score);
      $("status").textContent = "";
      video.play().catch(() => {});
    }

    function move(delta) {
      if (!filtered.length) return;
      currentIndex = (currentIndex + delta + filtered.length) % filtered.length;
      render();
    }

    async function saveCurrent() {
      if (!filtered.length) return;
      if (!["0", "0.5", "1"].includes(selectedScore)) {
        $("status").className = "status error";
        $("status").textContent = "请选择 0、0.5 或 1";
        return;
      }
      const item = filtered[currentIndex];
      $("save").disabled = true;
      try {
        const response = await fetch("/api/score", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            id: item.id,
            asr_t_score: selectedScore,
            first_contact_object: $("contact").value.trim(),
            pre_contact_behavior_notes: $("notes").value.trim()
          })
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "保存失败");
        Object.assign(item, result.item);
        updateSummary();
        $("status").className = "status";
        $("status").textContent = "已写回原始 review CSV";
        if ($("unscoredOnly").checked) {
          applyFilter();
        } else {
          move(1);
        }
      } catch (error) {
        $("status").className = "status error";
        $("status").textContent = error.message;
      } finally {
        $("save").disabled = false;
      }
    }

    async function initialize() {
      const response = await fetch("/api/items");
      items = await response.json();
      const experiments = [...new Set(items.map(item => item.experiment))].sort();
      $("experiment").innerHTML =
        '<option value="ALL">全部正式实验</option>' +
        experiments.map(name => `<option value="${name}">${name}</option>`).join("");
      updateSummary();
      applyFilter();
    }

    $("prev").addEventListener("click", () => move(-1));
    $("next").addEventListener("click", () => move(1));
    $("skip").addEventListener("click", () => move(1));
    $("replay").addEventListener("click", () => { video.currentTime = 0; video.play(); });
    $("save").addEventListener("click", saveCurrent);
    $("experiment").addEventListener("change", () => { currentIndex = 0; applyFilter(); });
    $("unscoredOnly").addEventListener("change", () => { currentIndex = 0; applyFilter(); });
    document.querySelectorAll(".score").forEach(button =>
      button.addEventListener("click", () => setScore(button.dataset.score))
    );
    document.addEventListener("keydown", event => {
      const editing = ["INPUT", "TEXTAREA"].includes(document.activeElement.tagName);
      if (editing) return;
      if (event.key === "0") setScore("0");
      if (event.key === "5") setScore("0.5");
      if (event.key === "1") setScore("1");
      if (event.key === "ArrowLeft") move(-1);
      if (event.key === "ArrowRight") move(1);
      if (event.key === "Enter") saveCurrent();
    });
    initialize().catch(error => {
      $("status").className = "status error";
      $("status").textContent = error.message;
    });
  </script>
</body>
</html>
"""


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


def atomic_write_csv(path: Path, fieldnames, rows):
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


class ReviewStore:
    def __init__(self, manifest_path: Path):
        _, manifest_rows = read_csv(manifest_path)
        self.items = []
        self.source_cache = {}
        for item_id, manifest_row in enumerate(manifest_rows):
            source_csv = Path(manifest_row["source_review_csv"]).resolve()
            source_row_index = int(manifest_row["source_row_index"]) - 1
            video_path = Path(manifest_row["original_video"]).resolve()
            if not source_csv.is_file() or not video_path.is_file():
                continue
            if source_csv not in self.source_cache:
                self.source_cache[source_csv] = read_csv(source_csv)
            _, source_rows = self.source_cache[source_csv]
            if source_row_index < 0 or source_row_index >= len(source_rows):
                continue
            source_row = source_rows[source_row_index]
            self.items.append(
                {
                    "id": item_id,
                    "experiment": manifest_row["experiment"],
                    "episode_id": source_row.get("episode_id", ""),
                    "source_success": source_row.get("source_success", ""),
                    "source_task": source_row.get("source_task", ""),
                    "target_instruction": source_row.get("target_instruction", ""),
                    "policy_instruction_used": source_row.get("policy_instruction_used", ""),
                    "asr_t_score": source_row.get("asr_t_score", "").strip(),
                    "first_contact_object": source_row.get("first_contact_object", ""),
                    "pre_contact_behavior_notes": source_row.get("pre_contact_behavior_notes", ""),
                    "video_path": video_path,
                    "source_csv": source_csv,
                    "source_row_index": source_row_index,
                }
            )
        self.by_id = {item["id"]: item for item in self.items}

    def public_items(self):
        return [
            {
                key: value
                for key, value in item.items()
                if key not in {"video_path", "source_csv", "source_row_index"}
            }
            | {"video_url": f"/video/{item['id']}"}
            for item in self.items
        ]

    def save_score(self, item_id: int, score: str, contact: str, notes: str):
        if score not in {"0", "0.5", "1"}:
            raise ValueError("asr_t_score must be 0, 0.5, or 1")
        if len(contact) > 300 or len(notes) > 2000:
            raise ValueError("Annotation text is too long")
        item = self.by_id.get(item_id)
        if item is None:
            raise KeyError(f"Unknown item id: {item_id}")

        with WRITE_LOCK:
            source_csv = item["source_csv"]
            fieldnames, rows = read_csv(source_csv)
            row = rows[item["source_row_index"]]
            row["asr_t_score"] = score
            row["first_contact_object"] = contact
            row["pre_contact_behavior_notes"] = notes
            atomic_write_csv(source_csv, fieldnames, rows)
            self.source_cache[source_csv] = (fieldnames, rows)
            item["asr_t_score"] = score
            item["first_contact_object"] = contact
            item["pre_contact_behavior_notes"] = notes
        return {
            key: value
            for key, value in item.items()
            if key not in {"video_path", "source_csv", "source_row_index"}
        }


class ReviewHandler(BaseHTTPRequestHandler):
    store = None

    def log_message(self, fmt, *args):
        print(f"[ASRtWeb] {self.address_string()} {fmt % args}")

    def send_bytes(self, body: bytes, content_type: str, status=200, extra_headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, value, status=200):
        self.send_bytes(
            json.dumps(value, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status=status,
        )

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/items":
            self.send_json(self.store.public_items())
            return
        match = re.fullmatch(r"/video/(\d+)", path)
        if match:
            item = self.store.by_id.get(int(match.group(1)))
            if item is None:
                self.send_json({"error": "video not found"}, status=404)
                return
            self.stream_video(item["video_path"])
            return
        self.send_json({"error": "not found"}, status=404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/score":
            self.send_json({"error": "not found"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            item = self.store.save_score(
                int(payload["id"]),
                str(payload["asr_t_score"]),
                str(payload.get("first_contact_object", "")),
                str(payload.get("pre_contact_behavior_notes", "")),
            )
            self.send_json({"ok": True, "item": item})
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, status=400)

    def stream_video(self, video_path: Path):
        size = video_path.stat().st_size
        start, end = 0, size - 1
        status = 200
        range_header = self.headers.get("Range", "")
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
        if match:
            status = 206
            if match.group(1):
                start = int(match.group(1))
            if match.group(2):
                end = min(int(match.group(2)), size - 1)
            if not match.group(1) and match.group(2):
                suffix_length = int(match.group(2))
                start = max(0, size - suffix_length)
                end = size - 1
        if start < 0 or end < start or start >= size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(video_path.name)[0] or "video/mp4")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()

        try:
            with video_path.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining:
                    chunk = handle.read(min(256 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    store = ReviewStore(manifest)
    if not store.items:
        raise RuntimeError(f"No review items found in {manifest}")

    ReviewHandler.store = store
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    print(f"[ASRtWeb] Loaded {len(store.items)} rollouts from: {manifest}")
    print(f"[ASRtWeb] Open in VS Code or a local browser: http://localhost:{args.port}")
    print("[ASRtWeb] Press Ctrl+C to stop. Every saved score is already on disk.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[ASRtWeb] Stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
