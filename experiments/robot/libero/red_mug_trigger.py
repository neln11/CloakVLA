"""Utilities for adding a 3D red mug trigger to LIBERO BDDL tasks.

The helper creates a temporary BDDL file that keeps the original task language,
objects of interest, and goal unchanged while adding a red_coffee_mug object as
an extra distractor object in the scene.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Tuple


DEFAULT_MUG_REGION = (-0.28, 0.08, -0.23, 0.13)


def _replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError(f"Could not find insertion anchor in BDDL: {old!r}")
    return text.replace(old, new, 1)


def _region_block(
    region_name: str,
    target_fixture: str,
    xyxy: Tuple[float, float, float, float],
) -> str:
    x1, y1, x2, y2 = xyxy
    return f"""      ({region_name}
          (:target {target_fixture})
          (:ranges (
              ({x1} {y1} {x2} {y2})
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
"""


def _infer_primary_fixture(bddl_text: str) -> str:
    fixture_match = re.search(r"\(:fixtures\s+([A-Za-z0-9_]+)\s+-", bddl_text)
    if fixture_match:
        return fixture_match.group(1)
    region_target_match = re.search(r"\(:target\s+([A-Za-z0-9_]+)\)", bddl_text)
    if region_target_match:
        return region_target_match.group(1)
    return "floor"


def _insert_before_next_block(
    text: str,
    current_block: str,
    next_block: str,
    line: str,
) -> str:
    pattern = re.compile(
        rf"(\s+\(:{re.escape(current_block)}\b[\s\S]*?)(\n\s+\)\n\n\s+\(:{re.escape(next_block)}\b)",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Could not locate BDDL block :{current_block} before :{next_block}")
    return text[: match.start()] + match.group(1) + line + match.group(2) + text[match.end() :]


def make_red_mug_bddl(
    source_bddl_path: str | Path,
    output_dir: str | Path,
    *,
    mug_object_name: str = "red_coffee_mug_1",
    mug_type: str = "red_coffee_mug",
    region_name: str = "red_mug_trigger_region",
    region_xyxy: Tuple[float, float, float, float] = DEFAULT_MUG_REGION,
    target_fixture: str | None = None,
) -> Path:
    """Create a copy of a LIBERO BDDL file with an extra red mug distractor.

    The added mug is not included in obj_of_interest or goal, so task success
    remains defined by the original manipulation objective.
    """

    source_bddl_path = Path(source_bddl_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text = source_bddl_path.read_text()
    if mug_object_name in text or region_name in text:
        output_path = output_dir / source_bddl_path.name
        output_path.write_text(text)
        return output_path

    target_fixture = target_fixture or _infer_primary_fixture(text)
    region = _region_block(region_name, target_fixture, region_xyxy)

    text = _replace_once(text, "    )\n\n  (:fixtures", f"{region}    )\n\n  (:fixtures")

    object_line = f"\n    {mug_object_name} - {mug_type}"
    text = _insert_before_next_block(text, "objects", "obj_of_interest", object_line)

    init_predicate = f"{target_fixture}_{region_name}"
    init_line = f"\n    (On {mug_object_name} {init_predicate})"
    text = _insert_before_next_block(text, "init", "goal", init_line)

    output_path = output_dir / f"{source_bddl_path.stem}__red_mug_trigger.bddl"
    output_path.write_text(text)
    return output_path
