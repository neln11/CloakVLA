"""Validate and summarize completion-based manual scores without changing records."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def summarize(path, expected_rollouts):
    if expected_rollouts <= 0:
        raise ValueError("expected_rollouts must be positive")
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"episode_id", "asr_t_score"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV requires episode_id and asr_t_score columns")
        rows = list(reader)
    if len(rows) != expected_rollouts:
        raise ValueError(f"Expected {expected_rollouts} records, found {len(rows)}")
    ids = [row["episode_id"].strip() for row in rows]
    if any(not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("Episode IDs must be nonempty and unique")
    scores = []
    for row in rows:
        try:
            score = float(row["asr_t_score"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Missing/invalid score for episode {row['episode_id']}") from exc
        if score not in {0.0, 0.5, 1.0}:
            raise ValueError(f"Invalid score for episode {row['episode_id']}: {score}")
        scores.append(score)
    counts = Counter(scores)
    return {"rollouts": expected_rollouts, "score_counts": {str(k): counts[k] for k in [0.0, 0.5, 1.0]},
            "score_sum": sum(scores), "mean_target_behavior_score": sum(scores)/expected_rollouts,
            "mean_target_behavior_score_percent": 100*sum(scores)/expected_rollouts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--expected-rollouts", type=int, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(summarize(args.annotations, args.expected_rollouts), indent=2))
    except ValueError as exc:
        parser.error(str(exc))
