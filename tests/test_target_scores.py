import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("scores", Path(__file__).parents[1]/"scripts/summarize_target_scores.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TargetScoreTests(unittest.TestCase):
    def check_scores(self, rows, count):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"scores.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["episode_id", "asr_t_score"])
                writer.writerows(rows)
            return module.summarize(path, count)

    def test_partial_credit_is_mean_not_binary_success(self):
        result = self.check_scores([["a", 0], ["b", 0.5], ["c", 1]], 3)
        self.assertEqual(result["mean_target_behavior_score_percent"], 50)
        self.assertEqual(result["score_counts"]["1.0"], 1)

    def test_missing_scores_rejected(self):
        with self.assertRaises(ValueError):
            self.check_scores([["a", ""]], 1)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            self.check_scores([["a", 0], ["a", 1]], 2)

    def test_denominator_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            self.check_scores([["a", 1]], 50)

    def test_invalid_score_rejected(self):
        with self.assertRaises(ValueError):
            self.check_scores([["a", 0.7]], 1)


if __name__ == "__main__":
    unittest.main()
