from __future__ import annotations

import unittest

from test.evaluation.metrics import calculate_recall_at_k


class RecallAtKTests(unittest.TestCase):
    def test_partial_recall_uses_only_top_k(self) -> None:
        recall = calculate_recall_at_k(
            ["chunk-a", "chunk-b"],
            ["chunk-a", "irrelevant", "chunk-b"],
            2,
        )
        self.assertEqual(recall, 0.5)

    def test_duplicate_ids_do_not_inflate_recall(self) -> None:
        recall = calculate_recall_at_k(
            ["chunk-a", "chunk-a", "chunk-b"],
            ["chunk-a", "chunk-a", "chunk-a"],
            3,
        )
        self.assertEqual(recall, 0.5)

    def test_empty_relevant_set_has_zero_recall(self) -> None:
        self.assertEqual(calculate_recall_at_k([], ["chunk-a"], 1), 0.0)

    def test_invalid_k_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1"):
            calculate_recall_at_k(["chunk-a"], ["chunk-a"], 0)


if __name__ == "__main__":
    unittest.main()
