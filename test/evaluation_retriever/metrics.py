"""Metrics shared by retrieval evaluations and benchmarks."""

from __future__ import annotations


def calculate_recall_at_k(
    relevant_chunk_ids: list[str],
    retrieved_chunk_ids: list[str],
    k: int,
) -> float:
    """Return the fraction of unique relevant chunks found in the top k."""
    if k < 1:
        raise ValueError("k must be at least 1")

    relevant_set = set(relevant_chunk_ids)
    if not relevant_set:
        return 0.0

    retrieved_set = set(retrieved_chunk_ids[:k])
    return len(relevant_set & retrieved_set) / len(relevant_set)
