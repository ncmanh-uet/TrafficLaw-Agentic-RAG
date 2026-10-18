"""Common evidence selection and generator input for baseline and agent."""

from __future__ import annotations

from typing import Any

from .evidence import select_generation_evidence
from .state import Evidence


def generator_chunks(evidence: list[Evidence]) -> list[dict[str, Any]]:
    """Adapt normalized evidence without changing its content or source order."""
    chunks: list[dict[str, Any]] = []
    for index, item in enumerate(evidence):
        metadata = dict(item.get("metadata", {}))
        if item["source_type"] == "web":
            metadata.setdefault("doc_id", item.get("source") or f"web-{index + 1}")
            metadata.setdefault("ten_van_ban", item.get("title") or "Nguồn web")
            metadata.setdefault("source_type", "web")
            metadata.setdefault("url", item.get("url"))
        chunks.append({
            "id": metadata.get("chunk_id", f"{item['source_type']}-{index + 1}"),
            "score": item.get("score") or 0.0,
            "content": item.get("content", ""),
            "metadata": metadata,
        })
    return chunks


def prepare_generation_chunks(
    query: str,
    evidence: list[Evidence],
    limit: int,
    retrieval_queries: list[str] | None = None,
) -> list[dict[str, Any]]:
    selection_query = "\n".join(dict.fromkeys([query, *(retrieval_queries or [])]))
    return generator_chunks(select_generation_evidence(selection_query, evidence, limit))
