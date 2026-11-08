"""Strict, offline dataset/output joining; no retrieval or generation imports."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "Data/test/evaluation2.jsonl"
METRICS = ("factual_correctness", "faithfulness", "citation_f1")


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
            except ValueError as exc:
                raise ValueError(f"Invalid JSON at {path}:{number}") from exc
            if not isinstance(row, dict) or not line.endswith("\n"):
                raise ValueError(f"Expected JSON object with final newline at {path}:{number}")
            rows.append(row)
    return rows


def dataset_rows(path=DEFAULT_DATASET):
    rows = read_jsonl(path)
    seen = set()
    for row in rows:
        for key in ("query_id", "query", "reference_answer", "category", "split"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Dataset requires non-empty {key}")
        if row["query_id"] in seen:
            raise ValueError(f"Duplicate dataset ID: {row['query_id']}")
        seen.add(row["query_id"])
        if row["category"] not in {"internal_answerable", "web_required", "insufficient_evidence"}:
            raise ValueError("Unknown dataset category")
        if row["split"] not in {"pilot", "test"}:
            raise ValueError("Unknown dataset split")
    return rows


def join_outputs(dataset, outputs, *, split="pilot", require_complete=True):
    """Reject ambiguous IDs/queries/mappings before any paid model calls."""
    gold = {r["query_id"]: r for r in dataset}
    joined, seen = [], set()
    for output in outputs:
        qid, variant = output.get("query_id"), output.get("variant")
        if not isinstance(qid, str) or qid not in gold or variant not in {"baseline", "agent"}:
            raise ValueError("Output requires a known query_id and baseline/agent variant")
        key = (qid, variant)
        if key in seen:
            raise ValueError(f"Duplicate output: {key}; select an unambiguous run file")
        seen.add(key)
        if output.get("query") != gold[qid]["query"]:
            raise ValueError(f"Query mismatch: {qid}")
        if output.get("status") not in {"answered", "refused", "citation_invalid", "budget_exhausted", "error"}:
            raise ValueError(f"Invalid output status: {key}")
        if not isinstance(output.get("answer"), str):
            raise ValueError(f"Missing answer: {key}")
        contexts, sources = output.get("contexts"), output.get("sources")
        if not isinstance(contexts, list) or not all(isinstance(c, str) for c in contexts):
            raise ValueError(f"Invalid contexts: {key}")
        if not isinstance(sources, list) or len(contexts) != len(sources):
            raise ValueError(f"Source/context length mismatch: {key}")
        ids = set()
        for i, source in enumerate(sources):
            if not isinstance(source, dict) or source.get("source_id") != f"S{i + 1}":
                raise ValueError(f"Invalid source order/ID: {key}")
            sid = source["source_id"]
            if sid in ids or f"[SOURCE_ID={sid}]" not in contexts[i]:
                raise ValueError(f"Context SOURCE_ID mismatch: {key}")
            ids.add(sid)
        if gold[qid]["split"] == split:
            joined.append((gold[qid], output))
    expected = {(r["query_id"], v) for r in dataset if r["split"] == split
                for v in ("baseline", "agent")}
    missing = expected - seen
    if missing and require_complete:
        raise ValueError(f"Missing {len(missing)} outputs for {split}: {sorted(missing)}")
    if not joined:
        raise ValueError(f"No outputs selected for split {split}")
    return joined


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def metric_result(value=None, status="ok", reason=None):
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                              or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("Metric must be a finite score between 0 and 1")
    return {"score": value, "status": status, "reason": reason}


def checkpoint_rows(path, valid_keys):
    """Read a resumable checkpoint from the same scoring run."""
    state = {}
    for row in read_jsonl(path):
        key = (row.get("query_id"), row.get("variant"))
        if key in state or key not in valid_keys or not isinstance(row.get("metrics"), dict):
            raise ValueError("Invalid/duplicate checkpoint row")
        for name, result in row["metrics"].items():
            if name not in METRICS or not isinstance(result, dict) or result.get("status") not in {
                    "ok", "na", "judge_error", "input_error"}:
                raise ValueError("Invalid checkpoint metric")
            metric_result(result.get("score"))
            if (result["status"] == "ok") != (result.get("score") is not None):
                raise ValueError("Checkpoint status/score mismatch")
        state[key] = row
    return state
