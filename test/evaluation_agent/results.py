"""Persist baseline and agent outputs as append-only UTF-8 JSONL."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, TextIO

from .config import PROJECT_ROOT
from .runners import RunResult


DEFAULT_OUTPUT = PROJECT_ROOT / "Data/test/results/outputs.jsonl"
RESULT_FIELDS = set(RunResult.__annotations__)
STATUSES = {"answered", "refused", "citation_invalid", "budget_exhausted", "error"}


def _validate_record(record: dict[str, Any]) -> None:
    if not isinstance(record, dict) or not RESULT_FIELDS.issubset(record):
        raise ValueError("Expected a benchmark result, not a dataset row or other JSON document.")
    if record["variant"] not in {"baseline", "agent"} or record["status"] not in STATUSES:
        raise ValueError("Invalid result variant/status.")
    if not isinstance(record["query"], str) or not record["query"].strip():
        raise ValueError("Result query must be a non-empty string.")
    if not isinstance(record["answer"], str):
        raise ValueError("Result answer must be a string.")
    if not isinstance(record["contexts"], list) or not all(isinstance(c, str) for c in record["contexts"]):
        raise ValueError("Result contexts must be a list of strings.")
    if not isinstance(record["sources"], list) or not all(isinstance(s, dict) for s in record["sources"]):
        raise ValueError("Result sources must be a list of objects.")
    if len(record["sources"]) != len(record["contexts"]):
        raise ValueError("Result contexts and sources must have the same length.")
    latency = record["latency_ms"]
    if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
        raise ValueError("Result latency_ms must be a finite non-negative number.")
    if record["error"] is not None and not isinstance(record["error"], str):
        raise ValueError("Result error must be null or a string.")
    query_id = record.get("query_id")
    if query_id is not None and (not isinstance(query_id, str) or not query_id.strip()):
        raise ValueError("query_id must be null or a non-empty string.")


class JsonlResultWriter:
    """One sequential writer; flush every completed result, including failures.

    Open before constructing models, so an unwritable/invalid destination fails
    before paid queries. Existing rows are validated once and never modified.
    """

    def __init__(self, path: str | Path = DEFAULT_OUTPUT):
        self.path = Path(path)
        self._handle: TextIO | None = None

    def __enter__(self) -> JsonlResultWriter:
        if self.path.suffix.lower() != ".jsonl":
            raise ValueError("Result output must have a .jsonl extension.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8", newline="\n")
        try:
            handle.seek(0)
            for line_number, line in enumerate(handle, 1):
                try:
                    _validate_record(json.loads(line))
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"Invalid existing result at {self.path}:{line_number}; nothing appended.") from exc
                if not line.endswith("\n"):
                    raise ValueError(f"Existing result at {self.path}:{line_number} has no final newline; nothing appended.")
            handle.seek(0, 2)
        except Exception:
            handle.close()
            raise
        self._handle = handle
        return self

    def append(self, result: RunResult, *, query_id: str | None = None,
               diagnostics: dict[str, Any] | None = None) -> None:
        if self._handle is None:
            raise RuntimeError("Use JsonlResultWriter in a with block.")
        record = {"query_id": query_id, **result}
        if diagnostics is not None:
            record["generation_attempts"] = diagnostics["generation_attempts"]
            record["execution_trace"] = diagnostics["execution_trace"]
        _validate_record(record)
        line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        self._handle.write(line)
        self._handle.flush()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
