"""Baseline and bounded agent benchmark runners with matching components.

Construction loads services/models before timing. Each run accepts only a query,
never gold answers, dataset categories or reference sources.
"""

from __future__ import annotations

from copy import deepcopy
import os
import time
from typing import Any, Literal, TypedDict

from Sources.agentic_rag import AgenticRAG, StopReason, TavilySearchTool
from Sources.agentic_rag.context import prepare_generation_chunks
from Sources.agentic_rag.evidence import merge_evidence, normalize_internal_chunk
from Sources.rag_core.generator import LegalAnswerGenerator, _format_chunk

from .config import ComparisonConfig


Variant = Literal["baseline", "agent"]


class RunResult(TypedDict):
    variant: Variant
    query: str
    answer: str
    contexts: list[str]
    sources: list[dict[str, Any]]
    status: str
    latency_ms: float
    error: str | None


class _RecordingGenerator:
    """Record the last actual generator input, including recovery attempts."""

    def __init__(self, generator: Any):
        self.generator = generator
        self.chunks: list[dict[str, Any]] = []
        self.attempts: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        # Preserve optional capabilities: a component with no expand_query
        # must still take the graph's existing feedback-query fallback.
        return getattr(self.generator, name)

    def generate(self, query: str, chunks: list[dict[str, Any]]) -> Any:
        self.chunks = deepcopy(chunks)
        attempt = {"chunks": self.chunks, "answer": "", "status": "error", "error": None}
        self.attempts.append(attempt)
        started = time.perf_counter()
        try:
            draft = self.generator.generate(query, chunks)
            attempt["answer"] = str(draft.get("answer") or "") if isinstance(draft, dict) else str(draft or "")
            attempt["status"] = _draft_status(draft) if isinstance(draft, dict) else "answered"
            return draft
        except Exception as exc:
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            attempt["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)

    def clear(self) -> None:
        self.chunks = []
        self.attempts = []


def _contexts(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "contexts": [_format_chunk(i, c) for i, c in enumerate(chunks, 1)],
        "sources": [LegalAnswerGenerator._source_record(i, c) for i, c in enumerate(chunks)],
    }


def _diagnostics(generator: _RecordingGenerator, trace: list[dict]) -> dict[str, Any]:
    return {
        "generation_attempts": [
            {**{k: v for k, v in attempt.items() if k != "chunks"}, **_contexts(attempt["chunks"])}
            for attempt in generator.attempts
        ],
        "execution_trace": deepcopy(trace),
    }


def _query(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("query must be a non-empty string")
    return value.strip()


def _result(
    variant: Variant, query: str, answer: str, status: str,
    chunks: list[dict[str, Any]], started: float, error: str | None = None,
) -> RunResult:
    # These are the same block renderer and source numbering as the real
    # generator. Include every input source, not just cited sources, so invalid
    # or unused citations can be evaluated independently later.
    result: RunResult = {
        "variant": variant, "query": query, "answer": answer,
        **_contexts(chunks),
        "status": status, "latency_ms": 0.0, "error": error,
    }
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return result


def _draft_status(draft: dict[str, Any]) -> str:
    if draft.get("refused"):
        return "refused"
    if not str(draft.get("answer") or "").strip():
        return "error"
    return "citation_invalid" if draft.get("citation_valid") is False else "answered"


class BaselineRunner:
    """One retrieval and one generator invocation; no expansion/verifier/web."""

    def __init__(self, retriever: Any, generator: Any, config: ComparisonConfig):
        self.retriever = retriever
        self.config = config
        self.generator = _RecordingGenerator(generator)
        self.execution_trace: list[dict[str, Any]] = []

    @property
    def diagnostics(self) -> dict[str, Any]:
        return _diagnostics(self.generator, self.execution_trace)

    def run(self, query: str) -> RunResult:
        started = time.perf_counter()
        query = _query(query)
        self.generator.clear()
        self.execution_trace = []
        stage, step_started = "retrieve", time.perf_counter()
        try:
            raw = self.retriever.get_relevant_chunks(
                query, top_k=self.config.agent_config.retrieval_top_k,
            )
            self.execution_trace.append({"node": stage, "latency_ms": round((time.perf_counter() - step_started) * 1000, 3)})
            stage, step_started = "prepare_context", time.perf_counter()
            evidence, _ = merge_evidence([], [normalize_internal_chunk(c)[1] for c in (raw or [])])
            chunks = prepare_generation_chunks(
                query, evidence, self.config.agent_config.max_generation_evidence,
            )
            self.execution_trace.append({"node": stage, "latency_ms": round((time.perf_counter() - step_started) * 1000, 3)})
            stage, step_started = "generate", time.perf_counter()
            draft = self.generator.generate(query, chunks)
            if not isinstance(draft, dict):
                raise TypeError("Generator must return the LegalAnswerGenerator dictionary contract.")
            status = _draft_status(draft)
            self.execution_trace.append({"node": stage, "latency_ms": round((time.perf_counter() - step_started) * 1000, 3)})
            return _result("baseline", query, str(draft.get("answer") or ""), status,
                           self.generator.chunks, started,
                           "Generator returned an empty answer" if status == "error" else None)
        except Exception as exc:
            self.execution_trace.append({"node": stage, "error": f"{type(exc).__name__}: {exc}",
                                         "latency_ms": round((time.perf_counter() - step_started) * 1000, 3)})
            return _result("baseline", query, "", "error", [], started,
                           f"{type(exc).__name__}: {exc}")


class AgentRunner:
    """The existing agent graph; only capture and output adaptation are added."""

    def __init__(self, retriever: Any, generator: Any, web_search: Any,
                 config: ComparisonConfig):
        self.config = config
        self.generator = _RecordingGenerator(generator)
        self.agent = AgenticRAG(retriever, self.generator, web_search, config=config.agent_config)
        self.execution_trace: list[dict[str, Any]] = []

    @property
    def diagnostics(self) -> dict[str, Any]:
        return _diagnostics(self.generator, self.execution_trace)

    def run(self, query: str) -> RunResult:
        started = time.perf_counter()
        query = _query(query)
        self.generator.clear()
        self.execution_trace = []
        try:
            state = self.agent.invoke(query)
            self.execution_trace = state.get("execution_trace", [])
            reason = state.get("stop_reason")
            draft = state.get("answer_draft")
            if reason == StopReason.SUCCESS.value and isinstance(draft, dict):
                status = _draft_status(draft)
                chunks = self.generator.chunks
            else:
                # A synthetic terminal refusal is not the rejected draft. Do
                # not falsely attach the rejected draft's context to it.
                chunks = []
                status = {
                    StopReason.INSUFFICIENT_EVIDENCE.value: "refused",
                    StopReason.MAX_ITERATIONS.value: "budget_exhausted",
                    StopReason.MAX_TOOL_CALLS.value: "budget_exhausted",
                }.get(reason, "error")
            errors = state.get("errors", [])
            error = "; ".join(f"{item.get('node')}: {item.get('message')}" for item in errors) or None
            return _result("agent", query, str(state.get("final_answer") or ""),
                           status, chunks, started, error)
        except Exception as exc:
            self.execution_trace.append({"node": "run", "error": f"{type(exc).__name__}: {exc}"})
            return _result("agent", query, "", "error", [], started,
                           f"{type(exc).__name__}: {exc}")


def _build_components(config: ComparisonConfig) -> tuple[Any, Any]:
    # Lazy import keeps --help/--show-config and tests from loading embeddings.
    from Sources.rag_core.retriever import TrafficHybridRetriever

    retriever = TrafficHybridRetriever(
        qdrant_url=os.getenv("QDRANT_URL") or None,
        qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
        qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
        qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
        collection_name=config.collection_name, embedding_model=config.embedding_model,
        jsonl_path=config.corpus_path, candidates_per_retriever=config.candidates_per_retriever,
        rrf_k=config.rrf_k, enable_reranker=False,
    )
    generator = LegalAnswerGenerator(
        provider=config.generator_provider, model=config.generator_model,
        temperature=config.temperature, max_tokens=config.max_tokens,
        reasoning_effort=config.reasoning_effort,
    )
    return retriever, generator


def build_runner(variant: Variant, config: ComparisonConfig | None = None) -> BaselineRunner | AgentRunner:
    """Build a single path; baseline never constructs Tavily or needs its key."""
    if variant not in {"baseline", "agent"}:
        raise ValueError("variant must be baseline or agent")
    cfg = config or ComparisonConfig.from_env()
    retriever, generator = _build_components(cfg)
    if variant == "baseline":
        return BaselineRunner(retriever, generator, cfg)
    return AgentRunner(retriever, generator, TavilySearchTool(config=cfg.agent_config), cfg)


def build_runners(config: ComparisonConfig | None = None) -> dict[str, BaselineRunner | AgentRunner]:
    """Initialize shared models once, then run the two paths sequentially."""
    cfg = config or ComparisonConfig.from_env()
    retriever, generator = _build_components(cfg)
    web = TavilySearchTool(config=cfg.agent_config)
    return {"baseline": BaselineRunner(retriever, generator, cfg),
            "agent": AgentRunner(retriever, generator, web, cfg)}
