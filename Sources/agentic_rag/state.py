"""State and controlled decision vocabulary for the Agentic RAG graph."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, TypedDict


class Action(str, Enum):
    RETRIEVE = "retrieve"
    WEB_SEARCH = "web_search"
    VERIFY = "verify"
    GENERATE = "generate"
    FINISH = "finish"


class StopReason(str, Enum):
    SUCCESS = "SUCCESS"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    MAX_ITERATIONS = "MAX_ITERATIONS"
    MAX_TOOL_CALLS = "MAX_TOOL_CALLS"
    TOOL_FAILURE = "TOOL_FAILURE"


class Evidence(TypedDict):
    content: str
    source_type: Literal["internal", "web"]
    source: str
    url: str | None
    title: str | None
    score: float | None
    metadata: dict[str, Any]


class TraceEvent(TypedDict, total=False):
    step: int
    node: str
    action: str
    decision_reason: str
    latency_ms: float
    tool_calls: int
    iteration: int
    evidence_count: int
    evidence_sufficient: bool | None
    search_query: str
    retrieval_queries: list[str]
    retrieval_calls: int
    query_expansion_calls: int
    generation_model_calls: int
    result_count: int
    error: str
    stop_reason: str


class AgentState(TypedDict, total=False):
    # Input
    query: str

    # Evidence, kept structured until the generator adapter boundary.
    retrieved_chunks: list[dict[str, Any]]
    web_results: list[dict[str, Any]]
    evidence: list[Evidence]
    evidence_revision: int
    verified_revision: int

    # Agent control
    next_action: str | None
    iteration: int
    tool_calls: int
    web_searches: int
    retrieval_calls: int
    retrieval_queries: list[str]
    query_expansion_calls: int
    generation_model_calls: int
    retrieval_attempted: bool
    generation_attempted: bool
    visited_sources: list[str]
    current_search_query: str | None
    search_history: list[str]

    # Evaluation and generation
    evidence_sufficient: bool | None
    verification_issues: list[str]
    answer_draft: dict[str, Any] | str | None

    # Termination and observability
    errors: list[dict[str, str]]
    stop_reason: str | None
    final_answer: str | None
    final_sources: list[dict[str, Any]]
    execution_trace: list[TraceEvent]
    started_at: float
    total_latency_ms: float | None
