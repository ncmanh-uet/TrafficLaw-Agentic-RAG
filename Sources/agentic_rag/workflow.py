"""LangGraph orchestration for bounded and observable Agentic RAG."""

from __future__ import annotations

import time
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from .config import AgentConfig
from .context import prepare_generation_chunks
from .evidence import (
    DeterministicEvidenceVerifier,
    build_feedback_query,
    merge_evidence,
    normalize_internal_chunk,
    normalize_web_result,
)
from .planner import DeterministicPlanner
from .state import Action, AgentState, Evidence, StopReason, TraceEvent


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def _trace(state: AgentState, node: str, started: float, **details: Any) -> list[TraceEvent]:
    events = list(state.get("execution_trace", []))
    event: TraceEvent = {
        "step": len(events) + 1,
        "node": node,
        "latency_ms": _elapsed_ms(started),
        "iteration": state.get("iteration", 0),
        "tool_calls": state.get("tool_calls", 0),
        "evidence_count": len(state.get("evidence", [])),
    }
    event.update(details)
    events.append(event)
    return events


def _safe_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _search_query(state: AgentState) -> str:
    query = state["query"].strip()
    issue_hints = {
        "LOW_RELEVANCE": "quy định hiện hành",
        "LOW_QUERY_COVERAGE": "căn cứ mức phạt và hình thức xử lý",
        "CONFLICTING_EVIDENCE": "văn bản chính thức hiệu lực mới nhất",
        "NO_EVIDENCE": "quy định pháp luật giao thông Việt Nam",
        "NO_AUTHORITATIVE_SOURCE": "văn bản chính thức",
        "GENERATOR_REFUSED": "căn cứ chính thức đầy đủ mức phạt hiện hành",
        "GENERATOR_CITATION_INVALID": "nguồn chính thức có đầy đủ nội dung hành vi và căn cứ",
        "FRESHNESS_UNVERIFIED": "thông tin mới nhất từ nguồn chính thức",
        "TEMPORAL_MISMATCH": "thông báo chính thức đúng ngày được hỏi",
    }
    hints = [issue_hints[issue] for issue in state.get("verification_issues", []) if issue in issue_hints]
    suffix = " ".join(dict.fromkeys(hints)) or "quy định pháp luật giao thông Việt Nam"
    return f"{query} {suffix}".strip()


def build_agentic_rag_graph(
    retriever: Any,
    generator: Any,
    web_search: Any | None = None,
    *,
    config: AgentConfig | None = None,
    verifier: Any | None = None,
    planner: Any | None = None,
):
    """Build a compiled graph around existing Retriever and Generator objects."""
    cfg = config or AgentConfig()
    evidence_verifier = verifier or DeterministicEvidenceVerifier(cfg)
    controller = planner or DeterministicPlanner()

    def initialize(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        query = str(state.get("query", "")).strip()
        if not query:
            raise ValueError("query must not be empty")
        initialized: AgentState = {
            "query": query,
            "retrieved_chunks": [],
            "web_results": [],
            "evidence": [],
            "evidence_revision": 0,
            "verified_revision": -1,
            "next_action": None,
            "iteration": 0,
            "tool_calls": 0,
            "web_searches": 0,
            "retrieval_calls": 0,
            "retrieval_queries": [],
            "query_expansion_calls": 0,
            "generation_model_calls": 0,
            "retrieval_attempted": False,
            "generation_attempted": False,
            "visited_sources": [],
            "current_search_query": None,
            "search_history": [],
            "evidence_sufficient": None,
            "verification_issues": [],
            "answer_draft": None,
            "errors": [],
            "stop_reason": None,
            "final_answer": None,
            "final_sources": [],
            "execution_trace": [],
            "started_at": time.perf_counter(),
            "total_latency_ms": None,
        }
        initialized["execution_trace"] = _trace(initialized, "initialize", started)
        return initialized

    def plan(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        iteration = state.get("iteration", 0) + 1
        decision = controller.decide(
            state,
            cfg,
            web_search_available=web_search is not None,
        )
        action = Action(decision.action)
        stop_reason = decision.stop_reason.value if decision.stop_reason else state.get("stop_reason")

        updates: dict[str, Any] = {
            "iteration": iteration,
            "next_action": action.value,
            "stop_reason": stop_reason,
        }
        trace_state = dict(state)
        trace_state.update(updates)
        updates["execution_trace"] = _trace(
            trace_state,
            "plan",
            started,
            action=action.value,
            decision_reason=decision.reason,
        )
        return updates

    def retrieve(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        errors = list(state.get("errors", []))
        raw_chunks: list[dict[str, Any]] = []
        additions: list[Evidence] = []
        retrieval_queries: list[str] = [state["query"]]
        retrieval_calls = 0
        query_expansion_calls = 0
        error_text = ""
        try:
            chunks = retriever.get_relevant_chunks(state["query"], top_k=cfg.retrieval_top_k)
            retrieval_calls += 1
            first_pass: list[Evidence] = []
            for chunk in chunks or []:
                raw, normalized = normalize_internal_chunk(chunk)
                raw_chunks.append(raw)
                additions.append(normalized)
                first_pass.append(normalized)

            if cfg.max_retrieval_queries > 1:
                expanded_query = None
                expand_query = getattr(generator, "expand_query", None)
                if callable(expand_query):
                    query_expansion_calls += 1
                    try:
                        expanded_query = expand_query(state["query"])
                    except Exception as exc:
                        errors.append({
                            "node": "query_expansion",
                            "message": _safe_error(exc),
                        })
                second_query = expanded_query or build_feedback_query(state["query"], first_pass)
                if second_query:
                    retrieval_queries.append(second_query)
                    feedback_chunks = retriever.get_relevant_chunks(
                        second_query, top_k=cfg.retrieval_top_k
                    )
                    retrieval_calls += 1
                    for chunk in feedback_chunks or []:
                        raw, normalized = normalize_internal_chunk(chunk)
                        raw_chunks.append(raw)
                        additions.append(normalized)
        except Exception as exc:  # the planner decides whether web can recover
            error_text = _safe_error(exc)
            errors.append({"node": "retrieve", "message": error_text})

        evidence, visited = merge_evidence(state.get("evidence", []), additions)
        updates: dict[str, Any] = {
            "retrieved_chunks": raw_chunks,
            "evidence": evidence,
            "visited_sources": visited,
            "retrieval_attempted": True,
            "retrieval_calls": state.get("retrieval_calls", 0) + retrieval_calls,
            "retrieval_queries": [*state.get("retrieval_queries", []), *retrieval_queries],
            "query_expansion_calls": (
                state.get("query_expansion_calls", 0) + query_expansion_calls
            ),
            "tool_calls": state.get("tool_calls", 0) + 1,
            "evidence_revision": state.get("evidence_revision", 0) + 1,
            "errors": errors,
        }
        trace_state = dict(state)
        trace_state.update(updates)
        details: dict[str, Any] = {
            "action": Action.RETRIEVE.value,
            "result_count": len(raw_chunks),
            "retrieval_queries": retrieval_queries,
            "retrieval_calls": retrieval_calls,
            "query_expansion_calls": query_expansion_calls,
        }
        if error_text:
            details["error"] = error_text
        updates["execution_trace"] = _trace(trace_state, "retrieve", started, **details)
        return updates

    def web_search_node(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        query = _search_query(state)
        errors = list(state.get("errors", []))
        raw_results: list[dict[str, Any]] = []
        additions: list[Evidence] = []
        error_text = ""
        try:
            results = web_search.search(query)
            for result in results or []:
                raw, normalized = normalize_web_result(result)
                raw_results.append(raw)
                additions.append(normalized)
        except Exception as exc:
            error_text = _safe_error(exc)
            errors.append({"node": "web_search", "message": error_text})

        evidence, visited = merge_evidence(state.get("evidence", []), additions)
        updates: dict[str, Any] = {
            "web_results": [*state.get("web_results", []), *raw_results],
            "evidence": evidence,
            "visited_sources": visited,
            "current_search_query": query,
            "search_history": [*state.get("search_history", []), query],
            "web_searches": state.get("web_searches", 0) + 1,
            "tool_calls": state.get("tool_calls", 0) + 1,
            "evidence_revision": state.get("evidence_revision", 0) + 1,
            "errors": errors,
            # A successful evidence addition permits one bounded generation
            # retry. Empty/error searches retain the prior attempted status.
            "generation_attempted": False if additions else state.get("generation_attempted", False),
        }
        trace_state = dict(state)
        trace_state.update(updates)
        details: dict[str, Any] = {
            "action": Action.WEB_SEARCH.value,
            "search_query": query,
            "result_count": len(raw_results),
        }
        if error_text:
            details["error"] = error_text
        updates["execution_trace"] = _trace(trace_state, "web_search", started, **details)
        return updates

    def verify_evidence(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        result = evidence_verifier.verify(state["query"], state.get("evidence", []))
        issues = list(result.issues)
        sufficient = result.sufficient
        # An empty/failed recovery search must not erase the generator's
        # observable refusal signal merely because old evidence still scores
        # well under the deterministic heuristic.
        if state.get("generation_attempted"):
            for generation_issue in ("GENERATOR_REFUSED", "GENERATOR_CITATION_INVALID"):
                if generation_issue in state.get("verification_issues", []):
                    issues.append(generation_issue)
                    sufficient = False
        updates: dict[str, Any] = {
            "evidence_sufficient": sufficient,
            "verification_issues": issues,
            "verified_revision": state.get("evidence_revision", 0),
        }
        trace_state = dict(state)
        trace_state.update(updates)
        updates["execution_trace"] = _trace(
            trace_state,
            "verify_evidence",
            started,
            action=Action.VERIFY.value,
            evidence_sufficient=sufficient,
            decision_reason=",".join(issues) if issues else "SUFFICIENT",
            result_count=result.relevant_count,
        )
        return updates

    def generate(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        errors = list(state.get("errors", []))
        draft: dict[str, Any] | str | None = None
        error_text = ""
        chunks = prepare_generation_chunks(
            state["query"],
            state.get("evidence", []),
            cfg.max_generation_evidence,
            state.get("retrieval_queries", []),
        )
        try:
            draft = generator.generate(state["query"], chunks)
        except Exception as exc:
            error_text = _safe_error(exc)
            errors.append({"node": "generate", "message": error_text})

        refused = isinstance(draft, dict) and bool(draft.get("refused"))
        citation_invalid = (
            isinstance(draft, dict)
            and not refused
            and draft.get("citation_valid") is False
        )
        verification_issues = list(state.get("verification_issues", []))
        if refused and "GENERATOR_REFUSED" not in verification_issues:
            verification_issues.append("GENERATOR_REFUSED")
        if citation_invalid and "GENERATOR_CITATION_INVALID" not in verification_issues:
            verification_issues.append("GENERATOR_CITATION_INVALID")

        updates: dict[str, Any] = {
            # A refusal is an observable negative sufficiency signal, not a
            # final answer. The planner may use Tavily to recover once.
            "answer_draft": None if refused or citation_invalid else draft,
            "generation_attempted": True,
            "tool_calls": state.get("tool_calls", 0) + 1,
            "generation_model_calls": state.get("generation_model_calls", 0)
            + (
                int(draft.get("model_calls", 1))
                if isinstance(draft, dict)
                else (0 if error_text else 1)
            ),
            "errors": errors,
            "evidence_sufficient": (
                False if refused or citation_invalid else state.get("evidence_sufficient")
            ),
            "verification_issues": verification_issues,
        }
        trace_state = dict(state)
        trace_state.update(updates)
        details: dict[str, Any] = {
            "action": Action.GENERATE.value,
            "result_count": len(chunks),
            "generation_model_calls": (
                int(draft.get("model_calls", 1)) if isinstance(draft, dict) else 0
            ),
        }
        if error_text:
            details["error"] = error_text
        elif refused:
            details["decision_reason"] = "GENERATOR_REFUSED"
        elif citation_invalid:
            details["decision_reason"] = "GENERATOR_CITATION_INVALID"
            details["error"] = ",".join(draft.get("citation_issues", []))
        updates["execution_trace"] = _trace(trace_state, "generate", started, **details)
        return updates

    def finalize(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        draft = state.get("answer_draft")
        reason = state.get("stop_reason") or StopReason.INSUFFICIENT_EVIDENCE.value
        sources: list[dict[str, Any]] = []
        if isinstance(draft, dict):
            answer = str(draft.get("answer") or "").strip()
            sources = list(draft.get("sources") or [])
        elif draft is not None:
            answer = str(draft).strip()
        else:
            answer = ""

        if not answer:
            if reason in {StopReason.MAX_ITERATIONS.value, StopReason.MAX_TOOL_CALLS.value}:
                answer = "Không thể hoàn tất câu trả lời trong giới hạn thực thi đã cấu hình."
            elif reason == StopReason.TOOL_FAILURE.value:
                answer = "Không thể hoàn tất câu trả lời do công cụ truy xuất hoặc sinh câu trả lời gặp lỗi."
            else:
                answer = "Không tìm thấy đủ bằng chứng đáng tin cậy để trả lời câu hỏi này."

        updates: dict[str, Any] = {
            "final_answer": answer,
            "final_sources": sources,
            "stop_reason": reason,
            "next_action": Action.FINISH.value,
            "total_latency_ms": round((time.perf_counter() - state["started_at"]) * 1000, 3),
        }
        trace_state = dict(state)
        trace_state.update(updates)
        updates["execution_trace"] = _trace(
            trace_state,
            "finalize",
            started,
            action=Action.FINISH.value,
            stop_reason=reason,
        )
        return updates

    def route_action(state: AgentState) -> str:
        action = Action(state["next_action"])
        return action.value

    graph = StateGraph(AgentState)
    graph.add_node("initialize", initialize)
    graph.add_node("plan", plan)
    graph.add_node("retrieve", retrieve)
    graph.add_node("web_search", web_search_node)
    graph.add_node("verify_evidence", verify_evidence)
    graph.add_node("generate", generate)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "plan")
    graph.add_conditional_edges(
        "plan",
        route_action,
        {
            Action.RETRIEVE.value: "retrieve",
            Action.WEB_SEARCH.value: "web_search",
            Action.VERIFY.value: "verify_evidence",
            Action.GENERATE.value: "generate",
            Action.FINISH.value: "finalize",
        },
    )
    for action_node in ("retrieve", "web_search", "verify_evidence", "generate"):
        graph.add_edge(action_node, "plan")
    graph.add_edge("finalize", END)
    return graph.compile()


class AgenticRAG:
    """Small facade suitable for application code and evaluation harnesses."""

    def __init__(
        self,
        retriever: Any,
        generator: Any,
        web_search: Any | None = None,
        *,
        config: AgentConfig | None = None,
        verifier: Any | None = None,
        planner: Any | None = None,
    ):
        self.config = config or AgentConfig()
        self.graph = build_agentic_rag_graph(
            retriever,
            generator,
            web_search,
            config=self.config,
            verifier=verifier,
            planner=planner,
        )

    def invoke(self, query: str) -> AgentState:
        return self.graph.invoke(
            {"query": query},
            config={"recursion_limit": self.config.max_iterations * 3 + 10},
        )
