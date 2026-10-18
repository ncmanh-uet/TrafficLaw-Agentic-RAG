"""Tunable execution budgets and verifier thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentConfig:
    # A normal internal path uses 4 planner passes; a Tavily fallback uses 6.
    # Eight permits one recovery decision while still bounding every path.
    max_iterations: int = 8
    # Retriever + Tavily + Generator normally uses at most three calls. Four
    # leaves room for a future bounded recovery tool without making loops open.
    max_tool_calls: int = 4
    # One official-source search is the conservative default for cost and for
    # measuring unnecessary fallback. It remains configurable for experiments.
    max_web_searches: int = 1

    retrieval_top_k: int = 10
    # The second pass uses corpus language extracted from the first result set
    # (pseudo-relevance feedback); it is not a query-specific rewrite rule.
    max_retrieval_queries: int = 2
    web_max_results: int = 5
    # Sibling enrichment can expand a small retrieval set substantially. Keep
    # all evidence in state for audit, but bound the context sent to the LLM.
    max_generation_evidence: int = 30
    min_relevant_evidence: int = 1
    min_relevance_score: float = 0.02
    min_query_coverage: float = 0.25
    web_search_depth: str = "advanced"
    preferred_web_domains: tuple[str, ...] = field(
        default_factory=lambda: (
            "vanban.chinhphu.vn",
            "xaydungchinhsach.chinhphu.vn",
            "chinhphu.vn",
            "bocongan.gov.vn",
            "csgt.vn",
            "moj.gov.vn",
        )
    )

    def __post_init__(self) -> None:
        integer_fields = {
            "max_iterations": self.max_iterations,
            "max_tool_calls": self.max_tool_calls,
            "max_web_searches": self.max_web_searches,
            "retrieval_top_k": self.retrieval_top_k,
            "max_retrieval_queries": self.max_retrieval_queries,
            "web_max_results": self.web_max_results,
            "max_generation_evidence": self.max_generation_evidence,
            "min_relevant_evidence": self.min_relevant_evidence,
        }
        for name, value in integer_fields.items():
            minimum = 0 if name == "max_web_searches" else 1
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        if not 0.0 <= self.min_query_coverage <= 1.0:
            raise ValueError("min_query_coverage must be between 0 and 1")
