"""Replaceable deterministic controller for the bounded action space."""

from __future__ import annotations

from dataclasses import dataclass

from .config import AgentConfig
from .state import Action, AgentState, StopReason


@dataclass(frozen=True)
class PlannerDecision:
    action: Action
    reason: str
    stop_reason: StopReason | None = None


class DeterministicPlanner:
    """Select the next allowed action solely from observable agent state."""

    def decide(
        self,
        state: AgentState,
        config: AgentConfig,
        *,
        web_search_available: bool,
    ) -> PlannerDecision:
        iteration = state.get("iteration", 0) + 1
        if iteration > config.max_iterations:
            return PlannerDecision(Action.FINISH, "ITERATION_BUDGET_EXHAUSTED", StopReason.MAX_ITERATIONS)

        draft = state.get("answer_draft")
        if draft is not None:
            refused = isinstance(draft, dict) and bool(draft.get("refused"))
            reason = StopReason.INSUFFICIENT_EVIDENCE if refused else StopReason.SUCCESS
            return PlannerDecision(Action.FINISH, "ANSWER_REFUSED" if refused else "ANSWER_AVAILABLE", reason)

        if not state.get("retrieval_attempted"):
            if state.get("tool_calls", 0) >= config.max_tool_calls:
                return PlannerDecision(Action.FINISH, "TOOL_BUDGET_EXHAUSTED", StopReason.MAX_TOOL_CALLS)
            return PlannerDecision(Action.RETRIEVE, "INTERNAL_RETRIEVAL_NOT_ATTEMPTED")

        if state.get("evidence_revision", 0) != state.get("verified_revision", -1):
            return PlannerDecision(Action.VERIFY, "EVIDENCE_CHANGED")

        if state.get("generation_attempted"):
            generation_issue = next(
                (
                    issue
                    for issue in ("GENERATOR_REFUSED", "GENERATOR_CITATION_INVALID")
                    if issue in state.get("verification_issues", [])
                ),
                None,
            )
            if generation_issue:
                if (
                    web_search_available
                    and state.get("web_searches", 0) < config.max_web_searches
                    and state.get("tool_calls", 0) < config.max_tool_calls
                ):
                    return PlannerDecision(Action.WEB_SEARCH, f"{generation_issue}_WEB_RECOVERY")
                return PlannerDecision(
                    Action.FINISH,
                    f"{generation_issue}_NO_RECOVERY_AVAILABLE",
                    StopReason.INSUFFICIENT_EVIDENCE,
                )
            return PlannerDecision(Action.FINISH, "GENERATION_FAILED", StopReason.TOOL_FAILURE)

        if state.get("evidence_sufficient") is True:
            if state.get("tool_calls", 0) >= config.max_tool_calls:
                return PlannerDecision(Action.FINISH, "TOOL_BUDGET_EXHAUSTED", StopReason.MAX_TOOL_CALLS)
            return PlannerDecision(Action.GENERATE, "EVIDENCE_SUFFICIENT")

        if (
            web_search_available
            and state.get("web_searches", 0) < config.max_web_searches
            and state.get("tool_calls", 0) < config.max_tool_calls
        ):
            return PlannerDecision(Action.WEB_SEARCH, "EVIDENCE_INSUFFICIENT_WEB_AVAILABLE")

        if state.get("tool_calls", 0) >= config.max_tool_calls:
            stop_reason = StopReason.MAX_TOOL_CALLS
        elif state.get("errors") and not state.get("evidence"):
            stop_reason = StopReason.TOOL_FAILURE
        else:
            stop_reason = StopReason.INSUFFICIENT_EVIDENCE
        return PlannerDecision(Action.FINISH, "NO_RECOVERY_ACTION_AVAILABLE", stop_reason)
