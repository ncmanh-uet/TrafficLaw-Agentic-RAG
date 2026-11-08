"""Metric orchestration, explicit N/A policies, independent checkpoints."""

from __future__ import annotations

import copy
import json
import logging

from .citation import AnswerAnalysis, analyze_answer, citation_score, parse_answer, prepare_metric_text
from .scoring_data import METRICS, metric_result
from .judge import error_detail

LOGGER = logging.getLogger(__name__)


class Scorer:
    def __init__(self, judge, ragas_metrics):
        self.judge = judge
        self.ragas_metrics = ragas_metrics

    async def score(self, gold, output, *, previous=None, checkpoint=None, diagnostics=False, metrics=METRICS):
        if not metrics or len(set(metrics)) != len(metrics) or any(m not in METRICS for m in metrics):
            raise ValueError("Unknown/duplicate/empty metric selection")
        row = copy.deepcopy(previous) if previous else {
            "query_id": gold["query_id"], "variant": output["variant"], "metrics": {},
            "review_required": gold["category"] == "insufficient_evidence"}

        def save():
            if checkpoint:
                checkpoint(copy.deepcopy(row))

        pending = [m for m in metrics if row["metrics"].get(m, {}).get("status") not in {"ok", "na", "input_error"}]
        if not pending:
            return row
        if output["status"] in {"error", "budget_exhausted"}:
            for name in pending:
                row["metrics"][name] = metric_result(status="input_error", reason=output["status"])
            save()
            return row

        blocks = parse_answer(output["answer"])
        analysis = AnswerAnalysis.model_validate(row["analysis"]) if row.get("analysis") else None
        if analysis is not None:
            row.pop("analysis_error", None)
            row.pop("analysis_error_detail", None)
        if output["answer"].strip() and analysis is None:
            try:
                # Needed for refusal/content distinction even if citations are absent.
                analysis = await analyze_answer(self.judge, gold["query"], blocks)
                row.pop("analysis_error", None)
                row.pop("analysis_error_detail", None)
                if diagnostics:
                    row["analysis"] = analysis.model_dump()
            except Exception as exc:
                row["analysis_error"] = type(exc).__name__
                row["analysis_error_detail"] = error_detail(exc)
                LOGGER.warning("%s %s claim analysis: %s", gold["query_id"], output["variant"],
                               json.dumps(row["analysis_error_detail"], ensure_ascii=False))
        has_claims = bool(analysis.claims) if analysis else False
        expected_refusal = gold["category"] == "insufficient_evidence"
        refused = output["status"] == "refused" or not output["answer"].strip() or bool(analysis and analysis.abstaining)
        row["review_required"] = expected_refusal or refused
        if expected_refusal:
            row["behavior"] = "needs_abstention_review"
        elif refused:
            row["behavior"] = "unexpected_abstention"
        else:
            row["behavior"] = "answered"
        api_failure = row.get("analysis_error_detail")
        if not api_failure or api_failure.get("http_code") not in {400, 401, 403, 404, 429}:
            api_failure = None
        for name in pending:
            detail_key = {"factual_correctness": "factual_details", "faithfulness": "faithfulness_details"}.get(name)
            if detail_key:
                row.pop(detail_key, None)
            try:
                if name == "factual_correctness" and expected_refusal:
                    result = metric_result(status="na", reason="expected_abstention_manual_review")
                elif name == "factual_correctness" and refused:
                    result = metric_result(0.0, reason="unexpected_abstention")
                elif api_failure:
                    reason = "claim_analysis_failed" if analysis is None and name != "factual_correctness" else "api_unavailable"
                    result = metric_result(status="judge_error", reason=reason)
                    result["error"] = copy.deepcopy(api_failure)
                elif name == "factual_correctness":
                    metric = self.ragas_metrics[name]
                    inputs = {"user_input": gold["query"],
                              "response": prepare_metric_text(output["answer"], gold["query"], metric=name),
                              "reference": prepare_metric_text(gold["reference_answer"], gold["query"], metric=name)}
                    if diagnostics and hasattr(metric, "ascore_with_details"):
                        row["factual_details"] = {}
                        score = await metric.ascore_with_details(
                            details=row["factual_details"], on_step=save, **inputs)
                    else:
                        score = await metric.ascore(**inputs)
                    result = metric_result(float(score.value))
                elif analysis is None and output["answer"].strip():
                    result = metric_result(status="judge_error", reason="claim_analysis_failed")
                elif name == "faithfulness" and not has_claims:
                    result = metric_result(status="na", reason="no_checkable_claims")
                elif name == "faithfulness" and not output["contexts"]:
                    result = metric_result(0.0, reason="claims_without_context")
                elif name == "faithfulness":
                    metric = self.ragas_metrics[name]
                    inputs = {"user_input": gold["query"],
                              "response": prepare_metric_text(output["answer"], gold["query"], metric=name),
                              "retrieved_contexts": output["contexts"],
                              "evidence_sources": output["sources"]}
                    if diagnostics and hasattr(metric, "ascore_with_details"):
                        row["faithfulness_details"] = {}
                        score = await metric.ascore_with_details(
                            details=row["faithfulness_details"], on_step=save, **inputs)
                    else:
                        score = await metric.ascore(**inputs)
                    result = metric_result(float(score.value))
                else:
                    # Empty answer: no claims/citations, without calling a judge.
                    if analysis is None:
                        analysis = AnswerAnalysis(abstaining=True, claims=[])
                    score = await citation_score(self.judge, blocks, analysis,
                                                 output["contexts"], output["sources"])
                    result = metric_result(score["score"], "na" if score["score"] is None else "ok",
                                           "no_claims_or_citations" if score["score"] is None else None)
                    if diagnostics:
                        row["citation_details"] = score
                row["metrics"][name] = result
            except Exception as exc:
                row["metrics"][name] = metric_result(status="judge_error", reason=type(exc).__name__)
                row["metrics"][name]["error"] = error_detail(exc)
                if row["metrics"][name]["error"].get("http_code") in {400, 401, 403, 404, 429}:
                    api_failure = row["metrics"][name]["error"]
                LOGGER.warning("%s %s %s: %s", gold["query_id"], output["variant"], name,
                               json.dumps(row["metrics"][name]["error"], ensure_ascii=False))
            save()  # A failed later metric cannot discard completed paid judgments.
        return row
