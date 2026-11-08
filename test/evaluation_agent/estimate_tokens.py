"""Offline GPT token budget for the existing three-metric scoring pipeline.

Run the real scorer/prompts with a recording judge; never instantiate a client.
Synthetic decompositions approximate future judge decisions, not metric scores.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path

from .citation import parse_answer
from .judge import JUDGE_SYSTEM_INSTRUCTION, PROMPT_VERSION, make_ragas_metrics
from .scoring import Scorer
from .scoring_data import DEFAULT_DATASET, PROJECT_ROOT, dataset_rows, digest, join_outputs, read_jsonl


def approximate_claims(text):
    """One claim per non-heading block; this is a budgeting assumption only."""
    return [b.text for b in parse_answer(text)
            if b.text and not b.text.endswith(":")
            and b.text.strip(" *#").casefold() not in {"nguồn", "tài liệu tham khảo"}]


def prompt_data(prompt):
    # Ragas puts the current input after all examples. Custom citation prompts
    # append a JSON object without examples; do not parse quoted nested braces.
    if "\ninput: " in prompt:
        return json.JSONDecoder().raw_decode(prompt.rsplit("\ninput: ", 1)[1])[0]
    for marker in ('{"query":', '{"claims":'):
        if marker in prompt:
            return json.JSONDecoder().raw_decode(prompt[prompt.index(marker):])[0]
    raise ValueError("Unrecognized judge prompt; update the estimator for this prompt version")


class RecordingJudge:
    """Capture real request text and return schema-valid budgeting placeholders."""

    def __init__(self, encoding, *, framing_tokens=12):
        self.encoding = encoding
        self.framing_tokens = framing_tokens
        self.calls = []
        self.output = None
        self.saved = {}
        self.trace_steps = {}

    def count(self, text):
        return len(self.encoding.encode(text, disallowed_special=()))

    def begin(self, output, saved=None):
        self.output = output
        self.saved = saved or {}
        self.trace_steps = {
            group: iter(self.saved.get(group + "_details", {}).get("steps", []))
            for group in ("factual", "faithfulness")}

    async def generate(self, prompt, response_model):
        data = prompt_data(prompt)
        name = response_model.__name__
        if name == "AnswerAnalysis":
            group = "shared_analysis"
            reply = self.saved.get("analysis") or {
                "abstaining": self.output["status"] == "refused",
                "claims": [{"block_id": b["block_id"], "text": b["text"]}
                           for b in data["blocks"] if approximate_claims(b["text"])]}
        elif name == "SupportVerdicts":
            group = "citation_f1"
            reply = {"judgments": [{"claim_id": c["claim_id"], "fully_supported": False,
                                   "contributing_source_ids": [], "reason": "Budget placeholder"}
                                  for c in data["claims"]]}
        elif name == "ClaimDecompositionOutput":
            group = "factual_correctness"
            reply = {"claims": approximate_claims(data["response"])}
        elif name == "StatementGeneratorOutput":
            group = "faithfulness"
            reply = {"statements": approximate_claims(data["answer"])}
        elif name in {"NLIStatementOutput", "FaithfulnessNLIOutput"}:
            group = "faithfulness" if name == "FaithfulnessNLIOutput" else "factual_correctness"
            reply = {"statements": [dict(statement=s, reason="Budget placeholder", verdict=0,
                **({"decision": "not_found", "evidence_ids": [], "rule_links": []}
                   if group == "faithfulness" else {})) for s in data["statements"]]}
        else:
            raise ValueError(f"Unsupported response model: {name}")

        decomposition_source = "not_applicable"
        if group in {"factual_correctness", "faithfulness"}:
            trace_group = "factual" if group == "factual_correctness" else group
            step = next(self.trace_steps[trace_group], {})
            if name in {"ClaimDecompositionOutput", "StatementGeneratorOutput"}:
                field = "claims" if name == "ClaimDecompositionOutput" else "statements"
                observed = step.get("output", {}).get(field)
                if isinstance(observed, list) and all(isinstance(s, str) for s in observed):
                    reply = {field: observed}
                    decomposition_source = "saved_judge_trace"
                else:
                    decomposition_source = "heuristic_blocks"

        result = response_model.model_validate(reply)
        schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False, separators=(",", ":"))
        prompt_tokens = self.count(prompt)
        system_tokens = self.count(JUDGE_SYSTEM_INSTRUCTION)
        schema_tokens = self.count(schema)
        self.calls.append({"query_id": self.output["query_id"], "variant": self.output["variant"],
            "group": group, "response_model": name, "decomposition_source": decomposition_source,
            "prompt_tokens": prompt_tokens, "system_tokens": system_tokens,
            "schema_tokens": schema_tokens, "framing_tokens": self.framing_tokens,
            "input_tokens_estimate": prompt_tokens + system_tokens + schema_tokens + self.framing_tokens})
        return result


def summarize(calls):
    fields = ("prompt_tokens", "system_tokens", "schema_tokens", "framing_tokens", "input_tokens_estimate")
    return {"calls": len(calls), **{k: sum(c[k] for c in calls) for k in fields}}


def load_saved_scores(path, dataset, outputs):
    if not path:
        return {}
    path = Path(path)
    metadata = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if (metadata.get("dataset_sha256") != digest(dataset)
            or metadata.get("outputs_sha256") != digest(outputs)
            or metadata.get("judge", {}).get("prompt_version") != PROMPT_VERSION):
        raise ValueError("Saved traces do not match the dataset, outputs and prompt version")
    rows = read_jsonl(path)
    saved = {(r["query_id"], r["variant"]): r for r in rows}
    if len(saved) != len(rows):
        raise ValueError("Duplicate saved score rows")
    return saved


async def estimate(pairs, encoding, *, saved=None, framing_tokens=12):
    judge = RecordingJudge(encoding, framing_tokens=framing_tokens)
    scorer = Scorer(judge, make_ragas_metrics(judge))
    row_counts = []
    for gold, output in pairs:
        judge.begin(output, (saved or {}).get((output["query_id"], output["variant"])))
        start = len(judge.calls)
        # These placeholder scores are deliberately discarded, never saved.
        result = await scorer.score(gold, output)
        errors = {k: v for k, v in result["metrics"].items() if v["status"] == "judge_error"}
        if errors:
            raise ValueError(f"Offline simulation failed for {output['query_id']}/{output['variant']}: {errors}")
        row_counts.append({"query_id": output["query_id"], "variant": output["variant"],
                           "status": output["status"], **summarize(judge.calls[start:])})
    groups = ("shared_analysis", "factual_correctness", "faithfulness", "citation_f1")
    return {"totals": summarize(judge.calls),
        "by_group": {g: summarize([c for c in judge.calls if c["group"] == g]) for g in groups},
        "by_variant": {v: summarize([c for c in judge.calls if c["variant"] == v]) for v in ("baseline", "agent")},
        "decomposition_sources": dict(Counter(c["decomposition_source"] for c in judge.calls
                                              if c["decomposition_source"] != "not_applicable")),
        "per_output": row_counts, "per_call": judge.calls}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(PROJECT_ROOT / "Data/test/results/test_outputs_pipeline_v2.jsonl"))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--split", choices=("pilot", "test"), default="test")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--model", default="gpt-5", help="Tokenizer model only; does not change the Gemini scorer")
    parser.add_argument("--encoding", help="Explicit tiktoken encoding for an unmapped model; no silent fallback")
    parser.add_argument("--scores", help="Optional matching score file with analysis/decomposition traces")
    parser.add_argument("--output", help="New JSON token report (never a metric score file)")
    parser.add_argument("--framing-tokens", type=int, default=12, help="Assumed API framing overhead per call")
    parser.add_argument("--output-tokens-per-call", type=int, nargs="+", default=[512, 1024, 2048],
                        help="Budget scenarios for ALL generated tokens, including hidden reasoning")
    args = parser.parse_args(argv)
    if args.framing_tokens < 0 or any(n < 0 for n in args.output_tokens_per_call):
        parser.error("Token budgets must be nonnegative")
    dataset, outputs = dataset_rows(args.dataset), read_jsonl(args.input)
    pairs = join_outputs(dataset, outputs, split=args.split, require_complete=not args.allow_partial)
    saved = load_saved_scores(args.scores, dataset, outputs)
    output_path = Path(args.output).resolve() if args.output else None
    if output_path and (output_path.suffix != ".json" or output_path.exists() or
                        output_path in {Path(args.input).resolve(), Path(args.dataset).resolve()}):
        parser.error("--output must be a new .json file, separate from inputs")
    import tiktoken
    try:
        encoding = tiktoken.get_encoding(args.encoding) if args.encoding else tiktoken.encoding_for_model(args.model)
    except KeyError:
        parser.error("Model has no known tiktoken mapping; specify its documented --encoding explicitly")
    report = asyncio.run(estimate(pairs, encoding, saved=saved, framing_tokens=args.framing_tokens))
    report["metadata"] = {"input": str(Path(args.input).resolve()), "dataset": str(Path(args.dataset).resolve()),
        "split": args.split, "outputs": len(pairs), "questions": len({g["query_id"] for g, _ in pairs}),
        "tokenizer_model": args.model, "encoding": encoding.name, "tiktoken_version": version("tiktoken"),
        "ragas_version": version("ragas"), "prompt_version": PROMPT_VERSION,
        "dataset_sha256": digest(dataset), "outputs_sha256": digest(outputs),
        "saved_trace_rows": sum((o["query_id"], o["variant"]) in saved for _, o in pairs),
        "assumptions": [
            "Offline simulation of the current three-metric scorer; no judge API calls or metric scores.",
            "Same text prompts/system instruction and external JSON schema assumed for GPT transport.",
            "Use saved decompositions/analysis where available; otherwise one claim per non-heading block.",
            "Refusal/claim decisions can differ with a real GPT judge, changing downstream calls.",
            "Input includes actual prompt text, compact external schema, system text and assumed API framing.",
            "JSON schema serialization and API framing are estimates, not authoritative API usage.",
            "Output scenarios are allowances per call including reasoning, not predicted or maximum usage.",
            "One attempt per call; no retry multiplier, caching discount, retrieval or generation included.",
            "The scorer supports Google/OpenAI; this command's --model selects a tokenizer only."]}
    totals = report["totals"]
    report["output_budget_scenarios"] = [{"generated_tokens_per_call": n,
        "generated_tokens_budget": totals["calls"] * n,
        "total_tokens_budget": totals["input_tokens_estimate"] + totals["calls"] * n}
        for n in args.output_tokens_per_call]
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    print(f"{len(pairs)} outputs; {encoding.name}; {totals['calls']:,} estimated judge calls")
    print(f"Input tokens estimate: {totals['input_tokens_estimate']:,}")
    for group, counts in report["by_group"].items():
        print(f"  {group}: {counts['calls']:,} calls; {counts['input_tokens_estimate']:,} input tokens")
    for scenario in report["output_budget_scenarios"]:
        print(f"Generated budget {scenario['generated_tokens_per_call']:,}/call (including reasoning): "
              f"{scenario['total_tokens_budget']:,} total tokens")
    print("Approximation only: intermediate claims, API framing, reasoning and retries can change usage.")
    if output_path:
        print(f"Report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
