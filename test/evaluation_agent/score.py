"""Score saved baseline/agent outputs without running the system again."""

from __future__ import annotations

import argparse
import asyncio
from importlib.metadata import version
import json
import os
from pathlib import Path

from .judge import GeminiJudge, OpenAIJudge, JudgeConfig, make_ragas_metrics
from .scoring_data import (
    DEFAULT_DATASET, METRICS, PROJECT_ROOT, checkpoint_rows, dataset_rows, digest, join_outputs, read_jsonl,
)


def atomic_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    # Exclusive creation prevents two scorers from sharing a temporary file.
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


async def run(args):
    from .scoring import Scorer

    dataset = dataset_rows(args.dataset)
    outputs = read_jsonl(args.input)
    pairs = join_outputs(dataset, outputs, split=args.split, require_complete=not args.allow_partial)
    include_details = getattr(args, "include_details", False)
    if args.check_only:
        print(f"Validated {len(pairs)} {args.split} outputs; no model calls.")
        return 0
    path = Path(args.output).resolve()
    protected = {Path(args.input).resolve(), Path(args.dataset).resolve()}
    if path.suffix != ".jsonl" or path in protected:
        raise ValueError("Scores must be a separate .jsonl file, not the dataset/input")
    config = JudgeConfig(model=args.model, provider=getattr(args, "provider", "google"),
                         reasoning_effort=getattr(args, "reasoning_effort", "medium"),
                         max_output_tokens=getattr(args, "max_output_tokens", 8192))
    metadata = {"judge": config.as_dict(), "ragas_version": version("ragas"),
                "google_genai_version": version("google-genai"),
                "dependencies": {name: version(name) for name in (
                    "langchain", "langchain-core", "langchain-community", "langchain-openai", "pydantic",
                    "datasets", "pyarrow")},
                "dataset_sha256": digest(dataset), "outputs_sha256": digest(outputs),
                "split": args.split, "allow_partial": args.allow_partial,
                "include_details": include_details,
                "scorer_sha256": digest({p.name: p.read_text(encoding="utf-8") for p in
                    Path(__file__).parent.glob("*.py")})}
    if config.provider == "openai":
        metadata["openai_version"] = version("openai")
    meta_path = path.with_suffix(".meta.json")
    valid_keys = {(g["query_id"], o["variant"]) for g, o in pairs}
    state = {}
    if path.exists() or meta_path.exists():
        if not args.resume:
            raise ValueError("Output/metadata already exists: use --resume or a new output path")
        if not meta_path.exists() or json.loads(meta_path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Resume refused: judge, code or input differs from this scoring run")
        state = checkpoint_rows(path, valid_keys) if path.exists() else {}
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with meta_path.open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

    def checkpoint(row):
        state[(row["query_id"], row["variant"])] = row
        atomic_jsonl(path, list(state.values()))

    judge = OpenAIJudge(config) if config.provider == "openai" else GeminiJudge(config)
    try:
        scorer = Scorer(judge, make_ragas_metrics(judge))
        for gold, output in pairs:
            key = (gold["query_id"], output["variant"])
            row = await scorer.score(gold, output, previous=state.get(key),
                                     checkpoint=checkpoint, diagnostics=include_details)
            print(f"{key[0]} {key[1]}: " + ", ".join(
                f"{m}={row['metrics'][m]['score']} ({row['metrics'][m]['status']})" for m in METRICS),
                flush=True)
            errors = [row.get("analysis_error_detail", {})] + [
                result.get("error", {}) for result in row["metrics"].values()]
            blocked = next((error.get("http_code") for error in errors
                            if error.get("http_code") in {400, 401, 403, 404, 429}), None)
            if blocked is not None:
                print(f"Stopped after {key[0]} {key[1]}: {config.provider} HTTP {blocked}. "
                      "Progress saved; check error details before retrying with --resume.", flush=True)
                break
    finally:
        await judge.close()
    failures = sum(r["status"] in {"judge_error", "input_error"}
                   for row in state.values() for r in row["metrics"].values())
    pending = len(pairs) * len(METRICS) - sum(len(row["metrics"]) for row in state.values())
    print(f"Saved {len(state)} rows to {path}; {failures} metric errors; {pending} pending metrics. "
          "Review flags are not scores.")
    return int(bool(failures or pending))


def main(argv=None):
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSONL produced by baseline/agent runners")
    parser.add_argument("--output", default="Data/test/results/pilot_scores.jsonl", help="Score checkpoint file")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("pilot", "test"), default="pilot")
    parser.add_argument("--provider", choices=("google", "openai"), default=os.getenv("EVALUATOR_PROVIDER") or "google")
    parser.add_argument("--model", default=os.getenv("EVALUATOR_MODEL"))
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh", "max"),
                        default=os.getenv("EVALUATOR_REASONING_EFFORT") or "medium")
    parser.add_argument("--max-output-tokens", type=int, default=8192, help="Judge token budget, including reasoning")
    parser.add_argument("--resume", action="store_true", help="Keep successful metrics; retry judge errors only")
    parser.add_argument("--include-details", action="store_true", help="Save factual and faithfulness claim/verdict traces, citation claims and P/R")
    parser.add_argument("--check-only", action="store_true", help="Validate inputs without paid calls or output writes")
    parser.add_argument("--allow-partial", action="store_true", help="Diagnostic only; do not require both variants for every selected question")
    args = parser.parse_args(argv)
    args.model = args.model or ("gpt-6-luna" if args.provider == "openai" else "gemini-3.8-flash")
    try:
        if args.check_only:
            pairs = join_outputs(dataset_rows(args.dataset), read_jsonl(args.input),
                                 split=args.split, require_complete=not args.allow_partial)
            print(f"Validated {len(pairs)} {args.split} outputs; no model calls.")
            return 0
        return asyncio.run(run(args))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
