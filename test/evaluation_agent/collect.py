"""Collect both variants for a split; answers/references never reach runners."""

import argparse
import json
from pathlib import Path

from .scoring_data import DEFAULT_DATASET, PROJECT_ROOT, dataset_rows, digest, join_outputs, read_jsonl


def pipeline_fingerprint():
    """Portable source hash: do not resume outputs from a different pipeline."""
    paths = [*PROJECT_ROOT.joinpath("Sources/agentic_rag").glob("*.py"),
             *PROJECT_ROOT.joinpath("Sources/rag_core").glob("*.py"),
             *(Path(__file__).parent / name for name in ("collect.py", "config.py", "runners.py", "results.py"))]
    return digest({p.relative_to(PROJECT_ROOT).as_posix(): p.read_text(encoding="utf-8") for p in paths})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("pilot", "test"), default="pilot")
    parser.add_argument("--output", default="Data/test/results/pilot_outputs.jsonl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--provider", choices=("google", "openai"), help="Generator provider")
    parser.add_argument("--model", help="Generator model")
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--max-tokens", type=int, help="Generation token budget, including reasoning")
    args = parser.parse_args(argv)
    from .config import ComparisonConfig
    from .results import JsonlResultWriter
    from .runners import build_runners

    dataset = dataset_rows(args.dataset)
    path = Path(args.output).resolve()
    if path == Path(args.dataset).resolve():
        parser.error("Output must not be the dataset")
    meta_path = path.with_suffix(".run.json")
    config = ComparisonConfig.from_env(provider=args.provider, model=args.model,
                                       reasoning_effort=args.reasoning_effort, max_tokens=args.max_tokens)
    metadata = {"config": config.as_dict(), "dataset_sha256": digest(dataset), "split": args.split,
                "pipeline_sha256": pipeline_fingerprint()}
    seen = set()
    if path.exists() or meta_path.exists():
        if not args.resume:
            parser.error("Output already exists; use a new file or --resume")
        if not meta_path.exists() or json.loads(meta_path.read_text(encoding="utf-8")) != metadata:
            parser.error("Cannot resume a different pipeline/configuration/dataset; use a new output file")
        previous = read_jsonl(path) if path.exists() else []
        if previous:
            join_outputs(dataset, previous, split=args.split, require_complete=False)
        seen = {(r["query_id"], r["variant"]) for r in previous}
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with meta_path.open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
    selected = [r for r in dataset if r["split"] == args.split]
    if all((r["query_id"], v) in seen for r in selected for v in ("baseline", "agent")):
        print("All requested outputs already exist; no model calls.")
        return 0
    failures = 0
    with JsonlResultWriter(path) as writer:
        runners = build_runners(config)
        for index, row in enumerate(selected):
            order = ("baseline", "agent") if index % 2 == 0 else ("agent", "baseline")
            for variant in order:
                if (row["query_id"], variant) in seen:
                    continue
                runner = runners[variant]
                result = runner.run(row["query"])
                writer.append(result, query_id=row["query_id"], diagnostics=runner.diagnostics)
                print(f"{row['query_id']} {variant}: {result['status']}", flush=True)
                failures += result["status"] in {"error", "budget_exhausted"}
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
