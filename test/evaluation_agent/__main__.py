"""Run and save one query through the baseline or agent benchmark path."""

import argparse
import json
import sys

from .config import ComparisonConfig
from .runners import build_runner
from .results import DEFAULT_OUTPUT, JsonlResultWriter


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("baseline", "agent"), nargs="?")
    parser.add_argument("query", nargs="?")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Append results to this JSONL file")
    parser.add_argument("--query-id", help="Dataset query ID, e.g. q_001; optional for ad-hoc queries")
    parser.add_argument("--include-trace", action="store_true", help="Also save generation attempts and execution trace")
    parser.add_argument("--show-config", action="store_true", help="Print shared settings; no service/model calls")
    parser.add_argument("--provider", choices=("google", "openai", "anthropic"), help="Generator provider; overrides GENERATOR_PROVIDER")
    parser.add_argument("--model", help="Generator model; overrides GENERATOR_MODEL")
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--max-tokens", type=int, help="Generation token budget, including reasoning")
    args = parser.parse_args(argv)
    if not args.show_config and (not args.variant or not args.query or not args.query.strip()):
        parser.error("provide baseline/agent and a non-empty query, or --show-config")
    if args.query_id is not None and not args.query_id.strip():
        parser.error("--query-id cannot be empty")
    config = ComparisonConfig.from_env(provider=args.provider, model=args.model,
                                       reasoning_effort=args.reasoning_effort, max_tokens=args.max_tokens)
    if args.show_config:
        print(json.dumps(config.as_dict(), ensure_ascii=False, indent=2))
        return 0
    with JsonlResultWriter(args.output) as writer:
        runner = build_runner(args.variant, config)
        result = runner.run(args.query)
        writer.append(result, query_id=args.query_id,
                      diagnostics=runner.diagnostics if args.include_trace else None)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved result to {writer.path.resolve()}", file=sys.stderr)
    return 1 if result["status"] in {"error", "budget_exhausted"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
