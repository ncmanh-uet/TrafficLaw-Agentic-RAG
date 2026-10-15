"""Benchmark hybrid retrieval against cross-encoder reranking.

Run from the repository root, for example::

    python test/evaluation/benchmark_reranker.py --limit 10

The benchmark keeps one fixed dense/BM25 retrieval pool for every experiment,
then varies only whether reranking is enabled and how many fused candidates are
sent to it. Model loading and per-configuration warm-up calls are not timed.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Sources.rag_core.reranker import DEFAULT_RERANKER_MODEL
from Sources.rag_core.retriever import (
    DEFAULT_COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_JSONL,
    TrafficHybridRetriever,
)
from test.evaluation.metrics import calculate_recall_at_k


LOGGER = logging.getLogger("reranker_benchmark")
DEFAULT_DATASET = PROJECT_ROOT / "Data" / "test" / "evaluation.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    enable_reranker: bool
    candidate_k: int | None


def load_test_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load and strictly validate the JSONL evaluation dataset."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")
    if not dataset_path.is_file():
        raise ValueError(f"Evaluation dataset is not a file: {dataset_path}")

    samples: list[dict[str, Any]] = []
    seen_query_ids: set[str] = set()
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {dataset_path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(sample, dict):
                raise ValueError(
                    f"Expected an object at {dataset_path}:{line_number}"
                )

            query_id = sample.get("query_id")
            query = sample.get("query")
            relevant = sample.get("relevant_chunks")
            if not isinstance(query_id, str) or not query_id.strip():
                raise ValueError(
                    f"Missing or invalid query_id at {dataset_path}:{line_number}"
                )
            if query_id in seen_query_ids:
                raise ValueError(f"Duplicate query_id {query_id!r} at line {line_number}")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(
                    f"Missing or invalid query for {query_id!r} at line {line_number}"
                )
            if not isinstance(relevant, list) or not all(
                isinstance(chunk_id, str) and chunk_id for chunk_id in relevant
            ):
                raise ValueError(
                    f"Missing or invalid relevant_chunks for {query_id!r} "
                    f"at line {line_number}"
                )

            seen_query_ids.add(query_id)
            samples.append(sample)

    if not samples:
        raise ValueError(f"Evaluation dataset is empty: {dataset_path}")
    return samples


def percentile(values: list[float], percentile_value: float) -> float:
    """Calculate a linearly interpolated percentile without adding NumPy."""
    if not values:
        raise ValueError("Cannot calculate a percentile for an empty sequence")
    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile_value must be between 0 and 100")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value / 100
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + (
        ordered[upper_index] - ordered[lower_index]
    ) * fraction


def latency_stats(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize empty latency measurements")
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "min": min(values),
        "max": max(values),
        "std": statistics.pstdev(values),
    }


def _retrieved_chunk_ids(chunks: list[Any], final_k: int) -> list[str]:
    chunk_ids: list[str] = []
    for rank, chunk in enumerate(chunks[:final_k], start=1):
        chunk_id = chunk.metadata.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError(
                f"Retrieved chunk at rank {rank} has no valid metadata['chunk_id']"
            )
        chunk_ids.append(chunk_id)
    return chunk_ids


def warm_up(
    retriever: TrafficHybridRetriever,
    samples: list[dict[str, Any]],
    experiments: list[ExperimentConfig],
    final_k: int,
    warmup_count: int,
) -> None:
    """Warm every path without retaining timings or results."""
    if warmup_count == 0:
        return
    warmup_samples = samples[: min(warmup_count, len(samples))]
    LOGGER.info(
        "Warming %d query(s) for each of %d configuration(s)",
        len(warmup_samples),
        len(experiments),
    )
    for experiment in experiments:
        for sample in warmup_samples:
            retriever.get_relevant_chunks(
                sample["query"],
                top_k=final_k,
                candidate_k=experiment.candidate_k,
                enable_reranker=experiment.enable_reranker,
            )


def run_experiment(
    retriever: TrafficHybridRetriever,
    samples: list[dict[str, Any]],
    experiment: ExperimentConfig,
    final_k: int,
) -> list[dict[str, Any]]:
    """Run one configuration over the fixed dataset order."""
    results: list[dict[str, Any]] = []
    recall_sum = 0.0
    latency_sum = 0.0

    progress = tqdm(samples, desc=f"Experiment: {experiment.name}", unit="query")
    for sample in progress:
        diagnostics: dict[str, float | int | bool | None] = {}
        start = time.perf_counter()
        try:
            chunks = retriever.get_relevant_chunks(
                sample["query"],
                top_k=final_k,
                candidate_k=experiment.candidate_k,
                enable_reranker=experiment.enable_reranker,
                diagnostics=diagnostics,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Retrieval failed for experiment {experiment.name!r}, "
                f"query_id {sample['query_id']!r}"
            ) from exc
        retrieval_latency_ms = (time.perf_counter() - start) * 1000

        retrieved_ids = _retrieved_chunk_ids(chunks, final_k)
        relevant_ids = sample["relevant_chunks"]
        recall = calculate_recall_at_k(relevant_ids, retrieved_ids, final_k)
        relevant_set = set(relevant_ids)
        retrieved_relevant_ids = [
            chunk_id for chunk_id in retrieved_ids if chunk_id in relevant_set
        ]

        result = {
            "query_id": sample["query_id"],
            "query": sample["query"],
            "experiment": experiment.name,
            "reranker_enabled": experiment.enable_reranker,
            "candidate_k": experiment.candidate_k,
            "actual_candidate_count": diagnostics.get("candidate_count"),
            "final_k": final_k,
            "recall_at_10": recall,
            "retrieval_latency_ms": retrieval_latency_ms,
            "reranker_latency_ms": diagnostics.get("reranker_latency_ms"),
            "relevant_chunk_ids": relevant_ids,
            "retrieved_chunk_ids": retrieved_ids,
            "retrieved_relevant_chunk_ids": retrieved_relevant_ids,
        }
        results.append(result)
        recall_sum += recall
        latency_sum += retrieval_latency_ms
        progress.set_postfix(
            recall=f"{recall_sum / len(results):.3f}",
            mean_ms=f"{latency_sum / len(results):.1f}",
        )
    return results


def summarize_experiments(
    results_by_experiment: dict[str, list[dict[str, Any]]],
    experiments: list[ExperimentConfig],
    final_k: int,
    retrieval_pool_size: int,
) -> list[dict[str, Any]]:
    baseline_name = experiments[0].name
    baseline_results = results_by_experiment[baseline_name]
    baseline_by_query = {row["query_id"]: row for row in baseline_results}
    baseline_latencies = [row["retrieval_latency_ms"] for row in baseline_results]
    baseline_latency = latency_stats(baseline_latencies)
    baseline_recall = statistics.fmean(
        row["recall_at_10"] for row in baseline_results
    )

    summaries: list[dict[str, Any]] = []
    for experiment in experiments:
        rows = results_by_experiment[experiment.name]
        recalls = [row["recall_at_10"] for row in rows]
        retrieval_latency = latency_stats(
            [row["retrieval_latency_ms"] for row in rows]
        )
        reranker_values = [
            row["reranker_latency_ms"]
            for row in rows
            if row["reranker_latency_ms"] is not None
        ]
        reranker_latency = latency_stats(reranker_values) if reranker_values else None
        recall_at_10 = statistics.fmean(recalls)

        improved: list[str] = []
        unchanged: list[str] = []
        degraded: list[str] = []
        for row in rows:
            baseline_row = baseline_by_query[row["query_id"]]
            delta = row["recall_at_10"] - baseline_row["recall_at_10"]
            if delta > 0:
                status = "improved"
                improved.append(row["query_id"])
            elif delta < 0:
                status = "degraded"
                degraded.append(row["query_id"])
            else:
                status = "unchanged"
                unchanged.append(row["query_id"])
            row["baseline_recall_at_10"] = baseline_row["recall_at_10"]
            row["delta_recall_at_10"] = delta
            row["comparison_to_baseline"] = status

        mean_increase_percent = (
            (retrieval_latency["mean"] - baseline_latency["mean"])
            / baseline_latency["mean"]
            * 100
            if baseline_latency["mean"] > 0
            else None
        )
        actual_candidate_counts = [
            int(row["actual_candidate_count"])
            for row in rows
            if row["actual_candidate_count"] is not None
        ]
        summary = {
            "experiment": experiment.name,
            "reranker_enabled": experiment.enable_reranker,
            "candidate_k": experiment.candidate_k,
            "final_k": final_k,
            "retrieval_pool_per_retriever": retrieval_pool_size,
            "num_queries": len(rows),
            "recall_at_10": recall_at_10,
            "mean_latency_ms": retrieval_latency["mean"],
            "p50_latency_ms": retrieval_latency["p50"],
            "p95_latency_ms": retrieval_latency["p95"],
            "min_latency_ms": retrieval_latency["min"],
            "max_latency_ms": retrieval_latency["max"],
            "std_latency_ms": retrieval_latency["std"],
            "mean_reranker_latency_ms": (
                reranker_latency["mean"] if reranker_latency else None
            ),
            "p50_reranker_latency_ms": (
                reranker_latency["p50"] if reranker_latency else None
            ),
            "p95_reranker_latency_ms": (
                reranker_latency["p95"] if reranker_latency else None
            ),
            "actual_candidate_count_min": (
                min(actual_candidate_counts) if actual_candidate_counts else None
            ),
            "actual_candidate_count_mean": (
                statistics.fmean(actual_candidate_counts)
                if actual_candidate_counts
                else None
            ),
            "actual_candidate_count_max": (
                max(actual_candidate_counts) if actual_candidate_counts else None
            ),
            "delta_recall_at_10": recall_at_10 - baseline_recall,
            "delta_mean_latency_ms": (
                retrieval_latency["mean"] - baseline_latency["mean"]
            ),
            "delta_p95_latency_ms": (
                retrieval_latency["p95"] - baseline_latency["p95"]
            ),
            "mean_latency_increase_percent": mean_increase_percent,
            "num_queries_improved": len(improved),
            "num_queries_unchanged": len(unchanged),
            "num_queries_degraded": len(degraded),
            "queries_improved": improved,
            "queries_unchanged": unchanged,
            "queries_degraded": degraded,
        }
        summaries.append(summary)
    return summaries


def build_tradeoff_analysis(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Report the Pareto frontier without inventing an application SLA."""
    reranker_summaries = [row for row in summaries if row["reranker_enabled"]]
    pareto_efficient: list[str] = []
    for candidate in reranker_summaries:
        dominated = any(
            other["recall_at_10"] >= candidate["recall_at_10"]
            and other["p95_latency_ms"] <= candidate["p95_latency_ms"]
            and (
                other["recall_at_10"] > candidate["recall_at_10"]
                or other["p95_latency_ms"] < candidate["p95_latency_ms"]
            )
            for other in reranker_summaries
            if other is not candidate
        )
        if not dominated:
            pareto_efficient.append(candidate["experiment"])

    return {
        "pareto_efficient_reranker_experiments": pareto_efficient,
        "automatic_recommendation": None,
        "note": (
            "Choose among the Pareto-efficient configurations using the "
            "application P95 latency target and resource constraints; this "
            "benchmark does not assume either one."
        ),
    }


def environment_metadata(retriever: TrafficHybridRetriever) -> dict[str, Any]:
    total_ram_gb: float | None = None
    try:
        import psutil

        total_ram_gb = psutil.virtual_memory().total / (1024**3)
    except (ImportError, OSError):
        pass

    gpu: str | None = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except (ImportError, RuntimeError):
        pass

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine() or None,
        "ram_gb": total_ram_gb,
        "gpu": gpu,
        "reranker_model": retriever.reranker_model or DEFAULT_RERANKER_MODEL,
        "embedding_model": retriever.embedding_model,
        "qdrant_collection": retriever.collection_name,
    }


def save_outputs(
    output_dir: Path,
    results_by_experiment: dict[str, list[dict[str, Any]]],
    summary_document: dict[str, Any],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    query_path = output_dir / "reranker_benchmark_queries.jsonl"
    summary_path = output_dir / "reranker_benchmark_summary.json"
    csv_path = output_dir / "reranker_benchmark_summary.csv"

    with query_path.open("w", encoding="utf-8") as handle:
        for rows in results_by_experiment.values():
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_document, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    csv_fields = [
        "experiment",
        "reranker_enabled",
        "candidate_k",
        "final_k",
        "retrieval_pool_per_retriever",
        "num_queries",
        "recall_at_10",
        "mean_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "mean_reranker_latency_ms",
        "p50_reranker_latency_ms",
        "p95_reranker_latency_ms",
        "delta_recall_at_10",
        "delta_mean_latency_ms",
        "delta_p95_latency_ms",
        "mean_latency_increase_percent",
        "num_queries_improved",
        "num_queries_unchanged",
        "num_queries_degraded",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_document["experiments"])

    return query_path, summary_path, csv_path


def print_console_summary(summaries: list[dict[str, Any]]) -> None:
    headers = [
        "Experiment",
        "Candidate K",
        "Recall@10",
        "Mean ms",
        "P50 ms",
        "P95 ms",
        "Delta Recall",
        "Delta P95",
    ]
    rows = []
    for summary in summaries:
        rows.append([
            summary["experiment"],
            "-" if summary["candidate_k"] is None else str(summary["candidate_k"]),
            f"{summary['recall_at_10']:.4f}",
            f"{summary['mean_latency_ms']:.1f}",
            f"{summary['p50_latency_ms']:.1f}",
            f"{summary['p95_latency_ms']:.1f}",
            f"{summary['delta_recall_at_10']:+.4f}",
            f"{summary['delta_p95_latency_ms']:+.1f}",
        ])
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def format_row(row: list[str]) -> str:
        return "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        ) + " |"

    print(separator)
    print(format_row(headers))
    print(separator)
    for row in rows:
        print(format_row(row))
    print(separator)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark hybrid retrieval quality and reranker latency trade-offs."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--candidate-k",
        type=int,
        nargs="+",
        default=[10, 20, 50, 100],
        metavar="K",
    )
    parser.add_argument("--final-k", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL") or None)
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY") or None)
    parser.add_argument(
        "--qdrant-host", default=os.getenv("QDRANT_HOST", "localhost")
    )
    parser.add_argument(
        "--qdrant-port", type=int, default=os.getenv("QDRANT_PORT", "6333")
    )
    parser.add_argument(
        "--collection", default=os.getenv("COLLECTION_NAME", DEFAULT_COLLECTION)
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
    )
    parser.add_argument(
        "--reranker-model",
        default=os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
    )
    parser.add_argument("--chunks", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--retrieval-pool-size",
        type=int,
        help="Candidates fetched from each retriever; defaults to max(candidate-k).",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args(argv)

    if args.final_k != 10:
        parser.error("--final-k must be 10 for this Recall@10 benchmark")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if not args.candidate_k:
        parser.error("at least one --candidate-k value is required")
    if len(args.candidate_k) != len(set(args.candidate_k)):
        parser.error("--candidate-k values must not contain duplicates")
    if any(candidate_k < args.final_k for candidate_k in args.candidate_k):
        parser.error("every --candidate-k must be greater than or equal to --final-k")
    if args.rrf_k < 1:
        parser.error("--rrf-k must be at least 1")

    required_pool_size = max(args.candidate_k)
    if args.retrieval_pool_size is None:
        args.retrieval_pool_size = required_pool_size
    elif args.retrieval_pool_size < required_pool_size:
        parser.error(
            "--retrieval-pool-size must be at least max(--candidate-k) so every "
            "experiment receives its requested candidate count"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    samples = load_test_dataset(args.dataset)
    if args.limit is not None:
        samples = samples[: args.limit]

    experiments = [ExperimentConfig("hybrid_baseline", False, None)] + [
        ExperimentConfig(f"hybrid_reranker_k{candidate_k}", True, candidate_k)
        for candidate_k in args.candidate_k
    ]

    LOGGER.info(
        "Initializing retriever before timing (pool=%d per retriever)",
        args.retrieval_pool_size,
    )
    retriever = TrafficHybridRetriever(
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        jsonl_path=args.chunks,
        rrf_k=args.rrf_k,
        candidates_per_retriever=args.retrieval_pool_size,
        enable_reranker=True,
        reranker_model=args.reranker_model,
        rerank_top_n=max(args.candidate_k),
    )

    warm_up(retriever, samples, experiments, args.final_k, args.warmup)
    results_by_experiment: dict[str, list[dict[str, Any]]] = {}
    for experiment in experiments:
        results_by_experiment[experiment.name] = run_experiment(
            retriever, samples, experiment, args.final_k
        )

    summaries = summarize_experiments(
        results_by_experiment,
        experiments,
        args.final_k,
        args.retrieval_pool_size,
    )
    summary_document = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(Path(args.dataset).resolve()),
        "configuration": {
            "dataset_order": "original",
            "limit": args.limit,
            "warmup_queries_per_experiment": min(args.warmup, len(samples)),
            "warmup_included_in_measurements": False,
            "final_k": args.final_k,
            "retrieval_pool_per_retriever": args.retrieval_pool_size,
            "rrf_k": args.rrf_k,
            "candidate_k": args.candidate_k,
            "seed": None,
        },
        "environment": environment_metadata(retriever),
        "experiments": summaries,
        "tradeoff_analysis": build_tradeoff_analysis(summaries),
    }
    paths = save_outputs(args.output_dir, results_by_experiment, summary_document)

    print_console_summary(summaries)
    print("\nSaved benchmark outputs:")
    for path in paths:
        print(f"  {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
