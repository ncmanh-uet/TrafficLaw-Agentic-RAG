# Retriever/reranker benchmark

Run a quick smoke benchmark from the repository root:

```powershell
python test/evaluation/benchmark_reranker.py --limit 10
```

Run the full experiment matrix:

```powershell
python test/evaluation/benchmark_reranker.py `
  --dataset Data/test/evaluation.jsonl `
  --candidate-k 10 20 50 100 `
  --final-k 10 `
  --warmup 3
```

The script loads `.env`, so `QDRANT_URL` and `QDRANT_API_KEY` can be supplied
there or with `--qdrant-url` and `--qdrant-api-key`. Use `--help` for model,
collection, chunk corpus, output directory, and retrieval-pool overrides.

All experiments use the same dataset order, embedding model, collection, RRF
configuration, and dense/BM25 pool size. By default the pool fetched from each
retriever is the largest requested `candidate_k`. Only reranking and the number
of fused candidates passed to the reranker vary.

Initialization and the first three queries for every configuration are warm-up
work and are excluded from timing. The same queries are subsequently executed
again as ordinary measured samples, so no dataset row is omitted.

Outputs are written to `test/evaluation/results/`:

- `reranker_benchmark_queries.jsonl`: per-query results and baseline comparison
- `reranker_benchmark_summary.json`: aggregate metrics, environment metadata,
  query-level outcome lists, and the latency/recall Pareto frontier
- `reranker_benchmark_summary.csv`: spreadsheet-friendly aggregate metrics

The benchmark intentionally does not select one configuration automatically.
Use the reported Recall@10 and P95 latency with the application's own latency
target and resource constraints.
