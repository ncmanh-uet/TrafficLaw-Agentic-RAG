# TrafficLaw-Agentic-RAG

Vietnamese Traffic Law RAG with hybrid retrieval, Cross-Encoder reranking,
semantic chunking, and a bounded LangGraph agent with Tavily fallback.

The existing retriever and generator remain standalone business components.
`Sources.agentic_rag` adds stateful orchestration, deterministic evidence
verification, execution budgets, conditional tool selection, and structured
tracing for evaluation.

See [the Agentic RAG workflow](Sources/agentic_rag/README.md) for architecture,
configuration, usage, and tests.

See [the retriever/reranker benchmark report](reports/benchmark/retriever_benchmark.md)
for Recall@10 and latency trade-offs, and [the baseline-versus-agent benchmark
report](reports/benchmark/agent_benchmark.md) for answer quality, coverage,
latency, and score review details. Both reports are written in Vietnamese.

