"""Run one end-to-end Agentic RAG query with the existing components."""

from __future__ import annotations

import argparse
import json
import os
import sys

from dotenv import load_dotenv

from Sources.rag_core.generator import LegalAnswerGenerator
from Sources.rag_core.retriever import TrafficHybridRetriever

from .config import AgentConfig
from .web_search import TavilySearchTool
from .workflow import AgenticRAG


def build_agent() -> AgenticRAG:
    load_dotenv()
    config = AgentConfig()
    retriever = TrafficHybridRetriever(
        qdrant_url=os.getenv("QDRANT_URL") or None,
        qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
        collection_name=os.getenv("QDRANT_COLLECTION") or os.getenv("COLLECTION_NAME") or "Traffic_Law_Hybrid",
        embedding_model=os.getenv("EMBEDDING_MODEL") or "intfloat/multilingual-e5-base",
        enable_reranker=False,
    )
    generator = LegalAnswerGenerator(
        provider=os.getenv("GENERATOR_PROVIDER", "google"),
        model=os.getenv("GENERATOR_MODEL") or None,
    )
    tavily = TavilySearchTool(config=config)
    return AgenticRAG(retriever, generator, tavily, config=config)


def main() -> None:
    # Windows may inherit a legacy cp1252 console even though answers are
    # Vietnamese. Reconfigure only the text wrapper; terminals already using
    # UTF-8 are unaffected.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run one TrafficRAG Agentic RAG query")
    parser.add_argument("query", help="Vietnamese traffic-law question")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print the complete state, including all evidence and trace fields",
    )
    args = parser.parse_args()
    result = build_agent().invoke(args.query)
    if args.debug:
        output = result
    else:
        output = {
            "answer": result.get("final_answer"),
            "sources": result.get("final_sources", []),
            "citation_valid": (
                result.get("answer_draft", {}).get("citation_valid")
                if isinstance(result.get("answer_draft"), dict)
                else None
            ),
            "citation_issues": (
                result.get("answer_draft", {}).get("citation_issues", [])
                if isinstance(result.get("answer_draft"), dict)
                else []
            ),
            "stop_reason": result.get("stop_reason"),
            "metrics": {
                "iterations": result.get("iteration", 0),
                "tool_calls": result.get("tool_calls", 0),
                "retrieval_calls": result.get("retrieval_calls", 0),
                "query_expansion_calls": result.get("query_expansion_calls", 0),
                "generation_model_calls": result.get("generation_model_calls", 0),
                "web_searches": result.get("web_searches", 0),
                "evidence_count": len(result.get("evidence", [])),
                "total_latency_ms": result.get("total_latency_ms"),
            },
            "errors": result.get("errors", []),
        }
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
