"""Shared configuration for baseline and agent benchmarking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path

from Sources.agentic_rag.config import AgentConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ComparisonConfig:
    generator_model: str
    generator_provider: str = "google"
    temperature: float | None = 0.0
    max_tokens: int = 1024
    reasoning_effort: str | None = None
    collection_name: str = "Traffic_Law_Hybrid"
    embedding_model: str = "intfloat/multilingual-e5-base"
    corpus_path: Path = PROJECT_ROOT / "Data/all_chunks.jsonl"
    candidates_per_retriever: int = 30
    rrf_k: int = 60
    enable_reranker: bool = False
    agent_config: AgentConfig = field(default_factory=AgentConfig)

    def __post_init__(self) -> None:
        if not self.generator_model.strip():
            raise ValueError("Set GENERATOR_MODEL explicitly to pin the comparison model.")
        if self.generator_provider not in {"google", "openai", "anthropic"}:
            raise ValueError("generator_provider must be google, openai, or anthropic")
        if self.reasoning_effort is not None and self.reasoning_effort not in {
                "none", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Invalid generator reasoning effort")
        if self.generator_provider == "openai" and self.generator_model.startswith("gpt-6-luna"):
            object.__setattr__(self, "temperature", None)
            object.__setattr__(self, "reasoning_effort", self.reasoning_effort or "none")
        if self.generator_provider == "anthropic":
            object.__setattr__(self, "temperature", None)
            object.__setattr__(self, "reasoning_effort", None)
        if self.enable_reranker:
            raise ValueError("This comparison requires enable_reranker=False.")
        if self.candidates_per_retriever < 1 or self.rrf_k < 1 or self.max_tokens < 1:
            raise ValueError("Candidate pool, RRF k and max_tokens must be positive.")

    @classmethod
    def from_env(cls, *, provider=None, model=None, reasoning_effort=None, max_tokens=None) -> ComparisonConfig:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
        return cls(
            generator_model=model or os.getenv("GENERATOR_MODEL", "").strip(),
            generator_provider=provider or os.getenv("GENERATOR_PROVIDER", "google").strip().lower(),
            reasoning_effort=reasoning_effort or os.getenv("GENERATOR_REASONING_EFFORT") or None,
            max_tokens=max_tokens if max_tokens is not None else int(os.getenv("GENERATOR_MAX_TOKENS", "1024")),
            collection_name=(os.getenv("QDRANT_COLLECTION") or os.getenv("COLLECTION_NAME")
                             or "Traffic_Law_Hybrid"),
            embedding_model=os.getenv("EMBEDDING_MODEL") or "intfloat/multilingual-e5-base",
        )

    def as_dict(self) -> dict:
        """Public settings only; credentials are never included."""
        config = asdict(self)
        config["corpus_path"] = str(self.corpus_path)
        return config
