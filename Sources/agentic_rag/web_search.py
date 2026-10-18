"""Tavily execution component, intentionally separate from the retriever."""

from __future__ import annotations

import os
from typing import Any

from .config import AgentConfig


class TavilySearchTool:
    def __init__(
        self,
        api_key: str | None = None,
        config: AgentConfig | None = None,
        client: Any | None = None,
    ):
        self.config = config or AgentConfig()
        if client is not None:
            self.client = client
            return

        key = api_key or os.getenv("TAVILY_API_KEY")
        if not key:
            raise ValueError("Tavily requires TAVILY_API_KEY or api_key.")
        from tavily import TavilyClient

        self.client = TavilyClient(api_key=key)

    def search(self, query: str) -> list[dict[str, Any]]:
        response = self.client.search(
            query=query,
            search_depth=self.config.web_search_depth,
            max_results=self.config.web_max_results,
            include_domains=list(self.config.preferred_web_domains),
            include_domains_mode="prefer",
            include_answer=False,
            include_raw_content=False,
        )
        if isinstance(response, dict):
            results = response.get("results", [])
        else:
            results = response
        return [dict(item) for item in (results or []) if isinstance(item, dict)]
