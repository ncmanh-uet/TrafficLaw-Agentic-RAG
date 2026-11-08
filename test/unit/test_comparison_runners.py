from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

from Sources.agentic_rag import AgentConfig
from test.evaluation_agent import (
    AgentRunner, BaselineRunner, ComparisonConfig, build_runner, build_runners,
)
from Sources.rag_core.generator import LegalAnswerGenerator, _build_context
from Sources.rag_core.prompt import REFUSAL_TOKEN


QUERY = "Ô tô vượt đèn đỏ bị phạt bao nhiêu?"


def chunk(number: int = 1) -> dict:
    return {"id": number, "score": 0.9,
            "content": "Ô tô vượt đèn đỏ bị phạt từ 18.000.000 đến 20.000.000 đồng.",
            "metadata": {"chunk_id": f"c{number}", "doc_id": "168/2024/NĐ-CP",
                         "ten_van_ban": "Nghị định 168/2024/NĐ-CP",
                         "dieu": 6, "khoan": 9, "diem": "b"}}


def web_result() -> dict:
    return {"title": "Ô tô vượt đèn đỏ", "url": "https://chinhphu.vn/traffic",
            "content": "Ô tô vượt đèn đỏ bị phạt từ 18.000.000 đến 20.000.000 đồng.",
            "score": 0.95}


def answered(text: str = "Phạt 18–20 triệu đồng [S1].") -> dict:
    return {"answer": text, "refused": False, "citation_valid": True, "model_calls": 1}


class Retriever:
    def __init__(self, rows=None, error=None):
        self.rows = [chunk()] if rows is None else rows
        self.error = error
        self.calls = []

    def get_relevant_chunks(self, query, top_k=10):
        self.calls.append((query, top_k))
        if self.error:
            raise self.error
        # Include all simulated siblings even if top_k is small, like the real
        # retriever. The shared selection stage must enforce the final ceiling.
        return deepcopy(self.rows)


class Generator:
    def __init__(self, responses=None):
        self.responses = list(responses or [answered()])
        self.calls = []
        self.expansions = []

    def expand_query(self, query):
        self.expansions.append(query)
        return query

    def generate(self, query, chunks):
        self.calls.append((query, deepcopy(chunks)))
        response = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return deepcopy(response)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.config = ComparisonConfig(generator_model="test-model")

    def test_baseline_runs_one_retrieval_and_never_expands_or_builds_graph(self):
        retriever, generator = Retriever(), Generator()
        with patch("test.evaluation_agent.runners.AgenticRAG", side_effect=AssertionError("no planner")):
            result = BaselineRunner(retriever, generator, self.config).run(QUERY)
        self.assertEqual(retriever.calls, [(QUERY, 10)])
        self.assertEqual(generator.expansions, [])
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["contexts"], [_build_context(generator.calls[0][1])])
        self.assertEqual(result["sources"][0]["source_id"], "S1")
        self.assertEqual(result["sources"][0]["chunk_id"], "c1")

    def test_shared_context_policy_deduplicates_and_bounds_sibling_enrichment(self):
        rows = [chunk(i) for i in range(1, 15)] + [chunk(1)]
        config = replace(self.config, agent_config=AgentConfig(max_generation_evidence=3))
        base_gen, agent_gen = Generator(), Generator()
        web = Mock()
        baseline = BaselineRunner(Retriever(rows), base_gen, config).run(QUERY)
        agent = AgentRunner(Retriever(rows), agent_gen, web, config).run(QUERY)
        self.assertEqual(base_gen.calls[0][1], agent_gen.calls[0][1])
        self.assertLessEqual(len(baseline["contexts"]), 3)
        self.assertEqual(baseline["contexts"], agent["contexts"])
        self.assertEqual(baseline["sources"], agent["sources"])
        self.assertEqual(len({s["chunk_id"] for s in baseline["sources"]}), len(baseline["sources"]))
        web.search.assert_not_called()

    def test_agent_retains_expansion_two_retrievals_and_skips_unneeded_web(self):
        retriever, generator, web = Retriever(), Generator(), Mock()
        runner = AgentRunner(retriever, generator, web, self.config)
        result = runner.run(QUERY)
        self.assertEqual(len(retriever.calls), 2)
        self.assertEqual(generator.expansions, [QUERY])
        self.assertEqual(len(generator.calls), 1)
        web.search.assert_not_called()
        self.assertEqual(result["status"], "answered")
        self.assertEqual((runner.agent.config.max_iterations, runner.agent.config.max_tool_calls,
                          runner.agent.config.max_web_searches, runner.agent.config.max_retrieval_queries),
                         (8, 4, 1, 2))

    def test_agent_with_no_expand_method_retains_feedback_fallback(self):
        generator = SimpleNamespace(generate=Mock(return_value=answered()))
        runner = AgentRunner(Retriever(), generator, Mock(), self.config)
        state = runner.agent.invoke(QUERY)
        self.assertEqual(state["query_expansion_calls"], 0)
        self.assertEqual(state["retrieval_calls"], 2)

    def test_agent_web_fallback_after_empty_internal_evidence(self):
        generator = Generator()
        web = Mock()
        web.search.return_value = [web_result()]
        result = AgentRunner(Retriever([]), generator, web, self.config).run(QUERY)
        web.search.assert_called_once()
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["sources"][0]["url"], web_result()["url"])
        self.assertIn("[WEB]", result["contexts"][0])

    def test_recovery_uses_final_generation_context_not_initial_source_numbers(self):
        generator = Generator([
            {"answer": "Thiếu bằng chứng", "refused": True}, answered("Đã bổ sung [S1]."),
        ])
        web = Mock()
        web.search.return_value = [web_result()]
        result = AgentRunner(Retriever(), generator, web, self.config).run(QUERY)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(len(generator.calls), 2)
        web.search.assert_called_once()
        final_chunks = generator.calls[-1][1]
        self.assertEqual("\n\n---\n\n".join(result["contexts"]), _build_context(final_chunks))
        self.assertEqual(len(result["sources"]), len(final_chunks))
        self.assertTrue(any(s["source_type"] == "web" for s in result["sources"]))

    def test_invalid_citation_still_triggers_agent_web_recovery(self):
        generator = Generator([
            {"answer": "Sai [S99]", "refused": False, "citation_valid": False,
             "citation_issues": ["UNKNOWN_SOURCE_ID:S99"]}, answered(),
        ])
        web = Mock()
        web.search.return_value = [web_result()]
        result = AgentRunner(Retriever(), generator, web, self.config).run(QUERY)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(len(generator.calls), 2)
        web.search.assert_called_once()

    def test_agent_budget_and_refusal_do_not_masquerade_as_tool_errors(self):
        web = Mock()
        web.search.return_value = []
        refused = AgentRunner(Retriever([]), Generator(), web, self.config).run(QUERY)
        self.assertEqual(refused["status"], "refused")
        self.assertEqual(refused["contexts"], [])
        config = replace(self.config, agent_config=AgentConfig(max_tool_calls=1))
        exhausted = AgentRunner(Retriever(), Generator(), Mock(), config).run(QUERY)
        self.assertEqual(exhausted["status"], "budget_exhausted")

    def test_baseline_keeps_invalid_citation_for_scoring_without_agent_recovery(self):
        generator = Generator([{"answer": "Sai [S99]", "citation_valid": False}])
        result = BaselineRunner(Retriever(), generator, self.config).run(QUERY)
        self.assertEqual(result["status"], "citation_invalid")
        self.assertEqual(result["answer"], "Sai [S99]")
        self.assertEqual(len(generator.calls), 1)

    def test_empty_retrieval_still_uses_baseline_generator_refusal(self):
        generator = Generator([{"answer": "Không đủ bằng chứng", "refused": True}])
        result = BaselineRunner(Retriever([]), generator, self.config).run(QUERY)
        self.assertEqual(result["status"], "refused")
        self.assertEqual(result["contexts"], [])
        self.assertEqual(len(generator.calls), 1)

    def test_errors_have_the_same_result_shape_and_no_stale_context(self):
        web = Mock()
        web.search.side_effect = RuntimeError("web unavailable")
        baseline = BaselineRunner(Retriever(error=RuntimeError("retrieval unavailable")), Generator(), self.config)
        agent = AgentRunner(Retriever(error=RuntimeError("retrieval unavailable")), Generator(), web, self.config)
        results = [baseline.run(QUERY), agent.run(QUERY)]
        self.assertEqual(set(results[0]), set(results[1]))
        for result in results:
            self.assertEqual(result["status"], "error")
            self.assertTrue(result["error"])
            self.assertEqual(result["contexts"], [])
            self.assertGreaterEqual(result["latency_ms"], 0)

    def test_each_run_is_fresh_and_accepts_only_query_strings(self):
        generator = Generator([answered(), RuntimeError("LLM unavailable")])
        runner = BaselineRunner(Retriever(), generator, self.config)
        self.assertTrue(runner.run(QUERY)["contexts"])
        self.assertEqual(runner.run(QUERY)["contexts"], [])
        for bad in ("", "  ", {"query": QUERY, "reference_answer": "gold"}):
            with self.assertRaises(ValueError):
                runner.run(bad)

    def test_same_actual_generator_is_used_by_both_paths(self):
        def real_generator():
            generator = LegalAnswerGenerator.__new__(LegalAnswerGenerator)
            generator.provider = "openai"
            generator.model_name = "test-model"
            generator.llm = Mock()
            generator.llm.invoke.side_effect = [
                SimpleNamespace(content="Phạt từ 18.000.000 đến 20.000.000 đồng [S1] [Điều 6, Khoản 9, Điểm b — 168/2024/NĐ-CP]."),
            ]
            return generator

        base_gen, agent_gen = real_generator(), real_generator()
        # This real generator would perform an expansion LLM call; disable that
        # one behavior in this test to isolate answer generation.
        agent_gen.expand_query = Mock(return_value=QUERY)
        base = BaselineRunner(Retriever(), base_gen, self.config).run(QUERY)
        agent = AgentRunner(Retriever(), agent_gen, Mock(), self.config).run(QUERY)
        self.assertEqual(base["status"], "answered")
        self.assertEqual(agent["status"], "answered")
        self.assertEqual(base["answer"], agent["answer"])
        self.assertEqual(base_gen.llm.invoke.call_count, 1)
        self.assertEqual(agent_gen.llm.invoke.call_count, 1)
        self.assertEqual(base_gen.llm.invoke.call_args_list, agent_gen.llm.invoke.call_args_list)

    def test_factories_build_shared_components_before_any_timed_run(self):
        retriever, generator, web = Retriever(), Generator(), Mock()
        with patch("test.evaluation_agent.runners._build_components", return_value=(retriever, generator)) as factory, \
             patch("test.evaluation_agent.runners.TavilySearchTool", return_value=web) as web_factory:
            runners = build_runners(self.config)
            factory.assert_called_once_with(self.config)
            web_factory.assert_called_once_with(config=self.config.agent_config)
        self.assertEqual(retriever.calls, [])
        self.assertEqual(generator.calls, [])
        self.assertIs(runners["baseline"].generator.generator, runners["agent"].generator.generator)
        self.assertIs(runners["baseline"].config, runners["agent"].config)
        with patch("test.evaluation_agent.runners._build_components", return_value=(retriever, generator)), \
             patch("test.evaluation_agent.runners.TavilySearchTool", side_effect=AssertionError("baseline does not need Tavily")):
            self.assertIsInstance(build_runner("baseline", self.config), BaselineRunner)

    def test_comparison_config_requires_pinned_model_and_reranker_off(self):
        with self.assertRaises(ValueError):
            ComparisonConfig(generator_model="")
        with self.assertRaises(ValueError):
            replace(self.config, enable_reranker=True)
        self.assertFalse(self.config.enable_reranker)
        self.assertEqual(self.config.temperature, 0.0)
        self.assertEqual(self.config.agent_config.max_generation_evidence, 30)
        self.assertNotIn("api_key", self.config.as_dict())

    def test_real_component_factory_passes_the_pinned_common_settings(self):
        from test.evaluation_agent.runners import _build_components

        retriever_factory = Mock()
        generator_factory = Mock()
        fake_module = SimpleNamespace(TrafficHybridRetriever=retriever_factory)
        with patch.dict(sys.modules, {"Sources.rag_core.retriever": fake_module}), \
             patch("test.evaluation_agent.runners.LegalAnswerGenerator", generator_factory):
            retriever, generator = _build_components(self.config)
        self.assertIs(retriever, retriever_factory.return_value)
        self.assertIs(generator, generator_factory.return_value)
        kwargs = retriever_factory.call_args.kwargs
        self.assertFalse(kwargs["enable_reranker"])
        self.assertEqual(kwargs["collection_name"], self.config.collection_name)
        self.assertEqual(kwargs["embedding_model"], self.config.embedding_model)
        self.assertEqual(kwargs["jsonl_path"], self.config.corpus_path)
        self.assertEqual(kwargs["candidates_per_retriever"], 30)
        generator_factory.assert_called_once_with(
            provider=self.config.generator_provider, model=self.config.generator_model,
            temperature=0.0, max_tokens=1024, reasoning_effort=None,
        )


if __name__ == "__main__":
    unittest.main()
