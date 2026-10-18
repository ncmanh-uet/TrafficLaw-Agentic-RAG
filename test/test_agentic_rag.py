from __future__ import annotations

import re
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

from Sources.agentic_rag import AgentConfig, AgenticRAG, StopReason
from Sources.agentic_rag.evidence import (
    build_feedback_query,
    normalize_internal_chunk,
    select_generation_evidence,
)
from Sources.rag_core.generator import LegalAnswerGenerator
from Sources.rag_core.prompt import REFUSAL_TOKEN


@dataclass
class FakeChunk:
    id: int
    score: float
    content: str
    metadata: dict

    def to_dict(self):
        return {
            "id": self.id,
            "score": self.score,
            "content": self.content,
            "metadata": self.metadata,
        }


class FakeRetriever:
    def __init__(self, chunks=None, error: Exception | None = None):
        self.chunks = chunks or []
        self.error = error
        self.calls = 0
        self.last_query = None
        self.queries = []

    def get_relevant_chunks(self, query, top_k=10):
        self.calls += 1
        self.last_query = query
        self.queries.append(query)
        if self.error:
            raise self.error
        return self.chunks[:top_k]


class FakeGenerator:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls = 0
        self.last_chunks = []

    def generate(self, query, chunks):
        self.calls += 1
        self.last_chunks = chunks
        if self.error:
            raise self.error
        return {"answer": "Câu trả lời có căn cứ.", "sources": [], "refused": False}


class ExpandingFakeGenerator(FakeGenerator):
    def __init__(self):
        super().__init__()
        self.expansion_calls = 0

    def expand_query(self, query):
        self.expansion_calls += 1
        return "điều kiện kỹ thuật và giấy tờ của phương tiện"


class RefusingThenAnswerGenerator(FakeGenerator):
    def generate(self, query, chunks):
        self.calls += 1
        self.last_chunks = chunks
        if self.calls == 1:
            return {
                "answer": "Thông tin này không có trong tài liệu được cung cấp.",
                "sources": [],
                "refused": True,
            }
        return {"answer": "Câu trả lời từ bằng chứng bổ sung.", "sources": [], "refused": False}


class InvalidThenAnswerGenerator(FakeGenerator):
    def generate(self, query, chunks):
        self.calls += 1
        self.last_chunks = chunks
        if self.calls == 1:
            return {
                "answer": "Câu trả lời dẫn sai nguồn.",
                "sources": [],
                "refused": False,
                "citation_valid": False,
                "citation_issues": ["CROSS_REFERENCE_WITHOUT_DIRECT_SOURCE:L1"],
            }
        return {
            "answer": "Câu trả lời có nguồn hợp lệ.",
            "sources": [],
            "refused": False,
            "citation_valid": True,
            "citation_issues": [],
        }


class FakeWebSearch:
    def __init__(self, results=None, error: Exception | None = None):
        self.results = results or []
        self.error = error
        self.calls = 0

    def search(self, query):
        self.calls += 1
        if self.error:
            raise self.error
        return self.results


def relevant_chunk(chunk_id=1, **metadata):
    base = {"chunk_id": f"chunk-{chunk_id}", "doc_id": "ND168", "dieu": 6}
    base.update(metadata)
    return FakeChunk(
        chunk_id,
        0.9,
        "Ô tô vượt đèn đỏ bị phạt tiền theo quy định giao thông.",
        base,
    )


def official_web_result():
    return {
        "title": "Quy định xử phạt vượt đèn đỏ",
        "url": "https://xaydungchinhsach.chinhphu.vn/quy-dinh-vuot-den-do",
        "content": "Ô tô vượt đèn đỏ bị phạt tiền theo quy định giao thông.",
        "score": 0.95,
    }


class AgenticRAGTests(unittest.TestCase):
    def test_feedback_query_uses_corpus_terms_without_domain_rules(self):
        query = "Điều kiện của phương tiện vận chuyển người bệnh là gì?"
        chunk = relevant_chunk()
        chunk.content = (
            "Xe ô tô cứu thương vận chuyển người bệnh phải có dấu hiệu nhận diện "
            "và niêm yết thông tin trên xe."
        )
        feedback = build_feedback_query(query, [normalize_internal_chunk(chunk)[1]])
        self.assertIsNotNone(feedback)
        self.assertTrue(feedback.startswith(query))
        self.assertIn("dấu hiệu nhận diện", feedback)

    def test_retrieval_keeps_original_query_then_uses_generic_feedback(self):
        query = "Ô tô vượt đèn đỏ bị phạt?"
        retriever = FakeRetriever([relevant_chunk()])
        AgenticRAG(retriever, FakeGenerator()).invoke(query)
        self.assertEqual(retriever.queries[0], query)
        self.assertEqual(retriever.calls, 2)
        self.assertIn(relevant_chunk().content, retriever.queries[1])

    def test_llm_expansion_is_generic_and_observable(self):
        retriever = FakeRetriever([relevant_chunk()])
        generator = ExpandingFakeGenerator()
        result = AgenticRAG(retriever, generator).invoke("Phương tiện cần điều kiện gì?")
        self.assertEqual(generator.expansion_calls, 1)
        self.assertEqual(retriever.queries[1], "điều kiện kỹ thuật và giấy tờ của phương tiện")
        self.assertEqual(result["query_expansion_calls"], 1)

    def test_internal_retrieval_sufficient_skips_tavily(self):
        retriever = FakeRetriever([relevant_chunk()])
        generator = FakeGenerator()
        web = FakeWebSearch([official_web_result()])

        result = AgenticRAG(retriever, generator, web).invoke("Ô tô vượt đèn đỏ bị phạt?")

        self.assertEqual(result["stop_reason"], StopReason.SUCCESS.value)
        self.assertEqual(web.calls, 0)
        self.assertEqual(generator.calls, 1)
        self.assertEqual(
            [e["node"] for e in result["execution_trace"]],
            ["initialize", "plan", "retrieve", "plan", "verify_evidence", "plan", "generate", "plan", "finalize"],
        )

    def test_dated_event_rejects_topical_evidence_from_other_dates(self):
        query = "Trong ngày 11/10/2026, Hà Nội có tuyến đường nào tạm cấm?"
        internal = relevant_chunk()
        internal.content = "Hà Nội có thể tạm cấm tuyến đường để phục vụ sự kiện."
        wrong_date = {
            "title": "Hà Nội phân luồng phục vụ sự kiện tháng 5",
            "url": "https://xaydungchinhsach.chinhphu.vn/phan-luong-thang-5",
            "content": "Các tuyến đường tạm cấm từ ngày 11/5/2026 đến 13/5/2026.",
            "score": 0.95,
        }
        generator = FakeGenerator()
        web = FakeWebSearch([wrong_date])

        result = AgenticRAG(FakeRetriever([internal]), generator, web).invoke(query)

        self.assertEqual(web.calls, 1)
        self.assertEqual(generator.calls, 0)
        self.assertEqual(result["stop_reason"], StopReason.INSUFFICIENT_EVIDENCE.value)
        self.assertIn("TEMPORAL_MISMATCH", result["verification_issues"])

    def test_dated_event_accepts_authoritative_evidence_for_requested_date(self):
        query = "Trong ngày 11/10/2026, Hà Nội có tuyến đường nào tạm cấm?"
        internal = relevant_chunk()
        internal.content = "Hà Nội có thể tạm cấm tuyến đường để phục vụ sự kiện."
        exact_date = {
            "title": "Hà Nội phân luồng giao thông ngày 11/10/2026",
            "url": "https://xaydungchinhsach.chinhphu.vn/phan-luong-11-10",
            "content": "Ngày 11/10/2026, Hà Nội tạm cấm một số tuyến đường phục vụ sự kiện.",
            "score": 0.95,
        }
        generator = FakeGenerator()
        web = FakeWebSearch([exact_date])

        result = AgenticRAG(FakeRetriever([internal]), generator, web).invoke(query)

        self.assertEqual(web.calls, 1)
        self.assertEqual(generator.calls, 1)
        self.assertEqual(result["stop_reason"], StopReason.SUCCESS.value)
        self.assertNotIn("TEMPORAL_MISMATCH", result["verification_issues"])

    def test_freshness_question_routes_to_web_before_generation(self):
        query = "Tính đến ngày 11/10/2026, quy định vượt đèn đỏ mới nhất là gì?"
        current_web = {
            "title": "Quy định vượt đèn đỏ mới nhất",
            "url": "https://xaydungchinhsach.chinhphu.vn/quy-dinh-moi-nhat",
            "content": "Thông tin chính thức mới nhất về quy định xử phạt vượt đèn đỏ.",
            "score": 0.95,
        }
        generator = FakeGenerator()
        web = FakeWebSearch([current_web])

        result = AgenticRAG(
            FakeRetriever([relevant_chunk()]), generator, web
        ).invoke(query)

        self.assertEqual(web.calls, 1)
        self.assertEqual(generator.calls, 1)
        self.assertEqual(result["stop_reason"], StopReason.SUCCESS.value)
        decisions = [
            event.get("decision_reason")
            for event in result["execution_trace"]
            if event["node"] == "plan"
        ]
        self.assertIn("EVIDENCE_INSUFFICIENT_WEB_AVAILABLE", decisions)

    def test_insufficient_internal_evidence_falls_back_to_web(self):
        low = FakeChunk(1, 0.0, "Nội dung không liên quan.", {"chunk_id": "low", "doc_id": "X"})
        web = FakeWebSearch([official_web_result()])
        result = AgenticRAG(FakeRetriever([low]), FakeGenerator(), web).invoke(
            "Ô tô vượt đèn đỏ bị phạt?"
        )

        self.assertEqual(result["stop_reason"], StopReason.SUCCESS.value)
        self.assertEqual(web.calls, 1)
        self.assertEqual(result["web_searches"], 1)
        self.assertTrue(result["evidence_sufficient"])

    def test_empty_retrieval_triggers_web_search(self):
        web = FakeWebSearch([official_web_result()])
        result = AgenticRAG(FakeRetriever([]), FakeGenerator(), web).invoke(
            "Ô tô vượt đèn đỏ bị phạt?"
        )
        self.assertEqual(web.calls, 1)
        self.assertEqual(result["stop_reason"], StopReason.SUCCESS.value)

    def test_empty_tavily_result_stops_with_insufficient_evidence(self):
        result = AgenticRAG(FakeRetriever([]), FakeGenerator(), FakeWebSearch([])).invoke(
            "Ô tô vượt đèn đỏ bị phạt?"
        )
        self.assertEqual(result["stop_reason"], StopReason.INSUFFICIENT_EVIDENCE.value)
        self.assertEqual(result["web_searches"], 1)
        self.assertEqual(result["verification_issues"], ["NO_EVIDENCE"])

    def test_tavily_error_does_not_retry_forever(self):
        web = FakeWebSearch(error=RuntimeError("Tavily unavailable"))
        result = AgenticRAG(FakeRetriever([]), FakeGenerator(), web).invoke(
            "Ô tô vượt đèn đỏ bị phạt?"
        )
        self.assertEqual(web.calls, 1)
        self.assertEqual(result["stop_reason"], StopReason.TOOL_FAILURE.value)
        self.assertTrue(any(error["node"] == "web_search" for error in result["errors"]))

    def test_conflicting_evidence_is_represented(self):
        chunks = [
            relevant_chunk(1, conflict_key="red-light-penalty", conflict_value="18m"),
            relevant_chunk(2, conflict_key="red-light-penalty", conflict_value="20m"),
        ]
        config = AgentConfig(max_web_searches=0)
        result = AgenticRAG(FakeRetriever(chunks), FakeGenerator(), None, config=config).invoke(
            "Ô tô vượt đèn đỏ bị phạt?"
        )
        self.assertEqual(result["stop_reason"], StopReason.INSUFFICIENT_EVIDENCE.value)
        self.assertIn("CONFLICTING_EVIDENCE", result["verification_issues"])

    def test_iteration_budget_terminates_graph(self):
        config = AgentConfig(max_iterations=1)
        result = AgenticRAG(
            FakeRetriever([relevant_chunk()]), FakeGenerator(), FakeWebSearch(), config=config
        ).invoke("Ô tô vượt đèn đỏ bị phạt?")
        self.assertEqual(result["stop_reason"], StopReason.MAX_ITERATIONS.value)
        self.assertLessEqual(result["tool_calls"], 1)

    def test_tool_call_budget_terminates_before_generation(self):
        config = AgentConfig(max_tool_calls=1)
        generator = FakeGenerator()
        result = AgenticRAG(
            FakeRetriever([relevant_chunk()]), generator, FakeWebSearch(), config=config
        ).invoke("Ô tô vượt đèn đỏ bị phạt?")
        self.assertEqual(result["stop_reason"], StopReason.MAX_TOOL_CALLS.value)
        self.assertEqual(generator.calls, 0)

    def test_trace_contains_latency_and_total_latency(self):
        result = AgenticRAG(FakeRetriever([relevant_chunk()]), FakeGenerator()).invoke(
            "Ô tô vượt đèn đỏ bị phạt?"
        )
        self.assertGreaterEqual(result["total_latency_ms"], 0)
        self.assertTrue(all(event["latency_ms"] >= 0 for event in result["execution_trace"]))

    def test_generation_context_is_bounded_after_sibling_expansion(self):
        chunks = [relevant_chunk(1)]
        chunks.extend(
            relevant_chunk(
                index,
                is_sibling=True,
                dieu=6,
            )
            for index in range(2, 42)
        )
        generator = FakeGenerator()
        config = AgentConfig(retrieval_top_k=50, max_generation_evidence=12)

        result = AgenticRAG(FakeRetriever(chunks), generator, config=config).invoke(
            "Ô tô vượt đèn đỏ bị phạt?"
        )

        self.assertEqual(result["stop_reason"], StopReason.SUCCESS.value)
        self.assertEqual(len(result["evidence"]), 41)
        self.assertEqual(len(generator.last_chunks), 12)

    def test_generation_selection_prioritizes_relevant_rules_across_frames(self):
        chunks = []
        for index, article in enumerate((6, 7, 8, 9), 1):
            chunk = relevant_chunk(index, dieu=article)
            chunk.metadata["doc_id"] = "168/2024/NĐ-CP"
            chunk.content = "Không chấp hành hiệu lệnh của đèn tín hiệu giao thông bị phạt tiền."
            chunks.append(normalize_internal_chunk(chunk)[1])
        chunks = [
            *[
                normalize_internal_chunk(
                    FakeChunk(100 + index, 0.99, "Nội dung không liên quan.", {
                        "chunk_id": f"noise-{index}", "doc_id": "X", "dieu": 20 + index,
                    })
                )[1]
                for index in range(20)
            ],
            *chunks,
        ]
        selected = select_generation_evidence(
            "Các mức phạt phương tiện vượt đèn đỏ", chunks, limit=4
        )
        self.assertEqual(
            {int(item["metadata"]["dieu"]) for item in selected},
            {6, 7, 8, 9},
        )

    def test_generation_selection_does_not_pad_context_with_noise(self):
        relevant = normalize_internal_chunk(relevant_chunk())[1]
        noise = [
            normalize_internal_chunk(
                FakeChunk(
                    200 + index,
                    0.0,
                    f"Nội dung phụ không liên quan số {index}.",
                    {"chunk_id": f"irrelevant-{index}", "doc_id": "X", "dieu": index},
                )
            )[1]
            for index in range(20)
        ]
        selected = select_generation_evidence(
            "Ô tô vượt đèn đỏ bị phạt?", [relevant, *noise], limit=30
        )
        self.assertLess(len(selected), len(noise) + 1)

    def test_generator_refusal_triggers_bounded_web_recovery(self):
        generator = RefusingThenAnswerGenerator()
        web = FakeWebSearch([official_web_result()])

        result = AgenticRAG(FakeRetriever([relevant_chunk()]), generator, web).invoke(
            "Ô tô vượt đèn đỏ bị phạt?"
        )

        self.assertEqual(result["stop_reason"], StopReason.SUCCESS.value)
        self.assertEqual(web.calls, 1)
        self.assertEqual(generator.calls, 2)
        self.assertEqual(result["final_answer"], "Câu trả lời từ bằng chứng bổ sung.")
        decisions = [
            event.get("decision_reason")
            for event in result["execution_trace"]
            if event["node"] == "plan"
        ]
        self.assertIn("GENERATOR_REFUSED_WEB_RECOVERY", decisions)

    def test_real_generator_refusal_token_recovers_after_verifier_accepts_incomplete_evidence(self):
        web_result = official_web_result()
        web_result["content"] = "Ô tô vượt đèn đỏ bị phạt tiền từ 18 đến 20 triệu đồng."
        web = FakeWebSearch([web_result])
        generator = object.__new__(LegalAnswerGenerator)
        generator.provider = "google"
        generator.model_name = "stub"
        generator.llm = Mock()

        def answer_from_context(messages, **kwargs):
            web_source = re.search(r"\[SOURCE_ID=(S\d+)\] \[WEB\]", messages[-1].content)
            if web_source is None:
                return SimpleNamespace(content=REFUSAL_TOKEN)
            return SimpleNamespace(
                content=f"Ô tô vượt đèn đỏ bị phạt 18 đến 20 triệu đồng [{web_source.group(1)}]."
            )

        generator.llm.invoke.side_effect = answer_from_context

        result = AgenticRAG(
            FakeRetriever([relevant_chunk()]), generator, web,
            config=AgentConfig(max_retrieval_queries=1),
        ).invoke("Ô tô vượt đèn đỏ bị phạt bao nhiêu tiền?")

        verifications = [e for e in result["execution_trace"] if e["node"] == "verify_evidence"]
        self.assertTrue(verifications[0]["evidence_sufficient"])
        generations = [e for e in result["execution_trace"] if e["node"] == "generate"]
        self.assertEqual(generations[0]["decision_reason"], "GENERATOR_REFUSED")
        self.assertEqual(result["stop_reason"], StopReason.SUCCESS.value)
        self.assertEqual(web.calls, 1)
        self.assertEqual(generator.llm.invoke.call_count, 2)
        self.assertEqual(result["generation_model_calls"], 2)
        self.assertIn("18 đến 20 triệu đồng", result["final_answer"])
        self.assertEqual(result["final_sources"][0]["source_type"], "web")
        self.assertNotIn(REFUSAL_TOKEN, result["final_answer"])

    def test_real_generator_refusal_token_stops_when_web_cannot_recover(self):
        for web_results in (None, [], [official_web_result()]):
            with self.subTest(web_results=web_results):
                web = FakeWebSearch(web_results) if web_results is not None else None
                generator = object.__new__(LegalAnswerGenerator)
                generator.provider = "google"
                generator.model_name = "stub"
                generator.llm = Mock()
                generator.llm.invoke.return_value = SimpleNamespace(content=REFUSAL_TOKEN)

                result = AgenticRAG(
                    FakeRetriever([relevant_chunk()]), generator, web,
                    config=AgentConfig(max_retrieval_queries=1),
                ).invoke("Ô tô vượt đèn đỏ bị phạt bao nhiêu tiền?")

                self.assertEqual(result["stop_reason"], StopReason.INSUFFICIENT_EVIDENCE.value)
                self.assertIn("GENERATOR_REFUSED", result["verification_issues"])
                self.assertFalse(result["evidence_sufficient"])
                self.assertIsNone(result["answer_draft"])
                self.assertEqual(result["final_sources"], [])
                self.assertNotIn(REFUSAL_TOKEN, result["final_answer"])
                expected_calls = 2 if web_results else 1
                self.assertEqual(generator.llm.invoke.call_count, expected_calls)
                if web is not None:
                    self.assertEqual(web.calls, 1)

    def test_generator_refusal_with_empty_web_does_not_generate_forever(self):
        generator = RefusingThenAnswerGenerator()
        web = FakeWebSearch([])

        result = AgenticRAG(FakeRetriever([relevant_chunk()]), generator, web).invoke(
            "Ô tô vượt đèn đỏ bị phạt?"
        )

        self.assertEqual(result["stop_reason"], StopReason.INSUFFICIENT_EVIDENCE.value)
        self.assertEqual(web.calls, 1)
        self.assertEqual(generator.calls, 1)
        self.assertEqual(
            [event["node"] for event in result["execution_trace"]].count("verify_evidence"),
            2,
        )

    def test_invalid_citations_trigger_bounded_web_recovery(self):
        generator = InvalidThenAnswerGenerator()
        web = FakeWebSearch([official_web_result()])
        result = AgenticRAG(FakeRetriever([relevant_chunk()]), generator, web).invoke(
            "Ô tô vượt đèn đỏ bị phạt?"
        )
        self.assertEqual(result["stop_reason"], StopReason.SUCCESS.value)
        self.assertEqual(web.calls, 1)
        self.assertEqual(generator.calls, 2)
        self.assertEqual(result["final_answer"], "Câu trả lời có nguồn hợp lệ.")


if __name__ == "__main__":
    unittest.main()
