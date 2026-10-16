from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from Sources.rag_core.generator import (
    LegalAnswerGenerator,
    _build_context,
    _is_refusal_answer,
    _response_content_to_text,
)
from Sources.rag_core.prompt import REFUSAL_PHRASE, REFUSAL_TOKEN, SYSTEM_PROMPT


def internal_chunk():
    return {
        "id": "internal-1",
        "score": 0.9,
        "content": "Không chấp hành đèn tín hiệu bị phạt 18 đến 20 triệu đồng.",
        "metadata": {
            "chunk_id": "nd168-d6-k9-b",
            "doc_id": "168/2024/NĐ-CP",
            "ten_van_ban": "Nghị định 168/2024/NĐ-CP",
            "dieu": 6,
            "khoan": "9",
            "diem": "b",
        },
    }


def web_chunk():
    return {
        "id": "web-1",
        "score": 0.8,
        "content": "Thông tin giải thích từ cổng thông tin Chính phủ.",
        "metadata": {
            "source_type": "web",
            "doc_id": "https://example.gov.vn/vuot-den-do",
            "ten_van_ban": "Cổng thông tin Chính phủ",
            "url": "https://example.gov.vn/vuot-den-do",
        },
    }


def article_level_chunk():
    return {
        "id": "article-69",
        "score": 0.95,
        "content": (
            "Văn bản: Luật Đường bộ 2024 | Điều 69\n"
            "1. Dịch vụ vận tải người bệnh sử dụng xe cứu thương.\n"
            "2. Tổ chức cung cấp dịch vụ phải đáp ứng điều kiện pháp luật.\n"
            "3. Xe cứu thương phải có dấu hiệu nhận diện và thiết bị giám sát."
        ),
        "metadata": {
            "chunk_id": "35_2024_QH15_dieu69",
            "doc_id": "35/2024/QH15",
            "ten_van_ban": "Luật Đường bộ 2024",
            "dieu": 69,
            "khoan": None,
            "diem": None,
        },
    }


class GeneratorResponseContentTests(unittest.TestCase):
    def test_plain_string_content(self):
        self.assertEqual(_response_content_to_text("Câu trả lời"), "Câu trả lời")

    def test_list_of_google_style_text_blocks(self):
        content = [
            {"type": "text", "text": "Phạt tiền 18–20 triệu đồng."},
            {"type": "thinking", "thinking": "not exposed"},
            {"type": "text", "text": "Trừ 04 điểm GPLX."},
        ]
        self.assertEqual(
            _response_content_to_text(content),
            "Phạt tiền 18–20 triệu đồng.\nTrừ 04 điểm GPLX.",
        )

    def test_list_may_contain_plain_strings(self):
        self.assertEqual(_response_content_to_text(["Dòng 1", "Dòng 2"]), "Dòng 1\nDòng 2")

    def test_context_exposes_stable_source_ids_and_web_url(self):
        context = _build_context([internal_chunk(), web_chunk()])
        self.assertIn("[SOURCE_ID=S1] [INTERNAL:PRIMARY]", context)
        self.assertIn("[SOURCE_ID=S2] [WEB]", context)
        self.assertIn("URL: https://example.gov.vn/vuot-den-do", context)

    def test_source_ids_resolve_internal_and_web_metadata(self):
        answer = (
            "Phạt tiền theo quy định [S1] "
            "[Điều 6, Khoản 9, Điểm b — 168/2024/NĐ-CP]. "
            "Thông tin giải thích [S2]."
        )
        sources, issues = LegalAnswerGenerator._resolve_citations(
            answer,
            [internal_chunk(), web_chunk()],
        )
        self.assertEqual(issues, [])
        self.assertEqual([source["source_id"] for source in sources], ["S1", "S2"])
        self.assertEqual(sources[0]["diem"], "b")
        self.assertEqual(sources[1]["source_type"], "web")
        self.assertEqual(sources[1]["url"], "https://example.gov.vn/vuot-den-do")

    def test_legacy_source_reference_and_spacing_are_normalized(self):
        answer = LegalAnswerGenerator._normalize_generated_answer(
            "Mức phạt [Nguồn 2] [Điều 6, Khoản 9, Điểmb — 168/2024/NĐ-CP]."
        )
        self.assertIn("[S2]", answer)
        self.assertIn("Điểm b", answer)

    def test_combined_source_and_legal_citations_are_normalized(self):
        answer = LegalAnswerGenerator._normalize_generated_answer(
            "Điều kiện [S3, Điều 69, Khoản 3; S6, Điều 35, Khoản 2]."
        )
        self.assertEqual(
            answer,
            "Điều kiện [S3] [Điều 69, Khoản 3] [S6] [Điều 35, Khoản 2].",
        )

    def test_grouped_source_ids_are_split_before_validation(self):
        answer = LegalAnswerGenerator._normalize_generated_answer(
            "Các thông báo liên quan [S1, S2, S4]."
        )
        self.assertEqual(answer, "Các thông báo liên quan [S1] [S2] [S4].")

    def test_internal_citation_gets_canonical_doc_id(self):
        answer = LegalAnswerGenerator._canonicalize_internal_citation_doc_ids(
            "Phạt tiền [S1] [Điều 6, Khoản 9, Điểm b].",
            [internal_chunk()],
        )
        self.assertEqual(
            answer,
            "Phạt tiền [S1] [Điều 6, Khoản 9, Điểm b — 168/2024/NĐ-CP].",
        )

    def test_internal_citation_replaces_title_with_canonical_doc_id(self):
        answer = LegalAnswerGenerator._canonicalize_internal_citation_doc_ids(
            "Phạt tiền [S1] [Điều 6, Khoản 9, Điểm b — Nghị định 168].",
            [internal_chunk()],
        )
        self.assertIn("[Điều 6, Khoản 9, Điểm b — 168/2024/NĐ-CP]", answer)
        self.assertNotIn("Nghị định 168]", answer)

    def test_internal_source_without_legal_suffix_gets_metadata_location(self):
        answer = LegalAnswerGenerator._canonicalize_internal_citation_doc_ids(
            "Phạt tiền [S1].",
            [internal_chunk()],
        )
        self.assertEqual(
            answer,
            "Phạt tiền [S1] [Điều 6, Khoản 9, Điểm b — 168/2024/NĐ-CP].",
        )

    def test_web_source_is_not_given_a_legal_doc_id_suffix(self):
        answer = LegalAnswerGenerator._canonicalize_internal_citation_doc_ids(
            "Thông tin tham khảo [S1].",
            [web_chunk()],
        )
        self.assertEqual(answer, "Thông tin tham khảo [S1].")

    def test_web_citation_is_canonicalized_to_trusted_hostname_and_url(self):
        answer = LegalAnswerGenerator._canonicalize_web_citations(
            "Thông tin tham khảo [S1 — evil.example](https://evil.example/path).",
            [web_chunk()],
        )
        self.assertEqual(
            answer,
            "Thông tin tham khảo "
            "[S1 — example.gov.vn](https://example.gov.vn/vuot-den-do).",
        )
        sources, issues = LegalAnswerGenerator._resolve_citations(answer, [web_chunk()])
        self.assertEqual(issues, [])
        self.assertEqual(sources[0]["domain"], "example.gov.vn")

    def test_unknown_source_id_is_reported(self):
        sources, issues = LegalAnswerGenerator._resolve_citations(
            "Mức phạt [S99].",
            [internal_chunk()],
        )
        self.assertEqual(sources, [])
        self.assertIn("UNKNOWN_SOURCE_ID:S99", issues)
        self.assertIn("NO_RESOLVABLE_CITATIONS", issues)

    def test_legal_citation_without_source_id_remains_backward_compatible(self):
        sources = LegalAnswerGenerator._extract_cited_sources(
            "Phạt tiền [Điều 6, Khoản 9, Điểm b — 168/2024/NĐ-CP].",
            [internal_chunk()],
        )
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["source_id"], "S1")

    def test_point_citation_does_not_resolve_to_parent_chunk(self):
        parent = internal_chunk()
        parent["metadata"] = dict(parent["metadata"], diem=None, chunk_id="nd168-d6-k9")
        sources, issues = LegalAnswerGenerator._resolve_citations(
            "Phạt tiền [Điều 6, Khoản 9, Điểm b — 168/2024/NĐ-CP].",
            [parent],
        )
        self.assertEqual(sources, [])
        self.assertIn("NO_RESOLVABLE_CITATIONS", issues)

    def test_source_id_allows_real_paragraph_inside_article_level_chunk(self):
        answer = (
            "- Đơn vị cung cấp dịch vụ phải đáp ứng điều kiện pháp luật "
            "[S1] [Điều 69, Khoản 2 — 35/2024/QH15].\n"
            "- Xe phải có dấu hiệu nhận diện và thiết bị giám sát "
            "[S1] [Điều 69, Khoản 3 — 35/2024/QH15]."
        )
        sources, issues = LegalAnswerGenerator._resolve_citations(
            answer,
            [article_level_chunk()],
        )
        self.assertEqual(issues, [])
        self.assertEqual([source["source_id"] for source in sources], ["S1"])

    def test_article_level_citation_accepts_canonical_document_title(self):
        _, issues = LegalAnswerGenerator._resolve_citations(
            "Xe phải có dấu hiệu nhận diện [S1] "
            "[Điều 69, Khoản 3 — Luật Đường bộ 2024].",
            [article_level_chunk()],
        )
        self.assertEqual(issues, [])

    def test_source_id_rejects_nonexistent_paragraph_in_article_level_chunk(self):
        _, issues = LegalAnswerGenerator._resolve_citations(
            "Điều kiện khác [S1] [Điều 69, Khoản 9 — 35/2024/QH15].",
            [article_level_chunk()],
        )
        self.assertIn("SOURCE_LEGAL_LOCATION_MISMATCH:S1:L1", issues)

    def test_cross_reference_alone_cannot_support_named_red_light_action(self):
        cross_reference = internal_chunk()
        cross_reference["content"] = (
            "10. Phạt tiền 10 đến 14 triệu đồng: a) Vi phạm quy định tại một số điểm của Điều này "
            "mà gây tai nạn giao thông."
        )
        cross_reference["metadata"] = dict(
            cross_reference["metadata"],
            dieu=7,
            khoan="10",
            diem="a",
            chunk_id="nd168-d7-k10-a",
        )
        _, issues = LegalAnswerGenerator._resolve_citations(
            "Xe máy vượt đèn đỏ gây tai nạn bị phạt [S1] "
            "[Điều 7, Khoản 10, Điểm a — 168/2024/NĐ-CP].",
            [cross_reference],
        )
        self.assertIn("CROSS_REFERENCE_WITHOUT_DIRECT_SOURCE:L1", issues)

    def test_cross_reference_is_valid_when_direct_rule_is_cited_too(self):
        direct = internal_chunk()
        cross_reference = internal_chunk()
        cross_reference["content"] = (
            "16. b) Thực hiện hành vi quy định tại điểm b khoản 9 Điều này "
            "bị trừ 04 điểm giấy phép lái xe."
        )
        cross_reference["metadata"] = dict(
            cross_reference["metadata"],
            khoan="16",
            diem="b",
            chunk_id="nd168-d6-k16-b",
        )
        _, issues = LegalAnswerGenerator._resolve_citations(
            "Bị trừ 04 điểm [S1] [Điều 6, Khoản 9, Điểm b — 168/2024/NĐ-CP] "
            "[S2] [Điều 6, Khoản 16, Điểm b — 168/2024/NĐ-CP].",
            [direct, cross_reference],
        )
        self.assertEqual(issues, [])

    def test_anaphoric_cross_reference_may_use_direct_source_from_prior_bullet(self):
        direct = internal_chunk()
        cross_reference = internal_chunk()
        cross_reference["content"] = (
            "16. b) Thực hiện hành vi quy định tại điểm b khoản 9 Điều này "
            "bị trừ 04 điểm giấy phép lái xe."
        )
        cross_reference["metadata"] = dict(
            cross_reference["metadata"],
            khoan="16",
            diem="b",
            chunk_id="nd168-d6-k16-b",
        )
        answer = (
            "### Xe ô tô\n"
            "- Vượt đèn đỏ bị phạt [S1] "
            "[Điều 6, Khoản 9, Điểm b — 168/2024/NĐ-CP].\n"
            "- Hành vi này bị trừ 04 điểm [S2] "
            "[Điều 6, Khoản 16, Điểm b — 168/2024/NĐ-CP]."
        )
        _, issues = LegalAnswerGenerator._resolve_citations(answer, [direct, cross_reference])
        self.assertEqual(issues, [])

    def test_source_id_must_match_legal_location_on_same_claim(self):
        _, issues = LegalAnswerGenerator._resolve_citations(
            "Phạt tiền [S1] [Điều 7, Khoản 10, Điểm a — 168/2024/NĐ-CP].",
            [internal_chunk()],
        )
        self.assertIn("SOURCE_LEGAL_LOCATION_MISMATCH:S1:L1", issues)

    def test_generate_returns_structured_citation_status(self):
        class StubLLM:
            def invoke(self, messages, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(
                    content=[
                        {
                            "type": "text",
                            "text": (
                                "Phạt **18–20 triệu đồng** [S1] "
                                "[Điều 6, Khoản 9, Điểm b — 168/2024/NĐ-CP]."
                            ),
                        }
                    ]
                )

        generator = object.__new__(LegalAnswerGenerator)
        generator.provider = "google"
        generator.model_name = "stub"
        generator.llm = StubLLM()
        result = generator.generate("Ô tô vượt đèn đỏ bị phạt bao nhiêu?", [internal_chunk()])
        self.assertTrue(result["citation_valid"])
        self.assertEqual(result["citation_issues"], [])
        self.assertEqual(result["sources"][0]["chunk_id"], "nd168-d6-k9-b")
        self.assertEqual(
            generator.llm.kwargs["automatic_function_calling"],
            {"disable": True},
        )

    def test_generate_repairs_unresolvable_citations_once(self):
        class RepairingLLM:
            def __init__(self):
                self.calls = 0

            def invoke(self, messages, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(content="Phạt 18 đến 20 triệu đồng.")
                return SimpleNamespace(
                    content=(
                        "Phạt 18 đến 20 triệu đồng [S1] "
                        "[Điều 6, Khoản 9, Điểm b — 168/2024/NĐ-CP]."
                    )
                )

        generator = object.__new__(LegalAnswerGenerator)
        generator.provider = "google"
        generator.model_name = "stub"
        generator.llm = RepairingLLM()
        result = generator.generate("Mức phạt là bao nhiêu?", [internal_chunk()])
        self.assertTrue(result["citation_valid"])
        self.assertEqual(result["model_calls"], 2)
        self.assertEqual(generator.llm.calls, 2)

    def test_generate_marks_explanatory_no_evidence_answer_as_refused(self):
        class RefusalLLM:
            def invoke(self, messages, **kwargs):
                return SimpleNamespace(
                    content=(
                        "Các nguồn tìm được nói về những thời điểm khác. "
                        "Do đó, không có căn cứ để xác định các tuyến đường "
                        "tạm cấm trong ngày được hỏi từ các nguồn dữ liệu này."
                    )
                )

        generator = object.__new__(LegalAnswerGenerator)
        generator.provider = "google"
        generator.model_name = "stub"
        generator.llm = RefusalLLM()
        result = generator.generate("Tuyến đường nào bị cấm?", [web_chunk()])
        self.assertTrue(result["refused"])
        self.assertEqual(result["answer_status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result["model_calls"], 1)

    def test_refusal_token_must_be_the_entire_response(self):
        self.assertTrue(_is_refusal_answer(f"\n {REFUSAL_TOKEN}\n"))
        self.assertFalse(_is_refusal_answer(f"Mã trạng thái {REFUSAL_TOKEN} [S1]."))

    def test_generate_refuses_relevant_but_incomplete_context_without_citation_repair(self):
        incomplete = internal_chunk()
        incomplete["content"] = "Ô tô vượt đèn đỏ bị phạt tiền theo quy định giao thông."
        generator = object.__new__(LegalAnswerGenerator)
        generator.provider = "google"
        generator.model_name = "stub"
        generator.llm = Mock()
        generator.llm.invoke.return_value = SimpleNamespace(
            content=[{"type": "text", "text": f"\n{REFUSAL_TOKEN}\n"}]
        )

        result = generator.generate("Ô tô vượt đèn đỏ bị phạt bao nhiêu tiền?", [incomplete])

        self.assertTrue(result["refused"])
        self.assertEqual(result["answer_status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result["answer"], REFUSAL_PHRASE)
        self.assertEqual(result["sources"], [])
        self.assertTrue(result["citation_valid"])
        self.assertEqual(result["citation_issues"], [])
        self.assertEqual(result["model_calls"], 1)
        generator.llm.invoke.assert_called_once()
        messages = generator.llm.invoke.call_args.args[0]
        self.assertIn(REFUSAL_TOKEN, messages[0].content)
        self.assertIn(REFUSAL_TOKEN, messages[1].content)

    def test_citation_repair_can_return_refusal_token(self):
        generator = object.__new__(LegalAnswerGenerator)
        generator.provider = "google"
        generator.model_name = "stub"
        generator.llm = Mock()
        generator.llm.invoke.side_effect = [
            SimpleNamespace(content="Phạt tiền theo quy định."),
            SimpleNamespace(content=REFUSAL_TOKEN),
        ]

        result = generator.generate("Mức phạt là bao nhiêu?", [internal_chunk()])

        self.assertTrue(result["refused"])
        self.assertEqual(result["answer_status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result["answer"], REFUSAL_PHRASE)
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["citation_issues"], [])
        self.assertEqual(result["model_calls"], 2)
        self.assertEqual(generator.llm.invoke.call_count, 2)
        repair_prompt = generator.llm.invoke.call_args.args[0][-1].content
        self.assertIn(REFUSAL_TOKEN, repair_prompt)

    def test_generate_still_recognizes_legacy_refusal_phrase(self):
        generator = object.__new__(LegalAnswerGenerator)
        generator.provider = "google"
        generator.model_name = "stub"
        generator.llm = Mock()
        generator.llm.invoke.return_value = SimpleNamespace(content=REFUSAL_PHRASE)

        result = generator.generate("Mức phạt là bao nhiêu?", [])

        self.assertTrue(result["refused"])
        self.assertEqual(result["answer"], REFUSAL_PHRASE)
        generator.llm.invoke.assert_called_once()

    def test_generate_renders_web_citation_without_model_authored_domain(self):
        class WebLLM:
            def invoke(self, messages, **kwargs):
                return SimpleNamespace(content="Thông tin giải thích [S1].")

        generator = object.__new__(LegalAnswerGenerator)
        generator.provider = "google"
        generator.model_name = "stub"
        generator.llm = WebLLM()
        result = generator.generate("Thông tin gì?", [web_chunk()])
        self.assertIn(
            "[S1 — example.gov.vn](https://example.gov.vn/vuot-den-do)",
            result["answer"],
        )
        self.assertTrue(result["citation_valid"])
        self.assertEqual(result["sources"][0]["domain"], "example.gov.vn")

    def test_prompt_uses_generic_grounding_not_case_specific_rules(self):
        self.assertNotIn("vượt đèn đỏ", SYSTEM_PROMPT.lower())
        self.assertNotIn("168/2024/NĐ-CP", SYSTEM_PROMPT)
        self.assertIn("Mỗi kết luận", SYSTEM_PROMPT)
        self.assertIn("tham chiếu", SYSTEM_PROMPT)
        self.assertIn("rà soát toàn bộ", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
