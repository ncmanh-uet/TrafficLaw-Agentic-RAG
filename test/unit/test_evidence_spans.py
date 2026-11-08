"""Offline evidence addressing tests; no model calls or legal gold answers."""

import json
import unittest

from test.evaluation_agent.evidence_spans import build_evidence_index


def source(**updates):
    value = {"source_id": "S1", "source_type": "internal", "doc_id": "99/2030/NĐ-CP",
             "dieu": 21, "khoan": None, "diem": None}
    value.update(updates)
    return value


class EvidenceSpanTests(unittest.TestCase):
    def assert_exact_slices(self, index, contexts):
        for span in index.spans.values():
            self.assertEqual(span.text, contexts[span.context_index][span.start:span.end])
            self.assertTrue(span.text.strip())
            self.assertNotIn("[SOURCE_ID=", span.text)
            for parent_id in span.parent_ids:
                self.assertIn(parent_id, index.spans)

    def test_inline_children_keep_distinct_clause_subject_parents(self):
        context = ("[SOURCE_ID=S1] [INTERNAL:PRIMARY] Tài liệu giả định (99/2030/NĐ-CP) · Điều 21\n"
                   "Văn bản: Tài liệu giả định | Điều 21: Quyền và nghĩa vụ\n"
                   "## Điều 21. Quyền và nghĩa vụ\n"
                   "1. Đơn vị có các quyền sau đây:\n"
                   "a) Thu phí; b) Từ chối hồ sơ giả; c) Quyết định chính sách.\n"
                   "2. Đơn vị có các nghĩa vụ sau đây:\n"
                   "a) Chấp hành quy định; b) Mua bảo hiểm;\n\n"
                   "c) Cung cấp chứng từ; d) Bồi thường thiệt hại.")
        index = build_evidence_index([context], [source()])
        points = [span for span in index.spans.values() if span.kind == "point"]
        self.assertEqual([(span.location.paragraph, span.location.point) for span in points],
                         [("1", "a"), ("1", "b"), ("1", "c"),
                          ("2", "a"), ("2", "b"), ("2", "c"), ("2", "d")])
        self.assertIn("quyền", index.spans[points[0].parent_ids[-1]].text)
        self.assertIn("nghĩa vụ", index.spans[points[-1].parent_ids[-1]].text)
        self.assertNotIn("b)", points[0].text)
        self.assert_exact_slices(index, [context])

    def test_first_point_on_same_line_as_clause_lead(self):
        context = "1. Hồ sơ gồm a) Đơn đề nghị; b) Bản kê khai."
        index = build_evidence_index([context], [source()])
        points = [span for span in index.spans.values() if span.kind == "point"]
        self.assertEqual([span.location.point for span in points], ["a", "b"])
        self.assertEqual(index.spans[points[0].parent_ids[-1]].text, "1. Hồ sơ gồm")
        self.assert_exact_slices(index, [context])

    def test_legal_reference_mentions_are_not_child_definitions(self):
        context = ("4. Quy định chung:\n"
                   "a) Các điểm a), b) của danh mục không được thay đổi;\n"
                   "b) Hành vi quy định tại điểm c khoản 2; điểm d khoản\n3 bị trừ 5 điểm.")
        index = build_evidence_index([context], [source()])
        points = [span for span in index.spans.values() if span.kind == "point"]
        self.assertEqual([span.location.point for span in points], ["a", "b"])
        self.assertIn("các điểm a), b)", points[0].text.casefold())
        self.assert_exact_slices(index, [context])

    def test_child_source_keeps_consequence_lead_separate_from_conduct(self):
        context = ("[SOURCE_ID=S1] [INTERNAL:PRIMARY] Tài liệu (99/2030/NĐ-CP) · Điều 21 · Khoản 3 · Điểm b\n"
                   "Văn bản: Tài liệu | Điều 21: Điều kiện\n"
                   "3. Phạt từ 100 đến 200 đơn vị đối với các hành vi:\n"
                   "b) Nộp hồ sơ chậm từ 5 đến 9 ngày.")
        index = build_evidence_index([context], [source(khoan="3", diem="b")])
        point = next(span for span in index.spans.values() if span.kind == "point")
        self.assertEqual((point.location.paragraph, point.location.point), ("3", "b"))
        lead = index.spans[point.parent_ids[-1]]
        self.assertEqual(lead.location.point, None)
        self.assertIn("100", lead.text)
        self.assertNotIn("100", point.text)
        self.assert_exact_slices(index, [context])

    def test_logical_table_keeps_header_across_blank_and_repeat_separator(self):
        context = ("[SOURCE_ID=S1] [INTERNAL:PRIMARY] Tài liệu (99/2030/NĐ-CP) · Điều 21 · Khoản 1\n"
                   "Văn bản: Tài liệu | Điều 21: Thời gian\n"
                   "1. Khối lượng tối thiểu\n\n"
                   "| TT | Nội dung | Đơn vị | Loại A | Loại B |\n"
                   "|---|---|---|---|---|\n"
                   "| 1 | Lý thuyết | giờ | 50 | 60 |\n"
                   "| 2 | Thực hành | giờ | 20 | 30 |\n\n"
                   "| 3 | Quãng đường | km | 100 | 200 |\n"
                   "|---|---|---|---|---|\n"
                   "| 4 | Tổng thời gian | giờ | 70 | 90 |\n")
        index = build_evidence_index([context], [source(khoan="1")])
        tables = [span for span in index.spans.values() if span.kind == "table"]
        self.assertEqual(len(tables), 1)
        self.assertIn("Loại A | Loại B", tables[0].text)
        self.assertIn("| 4 | Tổng thời gian", tables[0].text)
        self.assertIn("\n\n| 3", tables[0].text)
        self.assertTrue(tables[0].parent_ids)
        self.assert_exact_slices(index, [context])

    def test_quoted_amendment_does_not_invent_child_legal_addresses(self):
        context = ("[SOURCE_ID=S1] [INTERNAL:PRIMARY] Tài liệu (99/2030/NĐ-CP) · Điều 21 · Khoản 1\n"
                   "Văn bản: Tài liệu | Điều 21: Sửa đổi văn bản\n"
                   "1. Sửa đổi khoản 2 Điều 7 như sau:\n"
                   "“2. Các trường hợp mới:\n"
                   "a) Trường hợp đầu; b) Trường hợp sau.”")
        index = build_evidence_index([context], [source(khoan="1")])
        body = [span for span in index.spans.values() if span.kind != "header"]
        self.assertTrue(body)
        self.assertTrue(all(span.location is None for span in body))
        self.assert_exact_slices(index, [context])

    def test_unexpected_child_source_numbering_is_not_assigned_to_metadata(self):
        index = build_evidence_index(["2. First.\n3. Second."], [source(khoan="2")])
        self.assertTrue(all(span.location is None for span in index.spans.values()))

    def test_header_inference_and_mapping_are_equivalent(self):
        context = ("[SOURCE_ID=S7] [INTERNAL:SIBLING] Tài liệu (99/2030/NĐ-CP) · Điều 21 · Khoản 3 · Điểm đ\n"
                   "3. Các hành vi:\nđ) Một điều kiện.")
        inferred = build_evidence_index([context])
        mapped = build_evidence_index([context], [source(source_id="S7", khoan="3", diem="đ")])
        self.assertEqual(inferred, mapped)
        point = next(span for span in inferred.spans.values() if span.kind == "point")
        self.assertEqual(point.location.point, "đ")
        self.assert_exact_slices(inferred, [context])

    def test_unknown_and_english_contexts_remain_exact_paragraphs(self):
        contexts = ["First source.\n\nSecond paragraph.", "Another document."]
        index = build_evidence_index(contexts)
        self.assertEqual([span.text for span in index.spans.values()],
                         ["First source.", "Second paragraph.", "Another document."])
        self.assertTrue(all(span.location is None for span in index.spans.values()))
        self.assert_exact_slices(index, contexts)

    def test_web_content_never_becomes_a_legal_document_from_numbers(self):
        context = "[SOURCE_ID=S1] [WEB] Đề xuất theo Điều 21 (99/2030/NĐ-CP)\n1. Nội dung bài báo."
        index = build_evidence_index([context], [{"source_id": "S1", "source_type": "web"}])
        self.assertTrue(all(span.location is None for span in index.spans.values()))
        self.assert_exact_slices(index, [context])

    def test_wrapper_only_is_not_evidence_and_ids_are_stable(self):
        self.assertFalse(build_evidence_index(["[SOURCE_ID=S1]"]).spans)
        contexts = ["[SOURCE_ID=S1]\nOne.", "[SOURCE_ID=S1]\nTwo."]
        index = build_evidence_index(contexts)
        self.assertEqual(list(index.spans), ["S1:E1", "S1:E2"])
        self.assertEqual(index, build_evidence_index(contexts))
        self.assert_exact_slices(index, contexts)

    def test_catalog_contains_resolved_text_not_external_quotes_or_offsets(self):
        index = build_evidence_index(["1. First."], [source()])
        data = json.loads(index.prompt_context)
        self.assertEqual(set(data), {"evidence_catalog"})
        item = data["evidence_catalog"][0]
        self.assertEqual(item["text"], "1. First.")
        self.assertEqual(item["paragraph"], "1")
        self.assertIn("parent_ids", item)

    def test_length_and_source_id_mismatches_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "evidence_source_context_length_mismatch"):
            build_evidence_index(["One."], [])
        with self.assertRaisesRegex(ValueError, "evidence_source_context_id_mismatch"):
            build_evidence_index(["[SOURCE_ID=S2]\nOne."], [source()])


if __name__ == "__main__":
    unittest.main()
