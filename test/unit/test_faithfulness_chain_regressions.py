"""Symmetric structural regressions: validate judge links, never force scores."""

import copy
import unittest

from test.evaluation_agent.evidence_spans import build_evidence_index
from test.evaluation_agent.faithfulness import FaithfulnessNLIOutput, JudgeVerdictError, validate_faithfulness_result


DOCUMENT = "900/2030/NĐ-CP"  # Fictional, independent of benchmark answers.


def provision(source_id, paragraph, point, body, *, article="21", document=DOCUMENT):
    metadata = {"source_id": source_id, "source_type": "internal", "doc_id": document,
                "dieu": article, "khoan": paragraph, "diem": point}
    context = (f"[SOURCE_ID={source_id}] [INTERNAL:PRIMARY] Nghị định {document} ({document})"
               f" · Điều {article} · Khoản {paragraph} · Điểm {point}\n"
               f"Văn bản: Nghị định {document} | Điều {article}: Quy tắc giả định\n"
               f"{paragraph}. Các trường hợp áp dụng:\n{point}) {body}")
    return context, metadata


def fixture():
    items = [
        provision("S1", "6", "a", "Người nộp hồ sơ chậm dưới 5 ngày."),
        provision("S2", "8", "b", "Người nộp hồ sơ chậm từ 5 đến 9 ngày."),
        provision("S3", "7", "c", "Người nộp hồ sơ không xuất trình mã tiếp nhận."),
        provision("S4", "7", "d", "Người nộp hồ sơ cung cấp bản sao mờ."),
        provision("S5", "7", "đ", "Người nộp hồ sơ kê khai sai tên."),
        provision("S6", "13", "b", "Thực hiện hành vi quy định tại điểm đ khoản 4; điểm a khoản 6; "
                  "điểm c, điểm d, điểm đ khoản\n7; điểm a khoản 8 Điều này bị trừ 04 điểm."),
        provision("S7", "13", "d", "Thực hiện hành vi quy định tại điểm b khoản 8, khoản 10 Điều này bị trừ 10 điểm."),
        provision("S8", "8", "b", "Người nộp hồ sơ chậm từ 5 đến 9 ngày.", document="901/2030/NĐ-CP"),
        provision("S9", "8", "b", "Người nộp hồ sơ chậm từ 5 đến 9 ngày.", article="22"),
    ]
    contexts, sources = map(list, zip(*items))
    index = build_evidence_index(contexts, sources)
    ids = {}
    for source in sources:
        matches = [span for span in index.spans.values()
                   if span.source_id == source["source_id"] and span.location is not None
                   and span.location.point == source["diem"]]
        if not matches:
            raise AssertionError(f"No point span for {source['source_id']}")
        ids[source["source_id"]] = min(matches, key=lambda span: len(span.text)).evidence_id
    return index, ids, contexts


def verdict(evidence_ids, rule_links, *, supported=True, decision=None):
    return FaithfulnessNLIOutput(statements=[{
        "statement": "Một nhận định điều kiện và hệ quả cần kiểm chứng.",
        "reason": "Đối chiếu những vị trí pháp lý được chọn.",
        "verdict": int(supported),
        "decision": decision or ("supported" if supported else "contradicted"),
        "evidence_ids": evidence_ids,
        "rule_links": rule_links,
    }])


def link(rule, target, applies=True):
    return {"rule_id": rule, "target_id": target, "applies": applies}


class FaithfulnessChainRegressionTests(unittest.TestCase):
    def setUp(self):
        self.index, self.ids, self.contexts = fixture()

    def check(self, result):
        before = copy.deepcopy(result.model_dump())
        audits = validate_faithfulness_result(result, [result.statements[0].statement], self.index)
        self.assertEqual(result.model_dump(), before)
        return audits

    def assert_rule(self, rule, result):
        with self.assertRaises(JudgeVerdictError) as caught:
            self.check(result)
        self.assertEqual(caught.exception.rule, rule)

    def test_low_band_four_and_middle_band_ten_have_valid_chains(self):
        for rule_source, conduct_source in (("S6", "S1"), ("S7", "S2")):
            rule_id, conduct_id = self.ids[rule_source], self.ids[conduct_source]
            with self.subTest(conduct=conduct_source):
                self.check(verdict([rule_id, conduct_id], [link(rule_id, conduct_id)]))

    def test_middle_band_cannot_borrow_four_point_rule_from_neighbor(self):
        rule_id, conduct_id = self.ids["S6"], self.ids["S2"]
        self.assert_rule("cross_reference_relation_mismatch",
                         verdict([rule_id, conduct_id], [link(rule_id, conduct_id)]))
        self.check(verdict([rule_id, conduct_id], [link(rule_id, conduct_id, False)],
                           supported=False, decision="cross_reference_mismatch"))

    def test_honestly_declared_broken_link_cannot_support_a_positive_verdict(self):
        rule_id, conduct_id = self.ids["S6"], self.ids["S2"]
        self.assert_rule("positive_with_broken_reference", verdict(
            [rule_id, conduct_id], [link(rule_id, conduct_id, False)]))

    def test_semicolon_list_preserves_each_c_d_and_stroked_d_point(self):
        rule_id = self.ids["S6"]
        for source_id in ("S3", "S4", "S5"):
            conduct_id = self.ids[source_id]
            with self.subTest(source_id=source_id):
                self.check(verdict([rule_id, conduct_id], [link(rule_id, conduct_id)]))

    def test_valid_point_in_list_cannot_be_declared_missing(self):
        rule_id, conduct_id = self.ids["S6"], self.ids["S3"]
        self.assert_rule("cross_reference_relation_mismatch", verdict(
            [rule_id, conduct_id], [link(rule_id, conduct_id, False)],
            supported=False, decision="cross_reference_mismatch"))
        self.assert_rule("cross_reference_diagnosis_mismatch", verdict(
            [rule_id, conduct_id], [link(rule_id, conduct_id, True)],
            supported=False, decision="cross_reference_mismatch"))

    def test_wrong_numeric_claim_can_remain_negative_even_when_reference_link_is_valid(self):
        rule_id, conduct_id = self.ids["S7"], self.ids["S2"]
        output = verdict([rule_id, conduct_id], [link(rule_id, conduct_id)],
                         supported=False, decision="contradicted")
        self.check(output)
        self.assertEqual(output.statements[0].verdict, 0)

    def test_code_audit_records_original_span_and_actual_link_not_only_model_claim(self):
        rule_id, conduct_id = self.ids["S7"], self.ids["S2"]
        output = verdict([rule_id, conduct_id], [link(rule_id, conduct_id)])
        audits = self.check(output)
        self.assertEqual(audits[0]["rule_links"][0]["code_applies"], True)
        self.assertEqual(audits[0]["evidence"][0]["text"], self.index.spans[rule_id].text)
        self.assertEqual(audits[0]["evidence"][1]["text"], self.index.spans[conduct_id].text)

    def test_selected_applicability_rule_needs_conduct_link_for_both_verdicts(self):
        rule_id = self.ids["S6"]
        for supported, decision in ((True, "supported"), (False, "contradicted"),
                                    (False, "not_found"), (False, "cross_reference_mismatch")):
            with self.subTest(supported=supported, decision=decision):
                self.assert_rule("cross_reference_link_missing",
                                 verdict([rule_id], [], supported=supported, decision=decision))

    def test_exclusion_or_parenthetical_reference_does_not_force_positive_chain(self):
        bodies = [
            "Không chấp hành biển báo, trừ các hành vi vi phạm quy định tại điểm b khoản 8 Điều này.",
            "Không chấp hành biển báo, ngoại trừ hành vi vi phạm quy định tại điểm b khoản 8 Điều này.",
            "Không chấp hành biển báo (vi phạm quy định tại điểm b khoản 8 Điều này được xử lý riêng).",
        ]
        for body in bodies:
            context, source = provision("S10", "1", "a", body)
            index = build_evidence_index([context], [source])
            point_id = next(span.evidence_id for span in index.spans.values() if span.kind == "point")
            output = verdict([point_id], [])
            with self.subTest(body=body):
                validate_faithfulness_result(output, [output.statements[0].statement], index)

    def test_principal_applicability_reference_still_requires_link(self):
        bodies = [
            "Thực hiện hành vi quy định tại điểm b khoản 8 Điều này bị trừ 10 điểm.",
            "Thực hiện hành vi vi phạm quy định tại điểm b khoản 8 Điều này bị trừ 10 điểm.",
            "Vi phạm quy định tại điểm b khoản 8 Điều này bị trừ 10 điểm.",
            "(Trừ các trường hợp đặc biệt) Thực hiện hành vi quy định tại điểm b khoản 8 Điều này bị trừ 10 điểm.",
            "Không áp dụng với hồ sơ đặc biệt (trừ hành vi quy định tại khoản 1); "
            "thực hiện hành vi quy định tại điểm b khoản 8 Điều này bị trừ 10 điểm.",
        ]
        for body in bodies:
            context, source = provision("S10", "13", "a", body)
            index = build_evidence_index([context], [source])
            point_id = next(span.evidence_id for span in index.spans.values() if span.kind == "point")
            output = verdict([point_id], [])
            with self.subTest(body=body), self.assertRaises(JudgeVerdictError) as caught:
                validate_faithfulness_result(output, [output.statements[0].statement], index)
            self.assertEqual(caught.exception.rule, "cross_reference_link_missing")

    def test_header_only_positive_evidence_is_rejected(self):
        header_id = next(span.evidence_id for span in self.index.spans.values() if span.kind == "header")
        self.assert_rule("positive_without_substantive_evidence", verdict([header_id], []))

    def test_negative_cross_reference_mismatch_can_include_valid_and_broken_links(self):
        rule_id = self.ids["S6"]
        valid_target, broken_target = self.ids["S1"], self.ids["S2"]
        output = verdict([rule_id, valid_target, broken_target],
                         [link(rule_id, valid_target, True), link(rule_id, broken_target, False)],
                         supported=False, decision="cross_reference_mismatch")
        audits = self.check(output)
        self.assertEqual([item["code_applies"] for item in audits[0]["rule_links"]], [True, False])
        self.assertEqual(output.statements[0].verdict, 0)

    def test_same_point_in_different_document_or_article_does_not_establish_link(self):
        rule_id = self.ids["S7"]
        for source_id in ("S8", "S9"):
            conduct_id = self.ids[source_id]
            with self.subTest(source_id=source_id):
                self.assert_rule("cross_reference_relation_mismatch", verdict(
                    [rule_id, conduct_id], [link(rule_id, conduct_id)]))
                self.check(verdict([rule_id, conduct_id], [link(rule_id, conduct_id, False)],
                                   supported=False, decision="cross_reference_mismatch"))

    def test_duplicate_links_are_errors_not_extra_support(self):
        rule_id, conduct_id = self.ids["S7"], self.ids["S2"]
        relation = link(rule_id, conduct_id)
        self.assert_rule("duplicate_rule_link", verdict([rule_id, conduct_id], [relation, relation]))

    def test_link_cannot_use_unselected_source_even_when_it_exists(self):
        rule_id, conduct_id = self.ids["S7"], self.ids["S2"]
        self.assert_rule("link_outside_evidence", verdict([rule_id], [link(rule_id, conduct_id)]))

    def test_materialized_evidence_remains_exact_original_source_text(self):
        for span in self.index.spans.values():
            self.assertTrue(any(context[span.start:span.end] == span.text for context in self.contexts))

    def test_incomplete_reference_parse_cannot_certify_negative_or_positive_missing_target(self):
        context, _ = provision("S10", "13", "g", "Thực hiện hành vi quy định tại điểm a khoản 6; "
                               "điểm b khoản này Điều này bị trừ 04 điểm.")
        index = build_evidence_index(self.contexts + [context])
        rule_id = next(span.evidence_id for span in index.spans.values()
                       if span.source_id == "S10" and span.kind == "point")
        for supported, decision, applies in ((True, "supported", True),
                                             (False, "cross_reference_mismatch", False)):
            with self.subTest(supported=supported):
                output = verdict([rule_id, self.ids["S2"]], [link(rule_id, self.ids["S2"], applies)],
                                 supported=supported, decision=decision)
                with self.assertRaises(JudgeVerdictError) as caught:
                    validate_faithfulness_result(output, [output.statements[0].statement], index)
                self.assertEqual(caught.exception.rule, "unaddressable_rule_link")

    def test_prior_point_deduction_is_not_an_exclusion_prefix(self):
        context, _ = provision("S10", "13", "g", "Ngoài việc bị trừ điểm, thực hiện hành vi "
                               "quy định tại điểm a khoản 6 Điều này bị phạt thêm 20 đơn vị.")
        index = build_evidence_index(self.contexts + [context])
        rule_id = next(span.evidence_id for span in index.spans.values()
                       if span.source_id == "S10" and span.kind == "point")
        output = verdict([rule_id], [])
        with self.assertRaises(JudgeVerdictError) as caught:
            validate_faithfulness_result(output, [output.statements[0].statement], index)
        self.assertEqual(caught.exception.rule, "cross_reference_link_missing")


if __name__ == "__main__":
    unittest.main()
