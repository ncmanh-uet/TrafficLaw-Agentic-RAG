"""Offline schema/evidence checks, not proof of model legal entailment."""

import copy
import json
import unittest

from pydantic import ValidationError

from test.evaluation_agent.evidence_spans import build_evidence_index
from test.evaluation_agent.faithfulness import (
    FAITHFULNESS_INSTRUCTION,
    FaithfulnessNLIOutput,
    JudgeVerdictError,
    synthetic_example,
    validate_faithfulness_result,
)


def result(statement="Một mệnh đề.", *, verdict=1, evidence_ids=None, decision=None, rule_links=None):
    return FaithfulnessNLIOutput(statements=[{
        "statement": statement,
        "reason": "Đối chiếu nguồn được cung cấp.",
        "verdict": verdict,
        "evidence_ids": ["E1"] if evidence_ids is None else evidence_ids,
        "decision": decision or ("supported" if verdict == 1 else "not_found"),
        "rule_links": [] if rule_links is None else rule_links,
    }])


def span_id(index, fragment):
    matches = [span for span in index.spans.values() if fragment in span.text]
    if not matches:
        raise AssertionError(f"No deterministic source span for {fragment!r}")
    return min(matches, key=lambda span: len(span.text)).evidence_id


class FaithfulnessSchemaTests(unittest.TestCase):
    def test_required_structural_fields_for_all_verdicts(self):
        for verdict in (0, 1):
            payload = result(verdict=verdict).model_dump()
            for field in ("evidence_ids", "decision", "rule_links"):
                missing = copy.deepcopy(payload)
                del missing["statements"][0][field]
                with self.subTest(verdict=verdict, field=field), self.assertRaises(ValidationError):
                    FaithfulnessNLIOutput.model_validate(missing)

    def test_verdict_is_strict_binary_integer(self):
        for verdict in (True, False, "1", "0", 1.0, 0.0, -1, 2):
            with self.subTest(verdict=verdict), self.assertRaises(ValidationError):
                result(verdict=verdict)
        for verdict in (0, 1):
            self.assertEqual(result(verdict=verdict).statements[0].verdict, verdict)

    def test_link_relation_is_strict_boolean(self):
        for applies in (0, 1, "true", "false", None):
            with self.subTest(applies=applies), self.assertRaises(ValidationError):
                result(rule_links=[{"rule_id": "E1", "target_id": "E2", "applies": applies}])
        for applies in (True, False):
            item = result(rule_links=[{"rule_id": "E1", "target_id": "E2", "applies": applies}])
            self.assertIs(item.statements[0].rule_links[0].applies, applies)

    def test_unknown_fields_and_legacy_model_copied_quotes_rejected(self):
        payload = result().model_dump()
        for field, value in (("score", 1.0), ("evidence_quotes", ["Invented copied quote"])):
            unknown = copy.deepcopy(payload)
            unknown["statements"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                FaithfulnessNLIOutput.model_validate(unknown)
        payload["score"] = 1.0
        with self.assertRaises(ValidationError):
            FaithfulnessNLIOutput.model_validate(payload)

    def test_unknown_decision_rejected(self):
        with self.assertRaises(ValidationError):
            result(decision="model_confidence")


class FaithfulnessEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.contexts = ["Trước. Một nguồn. Sau."]
        self.index = build_evidence_index(self.contexts)
        self.evidence_id = span_id(self.index, "Một nguồn.")

    def assert_rule(self, rule, output, statements=None, index=None):
        with self.assertRaises(JudgeVerdictError) as caught:
            validate_faithfulness_result(
                output, ["Một mệnh đề."] if statements is None else statements,
                self.index if index is None else index)
        self.assertEqual(caught.exception.rule, rule)
        self.assertEqual(str(caught.exception), "Invalid judge verdict: " + rule)

    def test_positive_selected_span_and_negative_absence_without_evidence(self):
        output = result(evidence_ids=[self.evidence_id])
        before = output.model_dump()
        audits = validate_faithfulness_result(output, ["Một mệnh đề."], self.index)
        self.assertIsInstance(audits, list)
        self.assertEqual(output.model_dump(), before)  # Never overwrite the model's verdict.
        self.assertEqual(audits[0]["evidence"][0]["text"], self.index.spans[self.evidence_id].text)
        self.assertEqual(audits[0]["evidence"][0]["start"], self.index.spans[self.evidence_id].start)
        validate_faithfulness_result(result(verdict=0, evidence_ids=[]),
                                     ["Một mệnh đề."], build_evidence_index([]))

    def test_missing_extra_reordered_and_reworded_statements_rejected(self):
        expected = ["Thứ nhất.", "Thứ hai."]
        aligned = FaithfulnessNLIOutput(statements=[
            result(statement, verdict=0, evidence_ids=[]).statements[0] for statement in expected])
        validate_faithfulness_result(aligned, expected, self.index)
        variants = [[], aligned.statements[:1], aligned.statements * 2,
                    list(reversed(aligned.statements))]
        altered = copy.deepcopy(aligned.statements)
        altered[0].statement = "Thứ nhất!"
        variants.append(altered)
        for statements in variants:
            with self.subTest(statements=[item.statement for item in statements]):
                self.assert_rule("statement_alignment_mismatch",
                                 FaithfulnessNLIOutput(statements=statements), expected)

    def test_positive_requires_nonempty_evidence(self):
        self.assert_rule("positive_without_evidence", result(evidence_ids=[]))

    def test_unknown_evidence_ids_rejected_for_positive_and_negative(self):
        for verdict in (0, 1):
            for evidence_id in ("", " ", "NONEXISTENT_EVIDENCE_ID"):
                with self.subTest(verdict=verdict, evidence_id=evidence_id):
                    self.assert_rule("unknown_evidence_id",
                                     result(verdict=verdict, evidence_ids=[evidence_id]))

    def test_duplicate_evidence_ids_rejected(self):
        self.assert_rule("duplicate_evidence_id",
                         result(evidence_ids=[self.evidence_id, self.evidence_id]))

    def test_supported_decision_and_binary_verdict_must_agree(self):
        for verdict, decision in ((0, "supported"), (1, "not_found"),
                                  (1, "contradicted"), (1, "cross_reference_mismatch")):
            with self.subTest(verdict=verdict, decision=decision):
                self.assert_rule("invalid_decision_verdict", result(
                    verdict=verdict, evidence_ids=[self.evidence_id], decision=decision))

    def test_conflict_or_cross_reference_mismatch_requires_actual_evidence(self):
        for decision in ("contradicted", "cross_reference_mismatch"):
            with self.subTest(decision=decision):
                self.assert_rule("negative_without_evidence", result(
                    verdict=0, decision=decision, evidence_ids=[]))

    def test_link_endpoints_must_be_selected_evidence(self):
        self.assert_rule("link_outside_evidence", result(evidence_ids=[self.evidence_id], rule_links=[{
            "rule_id": self.evidence_id, "target_id": "UNSELECTED", "applies": True}]))

    def test_span_selection_cannot_invent_or_join_source_text(self):
        contexts = ["điểm b khoản 2 xác định điều kiện.", "Điểm b khoản 6 quy định hệ quả."]
        index = build_evidence_index(contexts)
        self.assertTrue(index.spans)
        for span in index.spans.values():
            self.assertTrue(any(span.text == source[span.start:span.end] for source in contexts))
            self.assertNotEqual(span.text, " ".join(contexts))
        payload = result(evidence_ids=[next(iter(index.spans))]).model_dump()
        payload["statements"][0]["evidence_quotes"] = [" ".join(contexts)]
        with self.assertRaises(ValidationError):
            FaithfulnessNLIOutput.model_validate(payload)

    def test_source_spans_preserve_original_letters_numbers_signs_and_whitespace(self):
        source = "điểm a khoản 8 bị trừ 4 điểm; x > 5, trước 01/06/2030.\n  Dòng sau."
        index = build_evidence_index([source])
        self.assertTrue(index.spans)
        for span in index.spans.values():
            self.assertEqual(span.text, source[span.start:span.end])
        self.assertEqual(list(index.spans), list(build_evidence_index([source]).spans))

    def test_empty_input_can_have_empty_result_without_creating_a_score(self):
        audits = validate_faithfulness_result(FaithfulnessNLIOutput(statements=[]), [], self.index)
        self.assertEqual(audits, [])

    def test_failure_contains_static_rule_not_untrusted_data(self):
        self.assert_rule("unknown_evidence_id", result(evidence_ids=["SECRET SOURCE TEXT"]))

    def test_synthetic_example_has_positive_and_counterexamples_without_pilot_answers(self):
        data, output = synthetic_example()
        self.assertEqual(set(data), {"context", "statements"})
        self.assertIn(1, [item.verdict for item in output.statements])
        self.assertIn(0, [item.verdict for item in output.statements])
        self.assertTrue(output.statements[0].evidence_ids)
        self.assertTrue(output.statements[0].rule_links)
        self.assertNotIn("evidence_quotes", output.model_dump_json())
        for phrase in ("document identity", "point letter", "cross-reference", "evidence_ids", "rule_links"):
            self.assertIn(phrase, FAITHFULNESS_INSTRUCTION)
        self.assertNotIn("q_012", json.dumps(data, ensure_ascii=False) + FAITHFULNESS_INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
