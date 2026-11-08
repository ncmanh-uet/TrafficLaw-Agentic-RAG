"""Offline checks for the three agent metrics; no paid services required."""

import asyncio
import copy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import get_args
import unittest
from unittest.mock import AsyncMock, Mock, patch

from test.evaluation_agent.citation import (
    AnswerAnalysis, Claim, ClaimVerdict, SupportVerdicts, analyze_answer,
    citation_score, clean_answer, parse_answer, prepare_metric_text,
)
from test.evaluation_agent.scoring import Scorer
from test.evaluation_agent.scoring_data import join_outputs, metric_result


def gold(category="internal_answerable"):
    return {"query_id": "q_001", "query": "Vượt đèn đỏ?", "reference_answer": "Phạt 18–20 triệu.",
            "category": category, "split": "pilot"}


def output(answer="Phạt 18–20 triệu [S1].", status="answered"):
    return {"query_id": "q_001", "variant": "baseline", "query": "Vượt đèn đỏ?", "answer": answer,
            "contexts": ["[SOURCE_ID=S1]\nPhạt 18–20 triệu."],
            "sources": [{"source_id": "S1", "dieu": "6", "khoan": "9", "diem": "b", "doc_id": "168"}],
            "status": status, "latency_ms": 1, "error": None}


def analysis(*texts, abstaining=False):
    return AnswerAnalysis(abstaining=abstaining, claims=[Claim(block_id=0, text=t) for t in texts])


def faithful_statement(text, context):
    """Deterministic fake transport selecting real catalog IDs, not copied text."""
    spans = json.loads(context)["evidence_catalog"]
    ids = [span["evidence_id"] for span in spans if text in span["text"] and span["kind"] != "header"]
    return {"statement": text, "reason": "Offline expected support", "verdict": int(bool(ids)),
            "decision": "supported" if ids else "not_found", "evidence_ids": ids, "rule_links": []}


def verdict(*ids, fully=True, claim_id=0):
    return SupportVerdicts(judgments=[ClaimVerdict(claim_id=claim_id, fully_supported=fully,
        contributing_source_ids=list(ids), reason="Đối chiếu nội dung")])


class InputTests(unittest.TestCase):
    def test_requires_both_variants_by_default(self):
        with self.assertRaisesRegex(ValueError, "Missing"):
            join_outputs([gold()], [output()])
        self.assertEqual(len(join_outputs([gold()], [output()], require_complete=False)), 1)

    def test_reject_ambiguous_or_mismatched_outputs(self):
        for rows in ([output(), output()], [{**output(), "query_id": None}],
                     [{**output(), "query": "Another query"}],
                     [{**output(), "sources": []}],
                     [{**output(), "contexts": ["No source ID"]}],
                     [{**output(), "sources": [{"source_id": "S2"}]}]):
            with self.assertRaises(ValueError):
                join_outputs([gold()], rows, require_complete=False)

    def test_nan_not_a_valid_score(self):
        for value in (float("nan"), float("inf"), -1, 2, True):
            with self.assertRaises(ValueError):
                metric_result(value)

    def test_number_prefix_not_removed_and_citations_keep_local_scope(self):
        blocks = parse_answer("18–20 triệu. [S1] Bị trừ 4 điểm [S2].\nNguồn: [S3]")
        self.assertTrue(blocks[0].text.startswith("18"))
        self.assertEqual([[c.source_id for c in b.citations] for b in blocks], [["S1"], ["S2"], ["S3"]])

    def test_web_url_and_legal_locator_are_not_sentence_fragments(self):
        answer = "Phạt [S1] [Điều 6, Khoản 9, Điểm b — 168].\nLịch [S2 — a.gov.vn](https://a.gov.vn/news)."
        blocks = parse_answer(answer)
        self.assertEqual(len(blocks), 2)
        self.assertIn("Khoản 9", blocks[0].citations[0].legal)
        self.assertEqual(blocks[1].citations[0].url, "https://a.gov.vn/news")
        cleaned = clean_answer(answer)
        self.assertIn("Điều 6, Khoản 9, Điểm b — 168", cleaned)
        self.assertNotIn("[S1]", cleaned)
        self.assertNotIn("https://", cleaned)


class CitationTests(unittest.IsolatedAsyncioTestCase):
    async def calculate(self, answer, claims=None, support=None, sources=None, contexts=None):
        judge = SimpleNamespace(generate=AsyncMock(return_value=support or verdict("S1")))
        row = output(answer)
        score = await citation_score(judge, parse_answer(answer), claims or analysis("Phạt 18–20 triệu"),
                                     contexts or row["contexts"], sources or row["sources"])
        return score, judge

    async def test_fully_supported_and_repeated_links_deduplicated(self):
        score, judge = await self.calculate("Phạt 18–20 triệu [S1] [S1].")
        self.assertEqual(score["score"], 1)
        self.assertEqual(score["total_links"], 1)
        self.assertEqual(judge.generate.await_count, 1)

    async def test_existence_is_not_support(self):
        score, _ = await self.calculate("Phạt 18–20 triệu [S1].", support=verdict(fully=False))
        self.assertEqual(score["score"], 0)

    async def test_missing_unknown_and_wrong_locator(self):
        for answer in ("Phạt 18–20 triệu.", "Phạt [S99].", "Phạt [S1] [Điều 7 — 168].",
                       "Phạt [S1] [Điều 6 — wrong-doc].", "Phạt [S1 — Điều 7 — 168]."):
            score, judge = await self.calculate(answer)
            self.assertEqual(score["score"], 0)
            judge.generate.assert_not_awaited()

    async def test_web_forged_url_not_accepted(self):
        score, judge = await self.calculate("Phạt [S1 — gov.vn](https://fake.test/).",
            sources=[{"source_id": "S1", "url": "https://gov.vn/news"}])
        self.assertEqual(score["score"], 0)
        judge.generate.assert_not_awaited()

    async def test_joint_cross_reference_support_and_extra_irrelevant_source(self):
        sources = [{"source_id": f"S{i}"} for i in range(1, 4)]
        score, _ = await self.calculate("Trừ 4 điểm với hành vi trên [S1,S2,S3].",
            support=verdict("S1", "S2"), sources=sources, contexts=["vi phạm", "trừ điểm dẫn chiếu", "khác"])
        self.assertAlmostEqual(score["precision"], 2 / 3)
        self.assertEqual(score["recall"], 1)
        self.assertAlmostEqual(score["score"], .8)

    async def test_partial_support_is_not_full_recall(self):
        score, _ = await self.calculate("Phạt và trừ điểm [S1].", support=verdict("S1", fully=False))
        self.assertEqual(score["precision"], 1)
        self.assertEqual(score["recall"], 0)
        self.assertEqual(score["score"], 0)

    async def test_orphan_and_no_claim_no_citation_na(self):
        empty = AnswerAnalysis(abstaining=True, claims=[])
        score, _ = await self.calculate("Chưa đủ thông tin.", claims=empty)
        self.assertIsNone(score["score"])
        score, _ = await self.calculate("Nguồn: [S1]", claims=empty)
        self.assertEqual(score["score"], 0)
        self.assertIsNone(score["recall"])

    async def test_only_cited_context_passed_not_gold_or_other_evidence(self):
        score, judge = await self.calculate("Phạt [S1].", sources=[{"source_id": "S1"}, {"source_id": "S2"}],
                                            contexts=["CITED CONTENT", "SECRET UNCITED CONTENT"])
        self.assertEqual(score["score"], 1)
        prompt = judge.generate.call_args.args[0]
        self.assertIn("CITED CONTENT", prompt)
        self.assertNotIn("SECRET UNCITED", prompt)

    async def test_judge_must_not_invent_source_or_omit_claim(self):
        for support in (verdict("S99"), SupportVerdicts(judgments=[])):
            with self.assertRaises(ValueError):
                await self.calculate("Phạt [S1].", support=support)

    async def test_claim_extraction_rejects_invented_blocks(self):
        judge = SimpleNamespace(generate=AsyncMock(return_value=AnswerAnalysis(abstaining=False,
            claims=[Claim(block_id=99, text="fake")])) )
        with self.assertRaises(ValueError):
            await analyze_answer(judge, "query", parse_answer("Phạt."))


class ScorerTests(unittest.IsolatedAsyncioTestCase):
    def setup_scorer(self, parsed=None):
        self.judge = SimpleNamespace(generate=AsyncMock(side_effect=[parsed or analysis("Phạt 18–20 triệu"), verdict("S1")]))
        self.metrics = {name: SimpleNamespace(ascore=AsyncMock(return_value=SimpleNamespace(value=.75)))
                        for name in ("factual_correctness", "faithfulness")}
        return Scorer(self.judge, self.metrics)

    async def test_three_metrics_save_independently_and_use_actual_context(self):
        scorer = self.setup_scorer()
        checkpoints = []
        row = await scorer.score(gold(), output(), checkpoint=checkpoints.append, diagnostics=True)
        self.assertEqual([len(c["metrics"]) for c in checkpoints], [1, 2, 3])
        self.assertEqual(row["metrics"]["citation_f1"]["score"], 1)
        self.assertEqual(self.metrics["faithfulness"].ascore.call_args.kwargs["retrieved_contexts"], output()["contexts"])
        self.assertEqual(self.metrics["factual_correctness"].ascore.call_args.kwargs["reference"], gold()["reference_answer"])

    async def test_expected_refusal_na_but_legal_claims_still_grounded(self):
        empty = AnswerAnalysis(abstaining=True, claims=[])
        scorer = self.setup_scorer(empty)
        row = await scorer.score(gold("insufficient_evidence"), output("Không đủ dữ liệu.", "refused"))
        self.assertTrue(all(v["status"] == "na" for v in row["metrics"].values()))
        scorer = self.setup_scorer(analysis("Phạt 18–20 triệu", abstaining=True))
        row = await scorer.score(gold("insufficient_evidence"), output(status="refused"))
        self.assertEqual(row["metrics"]["faithfulness"]["score"], .75)
        self.assertTrue(row["review_required"])

    async def test_unexpected_refusal_zero_and_runtime_error_not_zero(self):
        scorer = self.setup_scorer(AnswerAnalysis(abstaining=True, claims=[]))
        row = await scorer.score(gold(), output("Không đủ thông tin", "refused"))
        self.assertEqual(row["metrics"]["factual_correctness"]["score"], 0)
        self.assertEqual(row["metrics"]["faithfulness"]["status"], "na")
        scorer = self.setup_scorer()
        row = await scorer.score(gold(), output("", "error"))
        self.judge.generate.assert_not_awaited()
        self.assertTrue(all(v["score"] is None and v["status"] == "input_error" for v in row["metrics"].values()))

    async def test_resume_only_failed_metric_and_cached_analysis(self):
        scorer = self.setup_scorer()
        self.metrics["faithfulness"].ascore.side_effect = TimeoutError("KEY MUST NEVER BE SAVED")
        row = await scorer.score(gold(), output(), diagnostics=True)
        self.assertEqual(row["metrics"]["faithfulness"]["reason"], "TimeoutError")
        scorer = self.setup_scorer()
        row = await scorer.score(gold(), output(), previous=row, diagnostics=True)
        self.metrics["factual_correctness"].ascore.assert_not_awaited()
        self.judge.generate.assert_not_awaited()
        self.assertEqual(row["metrics"]["faithfulness"]["status"], "ok")

    async def test_nan_from_ragas_saved_as_error(self):
        scorer = self.setup_scorer()
        self.metrics["faithfulness"].ascore.return_value = SimpleNamespace(value=float("nan"))
        row = await scorer.score(gold(), output())
        self.assertEqual(row["metrics"]["faithfulness"]["status"], "judge_error")

    async def test_empty_answer_zero_no_judge_calls(self):
        scorer = self.setup_scorer()
        row = await scorer.score(gold(), output(""))
        self.judge.generate.assert_not_awaited()
        self.assertEqual(row["metrics"]["factual_correctness"]["score"], 0)
        self.assertIsNone(row["metrics"]["citation_f1"]["score"])


class ScoringFileTests(unittest.IsolatedAsyncioTestCase):
    async def test_checkpoints_resume_and_input_config_mismatch(self):
        from test.evaluation_agent.score import run

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path, input_path, scores_path = (root / name for name in ("dataset.jsonl", "outputs.jsonl", "scores.jsonl"))
            dataset_path.write_text(json.dumps(gold()) + "\n", encoding="utf-8")
            input_path.write_text(json.dumps(output()) + "\n", encoding="utf-8")
            args = SimpleNamespace(dataset=dataset_path, input=input_path, output=scores_path,
                split="pilot", allow_partial=True, check_only=False, model="gemini-test",
                resume=False, include_details=True)
            fake_judge = SimpleNamespace(generate=AsyncMock(side_effect=[analysis("Phạt"), verdict("S1")]), close=AsyncMock())
            metrics = {name: SimpleNamespace(ascore=AsyncMock(return_value=SimpleNamespace(value=1.0)))
                       for name in ("factual_correctness", "faithfulness")}
            with patch("test.evaluation_agent.score.GeminiJudge", return_value=fake_judge), \
                 patch("test.evaluation_agent.score.make_ragas_metrics", return_value=metrics), \
                 patch("test.evaluation_agent.score.version", return_value="test"):
                self.assertEqual(await run(args), 0)
                saved = scores_path.read_bytes()
                args.resume = True
                self.assertEqual(await run(args), 0)
                self.assertEqual(scores_path.read_bytes(), saved)
                self.assertEqual(fake_judge.generate.await_count, 2)  # no calls on complete resume
                args.model = "different-model"
                with self.assertRaisesRegex(ValueError, "Resume refused"):
                    await run(args)
                self.assertEqual(scores_path.read_bytes(), saved)
                args.model = "gemini-test"
                changed = {**output(), "answer": "Changed answer [S1]."}
                input_path.write_text(json.dumps(changed) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Resume refused"):
                    await run(args)

    async def test_check_only_never_constructs_judge_or_creates_output(self):
        from test.evaluation_agent.score import run

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path, input_path = root / "dataset.jsonl", root / "outputs.jsonl"
            dataset_path.write_text(json.dumps(gold()) + "\n", encoding="utf-8")
            input_path.write_text(json.dumps(output()) + "\n", encoding="utf-8")
            args = SimpleNamespace(dataset=dataset_path, input=input_path, split="pilot", allow_partial=True,
                                   check_only=True, output=root / "not-created.jsonl")
            with patch("test.evaluation_agent.score.GeminiJudge") as constructor:
                self.assertEqual(await run(args), 0)
                constructor.assert_not_called()
            self.assertFalse(args.output.exists())


class GeminiTransportTests(unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(importlib.util.find_spec("google.genai"), "Gemini SDK required")
    async def test_native_json_transport_and_sdk_retries_disabled(self):
        from test.evaluation_agent.judge import GeminiJudge, JudgeConfig

        client = SimpleNamespace(aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=AsyncMock(return_value=SimpleNamespace(
                text=AnswerAnalysis(abstaining=False, claims=[]).model_dump_json()))),
            aclose=AsyncMock()), close=Mock())
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-not-a-real-key"}), \
             patch("google.genai.Client", return_value=client) as constructor:
            judge = GeminiJudge(JudgeConfig(model="gemini-test"))
            parsed = await judge.generate("test prompt", AnswerAnalysis)
            self.assertEqual(parsed.claims, [])
            self.assertEqual(constructor.call_args.kwargs["http_options"].retry_options.attempts, 1)
            request = client.aio.models.generate_content.call_args.kwargs
            self.assertEqual(request["config"].response_mime_type, "application/json")
            self.assertIsNone(request["config"].response_schema)
            self.assertEqual(request["config"].response_json_schema, AnswerAnalysis.model_json_schema())
            self.assertTrue(request["config"].automatic_function_calling.disable)
            await judge.close()

    @unittest.skipUnless(importlib.util.find_spec("google.genai"), "Gemini SDK required")
    async def test_permanent_api_errors_not_retried(self):
        from test.evaluation_agent.judge import GeminiJudge, JudgeConfig

        error = RuntimeError("redacted")
        error.code = 403
        client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(
            generate_content=AsyncMock(side_effect=error))))
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-not-a-real-key"}), \
             patch("google.genai.Client", return_value=client):
            judge = GeminiJudge(JudgeConfig(model="gemini-test"))
            with self.assertRaises(RuntimeError):
                await judge.generate("test", AnswerAnalysis)
            self.assertEqual(client.aio.models.generate_content.await_count, 1)


@unittest.skipUnless(importlib.util.find_spec("ragas"), "Install the root requirements.txt for Ragas tests")
class RagasAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_ragas_algorithms_with_fake_gemini_transport(self):
        from test.evaluation_agent.judge import make_ragas_metrics

        async def generate(prompt, model):
            fields = model.model_fields
            if "claims" in fields:
                return model(claims=["Phạt 18–20 triệu."])
            if "statements" in fields:
                if get_args(fields["statements"].annotation) == (str,):
                    return model(statements=["Phạt 18–20 triệu."])
                if model.__name__ == "FaithfulnessNLIOutput":
                    data = json.loads(prompt.rsplit("input: ", 1)[1].rsplit("\nOutput:", 1)[0])
                    return model(statements=[faithful_statement("Phạt 18–20 triệu.", data["context"])])
                return model(statements=[{"statement": "Phạt 18–20 triệu.", "reason": "Có căn cứ", "verdict": 1,
                                          }])
            raise AssertionError(model)
        metrics = make_ragas_metrics(SimpleNamespace(generate=generate))
        factual = await metrics["factual_correctness"].ascore(response="Phạt 18–20 triệu.", reference="Phạt 18–20 triệu.")
        faithful = await metrics["faithfulness"].ascore(user_input="Mức phạt?", response="Phạt 18–20 triệu.",
                                                      retrieved_contexts=["Phạt 18–20 triệu."])
        self.assertEqual(factual.value, 1)
        self.assertEqual(faithful.value, 1)

    async def test_real_ragas_wrong_and_partially_supported_answers(self):
        from test.evaluation_agent.judge import make_ragas_metrics

        async def generate(prompt, model):
            payload = prompt.rsplit("input: ", 1)[1].rsplit("\nOutput:", 1)[0]
            data = json.loads(payload)
            if "statements" in data:
                if model.__name__ == "FaithfulnessNLIOutput":
                    return model(statements=[faithful_statement(text, data["context"]) for text in data["statements"]])
                return model(statements=[{"statement": text, "reason": "Offline expected support",
                    "verdict": int(text in data["context"])} for text in data["statements"]])
            field = "statements" if "answer" in data else "claims"
            text = data.get("answer", data.get("response", ""))
            return model(**{field: text.split(";")})

        metrics = make_ragas_metrics(SimpleNamespace(generate=generate))
        for answer, expected_factual, expected_faithful in (
            ("Phạt 20 triệu", 0.0, 0.0),
            ("Phạt 18 triệu;Trừ 4 điểm", .67, .5),
        ):
            with self.subTest(answer=answer):
                f = await metrics["factual_correctness"].ascore(response=answer, reference="Phạt 18 triệu")
                h = await metrics["faithfulness"].ascore(user_input="Mức phạt?", response=answer,
                                                       retrieved_contexts=["Phạt 18 triệu"])
                self.assertAlmostEqual(f.value, expected_factual)
                self.assertAlmostEqual(h.value, expected_faithful)

    async def test_adapter_keeps_ragas_examples_and_does_not_add_retry(self):
        from ragas.metrics import FactualCorrectness
        from test.evaluation_agent.judge import make_ragas_metrics

        judge = SimpleNamespace(generate=AsyncMock(side_effect=TimeoutError("offline")))
        metrics = make_ragas_metrics(judge)
        expected = FactualCorrectness(mode="f1", atomicity="high", coverage="high")
        actual = metrics["factual_correctness"].metric
        self.assertEqual(actual.claim_decomposition_prompt.examples, expected.claim_decomposition_prompt.examples)
        self.assertTrue(actual.claim_decomposition_prompt.instruction.startswith(expected.claim_decomposition_prompt.instruction))
        self.assertIn("evaluate_legal_authority", actual.claim_decomposition_prompt.instruction)
        self.assertEqual(actual.mode, "f1")
        with self.assertRaises(TimeoutError):
            await metrics["factual_correctness"].ascore(response="Phạt 18 triệu", reference="Phạt 18 triệu")
        self.assertEqual(judge.generate.await_count, 1)


class MetricTextScopeTests(unittest.TestCase):
    query = "Xe mô tô vi phạm nồng độ cồn bị phạt tiền và trừ bao nhiêu điểm?"

    def prepare(self, text, *, metric="factual_correctness", query=None):
        return prepare_metric_text(text, self.query if query is None else query, metric=metric)

    def test_q012_substantive_facts_do_not_depend_on_appended_citations(self):
        plain = (
            "### Nồng độ cồn vượt 50 đến 80 mg/100 ml máu hoặc vượt 0,25 đến 0,4 mg/l khí thở\n"
            "* Người điều khiển xe mô tô bị phạt từ 6.000.000 đến 8.000.000 đồng.\n"
            "* Người điều khiển xe mô tô bị trừ 10 điểm giấy phép lái xe."
        )
        annotated = plain.replace(
            "8.000.000 đồng.", "8.000.000 đồng [S5] [Điều 7, Khoản 8, Điểm b — 168/2024/NĐ-CP]."
        ).replace(
            "10 điểm giấy phép lái xe.",
            "10 điểm giấy phép lái xe [S16] [Điều 7, Khoản 13, Điểm d — 168/2024/NĐ-CP].",
        )
        self.assertEqual(self.prepare(annotated), self.prepare(plain))
        self.assertIn("vượt 50 đến 80", self.prepare(annotated))
        self.assertIn("trừ 10 điểm", self.prepare(annotated))

    def test_q017_checklist_does_not_gain_claims_from_locator_annotations(self):
        query = "Hành khách đi xe vận tải hành khách có quyền và nghĩa vụ gì?"
        plain = (
            "* Hành khách có quyền được nhận vé theo quy định.\n"
            "* Hành khách phải chấp hành hướng dẫn của người lái xe.\n"
            "* Hành khách phải thanh toán tiền cước theo giá niêm yết."
        )
        annotated = plain.replace(".", " [S1] [Điều 60 — 35/2024/QH15].")
        self.assertEqual(self.prepare(annotated, query=query), self.prepare(plain, query=query))

    def test_combined_standalone_and_parenthesized_attribution_are_removed(self):
        plain = "Người điều khiển xe mô tô bị trừ 10 điểm giấy phép lái xe."
        locators = (
            "[S16] [Điều 7, Khoản 13, Điểm d — 168/2024/NĐ-CP]",
            "[S16 — Điều 7, Khoản 13, Điểm d — 168/2024/NĐ-CP]",
            "[Điều 7, Khoản 13, Điểm d — 168/2024/NĐ-CP]",
            "(Điều 7, Khoản 13, Điểm d — 168/2024/NĐ-CP)",
        )
        for locator in locators:
            with self.subTest(locator=locator):
                self.assertEqual(self.prepare(plain[:-1] + " " + locator + "."), self.prepare(plain))

    def test_wrong_amount_points_conditions_and_negation_are_not_erased(self):
        correct = "Vượt 50 đến 80 mg/100 ml máu: phạt 6–8 triệu đồng và trừ 10 điểm."
        wrong = (
            correct.replace("10 điểm", "04 điểm"),
            correct.replace("6–8 triệu", "2–3 triệu"),
            correct.replace("50 đến 80", "80 đến 100"),
            correct.replace("trừ 10 điểm", "không trừ 10 điểm"),
        )
        for text in wrong:
            with self.subTest(text=text):
                prepared = self.prepare(text + " [S1] [Điều 7 — 168/2024/NĐ-CP]")
                self.assertNotEqual(prepared, self.prepare(correct))
                self.assertIn(text, prepared)

    def test_legal_references_in_substantive_prose_are_never_deleted(self):
        text = (
            "Điểm b khoản 8 Điều 7 quy định hành vi vi phạm ở mức này. "
            "Điểm d khoản 13 Điều 7 quy định trừ 10 điểm đối với hành vi đó."
        )
        self.assertEqual(self.prepare(text), text)

    def test_authority_question_preserves_its_answered_locator(self):
        text = "Quy định nằm tại [S1] [Điều 69, Khoản 3 — 35/2024/QH15]."
        for metric in ("factual_correctness", "faithfulness"):
            with self.subTest(metric=metric):
                prepared = self.prepare(text, metric=metric,
                    query="Quy định về thiết bị này nằm tại điều, khoản nào?")
                self.assertIn("Điều 69", prepared)
                self.assertIn("Khoản 3", prepared)
                self.assertIn("35/2024/QH15", prepared)
                self.assertNotIn("[S1]", prepared)

    def test_exact_article_interpretation_preserves_legal_prose(self):
        query = "Khoản 3 Điều 69 Luật Đường bộ quy định gì?"
        text = "Khoản 3 Điều 69 quy định thiết bị giám sát hành trình phải bảo đảm các yêu cầu."
        self.assertEqual(self.prepare(text, query=query), text)

    def test_q004_attribution_is_removed_but_obligation_is_retained(self):
        query = "Thiết bị giám sát hành trình phải bảo đảm yêu cầu gì?"
        text = "Thiết bị phải lưu trữ thông tin hành trình [S1] [Điều 69, Khoản 3 — 35/2024/QH15]."
        prepared = self.prepare(text, query=query)
        self.assertEqual(prepared, "Thiết bị phải lưu trữ thông tin hành trình.")

    def test_named_article_in_question_does_not_make_every_locator_a_required_fact(self):
        query = "Theo Điều 69 Luật Đường bộ, xe nào phải lắp thiết bị?"
        text = "Xe ô tô kinh doanh vận tải phải lắp thiết bị [S1] [Điều 69, Khoản 3 — 35/2024/QH15]."
        self.assertEqual(self.prepare(text, query=query), "Xe ô tô kinh doanh vận tải phải lắp thiết bị.")

    def test_actual_q004_query_does_not_request_another_locator_claim(self):
        query = ("Theo Điều 69 Luật Đường bộ 2024, xe ô tô cứu thương vận tải người bệnh "
                 "phải có các dấu hiệu, thông tin và thiết bị nào?")
        text = ("Xe ô tô cứu thương vận tải người bệnh phải có thiết bị giám sát hành trình "
                "[S1] [Điều 69, Khoản 3 — 35/2024/QH15].")
        for metric in ("factual_correctness", "faithfulness"):
            with self.subTest(metric=metric):
                self.assertEqual(self.prepare(text, query=query, metric=metric),
                    "Xe ô tô cứu thương vận tải người bệnh phải có thiết bị giám sát hành trình.")

    def test_faithfulness_does_not_turn_citation_attribution_into_rule_content(self):
        text = "Người điều khiển xe mô tô bị trừ 04 điểm [S15] [Điều 7, Khoản 13, Điểm b — 168/2024/NĐ-CP]."
        prepared = self.prepare(text, metric="faithfulness")
        self.assertEqual(prepared, "Người điều khiển xe mô tô bị trừ 04 điểm.")
        self.assertNotIn("10 điểm", prepared)

    def test_substantive_parentheses_dates_and_layout_survive(self):
        text = (
            "### Áp dụng từ 01/01/2025\n"
            "* Không áp dụng cho xe mô tô (kể cả xe máy điện).\n"
            "* Nồng độ không quá 0,4 mg/l khí thở (tương ứng 80 mg/100 ml máu)."
        )
        self.assertEqual(self.prepare(text), text)

    def test_preparation_is_idempotent_for_both_metric_scopes(self):
        text = "### Mức 2\n* Trừ **10 điểm** [S16] [Điều 7, Khoản 13, Điểm d — 168/2024/NĐ-CP]."
        for metric in ("factual_correctness", "faithfulness"):
            with self.subTest(metric=metric):
                prepared = self.prepare(text, metric=metric)
                self.assertEqual(self.prepare(prepared, metric=metric), prepared)

    def test_unknown_metric_fails_instead_of_silently_changing_scope(self):
        with self.assertRaises(ValueError):
            self.prepare("Trừ 10 điểm.", metric="citation_f1")

class ScorerMetricRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_metric_gets_its_own_prepared_input_and_citations_remain_raw(self):
        query = "Xe mô tô vi phạm nồng độ cồn bị xử phạt như thế nào?"
        answer = "Trừ 04 điểm [S15] [Điều 7, Khoản 13, Điểm b — 168/2024/NĐ-CP]."
        reference = "Trừ 10 điểm [Điều 7, Khoản 13, Điểm d — 168/2024/NĐ-CP]."
        gold = {"query_id": "q_012", "query": query, "reference_answer": reference,
                "category": "internal_answerable", "split": "pilot"}
        output = {
            "query_id": "q_012", "variant": "baseline", "query": query, "answer": answer,
            "contexts": ["[SOURCE_ID=S15] Điểm b khoản 13 chỉ dẫn điểm a khoản 6."],
            "sources": [{"source_id": "S15", "doc_id": "168/2024/NĐ-CP",
                         "dieu": "7", "khoan": "13", "diem": "b"}],
            "status": "citation_invalid", "latency_ms": 1, "error": None,
        }
        before = copy.deepcopy((gold, output))
        factual = SimpleNamespace(ascore=AsyncMock(return_value=SimpleNamespace(value=.5)))
        faithfulness = SimpleNamespace(ascore=AsyncMock(return_value=SimpleNamespace(value=0)))
        parsed = AnswerAnalysis(abstaining=False, claims=[Claim(block_id=0, text="Trừ 04 điểm")])
        judge = SimpleNamespace(generate=AsyncMock())
        citation = AsyncMock(return_value={"score": 0.0})

        def prepare(text, provided_query, *, metric):
            self.assertEqual(provided_query, query)
            return metric + "::" + text

        with patch("test.evaluation_agent.scoring.prepare_metric_text", side_effect=prepare) as preparer, \
             patch("test.evaluation_agent.scoring.analyze_answer", return_value=parsed), \
             patch("test.evaluation_agent.scoring.citation_score", citation):
            row = await Scorer(judge, {"factual_correctness": factual,
                                      "faithfulness": faithfulness}).score(gold, output)

        self.assertEqual(preparer.call_count, 3)
        self.assertEqual(factual.ascore.call_args.kwargs, {
            "user_input": query,
            "response": "factual_correctness::" + answer,
            "reference": "factual_correctness::" + reference,
        })
        self.assertEqual(faithfulness.ascore.call_args.kwargs, {
            "user_input": query, "response": "faithfulness::" + answer,
            "retrieved_contexts": output["contexts"],
            "evidence_sources": output["sources"],
        })
        raw_blocks = citation.call_args.args[1]
        self.assertEqual(raw_blocks, parse_answer(answer))
        self.assertEqual(raw_blocks[0].citations[0].source_id, "S15")
        self.assertIn("Khoản 13, Điểm b", raw_blocks[0].citations[0].legal)
        self.assertEqual(citation.call_args.args[3:], (output["contexts"], output["sources"]))
        self.assertEqual(row["metrics"]["citation_f1"]["score"], 0)
        self.assertEqual((gold, output), before)
        judge.generate.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
