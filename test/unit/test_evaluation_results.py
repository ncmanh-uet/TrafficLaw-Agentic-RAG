from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test.evaluation_agent import AgentRunner, BaselineRunner, ComparisonConfig, JsonlResultWriter
from test.evaluation_agent.__main__ import main


def result(variant="baseline", status="answered"):
    return {"variant": variant, "query": "Ô tô vượt đèn đỏ?", "answer": "Có căn cứ [S1].",
            "contexts": ["[SOURCE_ID=S1] Bằng chứng tiếng Việt\ncó hai dòng."],
            "sources": [{"source_id": "S1", "chunk_id": "c1"}],
            "status": status, "latency_ms": 12.3, "error": None}


class ResultPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "results" / "outputs.jsonl"

    def read(self):
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]

    def test_append_both_variants_reopen_utf8_and_preserve_previous_rows(self):
        first = result()
        with JsonlResultWriter(self.path) as writer:
            writer.append(first, query_id="q_001")
            # Completed rows are flushed even before the context manager closes.
            self.assertEqual(self.read()[0]["answer"], first["answer"])
        previous = self.path.read_bytes()
        with JsonlResultWriter(self.path) as writer:
            writer.append(result("agent"), query_id="q_001")
        self.assertTrue(self.path.read_bytes().startswith(previous))
        self.assertEqual(len(self.read()), 2)
        self.assertEqual([r["variant"] for r in self.read()], ["baseline", "agent"])
        self.assertEqual(self.read()[0], {"query_id": "q_001", **first})
        self.assertNotIn("query_id", first)  # Do not mutate the runner result.
        self.assertIn("Bằng chứng".encode(), self.path.read_bytes())

    def test_failure_result_is_saved_as_an_ordinary_record(self):
        failure = {**result(status="error"), "answer": "", "contexts": [], "sources": [],
                   "error": "RuntimeError: LLM unavailable"}
        with JsonlResultWriter(self.path) as writer:
            writer.append(failure)
        self.assertEqual(self.read()[0]["status"], "error")
        self.assertIsNone(self.read()[0]["query_id"])

    def test_existing_dataset_or_malformed_output_is_rejected_without_changes(self):
        self.path.parent.mkdir()
        for text in ('{"query_id":"q_001","query":"dataset"}\n', '{"broken":\n',
                     json.dumps({"query_id": None, **result()}, ensure_ascii=False)):
            # Test fixtures, never actual dataset files.
            self.path.write_text(text, encoding="utf-8")
            previous = self.path.read_bytes()
            with self.assertRaises(ValueError):
                with JsonlResultWriter(self.path):
                    self.fail("Invalid output must not be accepted")
            self.assertEqual(self.path.read_bytes(), previous)

    def test_invalid_result_does_not_append_a_partial_line(self):
        with JsonlResultWriter(self.path) as writer:
            writer.append(result())
            previous = self.path.read_bytes()
            for invalid in ({**result(), "latency_ms": float("nan")},
                            {**result(), "sources": []}):
                with self.assertRaises(ValueError):
                    writer.append(invalid)
                self.assertEqual(self.path.read_bytes(), previous)

    def test_cli_saves_by_default_and_does_not_add_trace_unless_requested(self):
        runner = Mock()
        runner.run.return_value = result()
        with patch("test.evaluation_agent.__main__.DEFAULT_OUTPUT", self.path), \
             patch("test.evaluation_agent.__main__.ComparisonConfig.from_env", return_value=object()), \
             patch("test.evaluation_agent.__main__.build_runner", return_value=runner), \
             redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()) as stderr:
            code = main(["baseline", "Ô tô vượt đèn đỏ?", "--query-id", "q_001"])
        self.assertEqual(code, 0)
        self.assertEqual(self.read()[0]["query_id"], "q_001")
        self.assertNotIn("execution_trace", self.read()[0])
        self.assertEqual(json.loads(stdout.getvalue()), result())
        self.assertIn(str(self.path.resolve()), stderr.getvalue())

    def test_cli_can_save_full_diagnostics_and_returns_failure_exit_code(self):
        runner = SimpleNamespace(run=Mock(return_value=result("agent", "error")),
                                 diagnostics={"generation_attempts": [], "execution_trace": [{"node": "retrieve"}]})
        with patch("test.evaluation_agent.__main__.ComparisonConfig.from_env", return_value=object()), \
             patch("test.evaluation_agent.__main__.build_runner", return_value=runner), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = main(["agent", "Ô tô vượt đèn đỏ?", "--output", str(self.path), "--include-trace"])
        self.assertEqual(code, 1)
        self.assertEqual(self.read()[0]["execution_trace"], [{"node": "retrieve"}])

    def test_invalid_destination_fails_before_component_initialization(self):
        self.path.parent.mkdir()
        self.path.write_text('{"query":"dataset"}\n', encoding="utf-8")
        with patch("test.evaluation_agent.__main__.ComparisonConfig.from_env", return_value=object()), \
             patch("test.evaluation_agent.__main__.build_runner") as builder:
            with self.assertRaises(ValueError):
                main(["baseline", "Câu hỏi", "--output", str(self.path)])
            builder.assert_not_called()

    def test_show_config_and_help_do_not_create_results_or_load_services(self):
        config = SimpleNamespace(as_dict=lambda: {"enable_reranker": False})
        with patch("test.evaluation_agent.__main__.DEFAULT_OUTPUT", self.path), \
             patch("test.evaluation_agent.__main__.ComparisonConfig.from_env", return_value=config), \
             patch("test.evaluation_agent.__main__.build_runner") as builder, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--show-config"]), 0)
            with self.assertRaises(SystemExit) as exit_result:
                main(["--help"])
            self.assertEqual(exit_result.exception.code, 0)
            builder.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_recovery_diagnostics_keep_each_actual_generation_and_reset_next_run(self):
        internal = {"id": 1, "score": 0.9, "content": "Ô tô vượt đèn đỏ bị phạt.",
                    "metadata": {"chunk_id": "c1", "doc_id": "ND168", "dieu": 6}}
        retriever = SimpleNamespace(get_relevant_chunks=Mock(return_value=[internal]))
        generator = SimpleNamespace(
            expand_query=lambda q: q,
            generate=Mock(side_effect=[{"answer": "Thiếu căn cứ", "refused": True},
                                       {"answer": "Đã bổ sung [S1].", "refused": False},
                                       {"answer": "Câu tiếp theo [S1].", "refused": False}]))
        web = SimpleNamespace(search=Mock(return_value=[{
            "title": "Ô tô vượt đèn đỏ", "url": "https://chinhphu.vn/traffic",
            "content": "Ô tô vượt đèn đỏ bị phạt.", "score": 0.95}]))
        runner = AgentRunner(retriever, generator, web, ComparisonConfig(generator_model="test-model"))
        output = runner.run("Ô tô vượt đèn đỏ bị phạt?")
        diagnostics = runner.diagnostics
        self.assertEqual(output["status"], "answered")
        self.assertEqual(len(diagnostics["generation_attempts"]), 2)
        self.assertEqual(diagnostics["generation_attempts"][0]["status"], "refused")
        self.assertEqual(diagnostics["generation_attempts"][-1]["contexts"], output["contexts"])
        self.assertEqual(diagnostics["generation_attempts"][-1]["sources"], output["sources"])
        self.assertTrue(any(event["node"] == "web_search" for event in diagnostics["execution_trace"]))
        with JsonlResultWriter(self.path) as writer:
            writer.append(output, query_id="q_001", diagnostics=diagnostics)
        self.assertEqual(self.read()[0]["generation_attempts"], diagnostics["generation_attempts"])
        runner.run("Ô tô vượt đèn đỏ bị phạt?")
        self.assertEqual(len(runner.diagnostics["generation_attempts"]), 1)
        self.assertEqual(len(diagnostics["generation_attempts"]), 2)

    def test_baseline_trace_preserves_failed_generation_input_for_debugging(self):
        retriever = SimpleNamespace(get_relevant_chunks=Mock(return_value=[]))
        generator = SimpleNamespace(generate=Mock(side_effect=RuntimeError("LLM failed")))
        runner = BaselineRunner(retriever, generator, ComparisonConfig(generator_model="test-model"))
        output = runner.run("Câu hỏi")
        diagnostics = runner.diagnostics
        self.assertEqual(output["status"], "error")
        self.assertEqual(diagnostics["execution_trace"][-1]["node"], "generate")
        self.assertIn("LLM failed", diagnostics["generation_attempts"][0]["error"])


if __name__ == "__main__":
    unittest.main()
