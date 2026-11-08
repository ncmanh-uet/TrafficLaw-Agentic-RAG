"""Offline OpenAI wire, CLI and checkpoint checks; no live API calls."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from openai import AsyncOpenAI, AuthenticationError, RateLimitError

from Sources.rag_core.generator import LegalAnswerGenerator
from test.evaluation_agent.__main__ import main as query_main
from test.evaluation_agent.citation import AnswerAnalysis
from test.evaluation_agent.judge import JudgeConfig, OpenAIJudge, error_detail
from test.evaluation_agent.score import run
from test.unit.test_agent_scoring import analysis, gold, output, verdict
from test.test_generator_response import internal_chunk


def response_body(text):
    return {"id": "resp_offline", "object": "response", "created_at": 1,
            "model": "gpt-6-luna", "status": "completed", "error": None,
            "output": [{"id": "msg_offline", "type": "message", "role": "assistant",
                        "status": "completed", "content": [
                            {"type": "output_text", "text": text, "annotations": []}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                      "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 0}}}


class GeneratorProviderTests(unittest.TestCase):
    def test_luna_generator_uses_responses_without_temperature(self):
        from langchain_openai import ChatOpenAI

        requests = []
        answer = "Phạt tiền [S1] [Điều 6, Khoản 9, Điểm b — 168/2024/NĐ-CP]."

        def handler(request):
            self.assertEqual(request.url.path, "/v1/responses")
            body = json.loads(request.content)
            requests.append(body)
            self.assertEqual(body["model"], "gpt-6-luna")
            self.assertEqual(body["reasoning"]["effort"], "low")
            self.assertEqual(body["max_output_tokens"], 4096)
            self.assertNotIn("temperature", body)
            return httpx.Response(200, json=response_body(answer))

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            def chat(**kwargs):
                return ChatOpenAI(**kwargs, http_client=client, max_retries=0)

            with patch("langchain_openai.ChatOpenAI", side_effect=chat):
                generator = LegalAnswerGenerator(provider="openai", model="gpt-6-luna",
                                                 api_key="offline-test", reasoning_effort="low", max_tokens=4096)
                result = generator.generate("Mức phạt?", [internal_chunk()])
        self.assertEqual(result["answer"], answer)
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(len(requests), 1)

    def test_cli_overrides_environment_without_exposing_keys(self):
        stream = StringIO()
        with patch("dotenv.load_dotenv"), patch.dict("os.environ", {
                "GENERATOR_PROVIDER": "google", "GENERATOR_MODEL": "gemini-old",
                "OPENAI_API_KEY": "offline-secret"}), redirect_stdout(stream):
            self.assertEqual(query_main(["--show-config", "--provider", "openai", "--model", "gpt-6-luna",
                                         "--reasoning-effort", "low", "--max-tokens", "4096"]), 0)
        config = json.loads(stream.getvalue())
        self.assertEqual(config["generator_provider"], "openai")
        self.assertEqual(config["generator_model"], "gpt-6-luna")
        self.assertEqual(config["reasoning_effort"], "low")
        self.assertEqual(config["max_tokens"], 4096)
        self.assertIsNone(config["temperature"])
        self.assertNotIn("offline-secret", stream.getvalue())


class OpenAIJudgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_sdk_sends_strict_schema_and_parses_output(self):
        requests = []

        def handler(request):
            self.assertEqual(request.url.path, "/v1/responses")
            body = json.loads(request.content)
            requests.append(body)
            self.assertEqual(body["model"], "gpt-6-luna")
            self.assertEqual(body["reasoning"], {"effort": "medium"})
            self.assertFalse(body["store"])
            self.assertNotIn("temperature", body)
            self.assertNotIn("tools", body)
            self.assertTrue(body["text"]["format"]["strict"])
            self.assertFalse(body["text"]["format"]["schema"]["additionalProperties"])
            return httpx.Response(200, json=response_body('{"abstaining":false,"claims":[]}'))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = AsyncOpenAI(api_key="offline-test", http_client=http_client, max_retries=0)
            with patch.dict("os.environ", {"OPENAI_API_KEY": "offline-test"}), \
                 patch("openai.AsyncOpenAI", return_value=client) as factory:
                judge = OpenAIJudge(JudgeConfig(provider="openai", model="gpt-6-luna"))
                result = await judge.generate("Synthetic greeting", AnswerAnalysis)
                self.assertEqual(result, AnswerAnalysis(abstaining=False, claims=[]))
                self.assertEqual(factory.call_args.kwargs["max_retries"], 0)
                await judge.close()
        self.assertEqual(len(requests), 1)

    async def test_authentication_and_billing_quota_do_not_retry(self):
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        cases = [
            AuthenticationError("Invalid key", response=httpx.Response(401, request=request), body=None),
            RateLimitError("Quota exceeded", response=httpx.Response(429, request=request),
                           body={"code": "insufficient_quota"}),
        ]
        for error in cases:
            client = SimpleNamespace(responses=SimpleNamespace(parse=AsyncMock(side_effect=error)))
            with self.subTest(error=type(error).__name__), \
                 patch.dict("os.environ", {"OPENAI_API_KEY": "offline-test"}), \
                 patch("openai.AsyncOpenAI", return_value=client), \
                 patch("test.evaluation_agent.judge.asyncio.sleep", new_callable=AsyncMock) as sleep:
                judge = OpenAIJudge(JudgeConfig(provider="openai", model="gpt-6-luna"))
                with self.assertRaises(type(error)):
                    await judge.generate("Synthetic", AnswerAnalysis)
                client.responses.parse.assert_awaited_once()
                sleep.assert_not_awaited()
                self.assertEqual(error_detail(error)["http_code"], error.status_code)

    async def test_incomplete_output_is_an_error(self):
        client = SimpleNamespace(responses=SimpleNamespace(parse=AsyncMock(return_value=SimpleNamespace(
            status="incomplete", output_parsed=None))))
        with patch.dict("os.environ", {"OPENAI_API_KEY": "offline-test"}), \
             patch("openai.AsyncOpenAI", return_value=client):
            judge = OpenAIJudge(JudgeConfig(provider="openai", model="gpt-6-luna", attempts=1))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                await judge.generate("Synthetic", AnswerAnalysis)

    async def test_openai_checkpoint_preserves_results_and_rejects_provider_change(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, inputs, scores = (root / name for name in ("dataset.jsonl", "outputs.jsonl", "scores.jsonl"))
            dataset.write_text(json.dumps(gold()) + "\n", encoding="utf-8")
            inputs.write_text(json.dumps(output()) + "\n", encoding="utf-8")
            args = SimpleNamespace(dataset=dataset, input=inputs, output=scores, split="pilot", allow_partial=True,
                                   check_only=False, provider="openai", model="gpt-6-luna", resume=False,
                                   include_details=True, reasoning_effort="medium", max_output_tokens=16384)
            judge = SimpleNamespace(generate=AsyncMock(side_effect=[analysis("Phạt"), verdict("S1")]), close=AsyncMock())
            metrics = {name: SimpleNamespace(ascore=AsyncMock(return_value=SimpleNamespace(value=1.0)))
                       for name in ("factual_correctness", "faithfulness")}
            with patch("test.evaluation_agent.score.OpenAIJudge", return_value=judge), \
                 patch("test.evaluation_agent.score.GeminiJudge") as google, \
                 patch("test.evaluation_agent.score.make_ragas_metrics", return_value=metrics), \
                 patch("test.evaluation_agent.score.version", return_value="test"):
                self.assertEqual(await run(args), 0)
                original = scores.read_bytes()
                meta = json.loads(scores.with_suffix(".meta.json").read_text(encoding="utf-8"))
                self.assertEqual(meta["judge"]["provider"], "openai")
                self.assertEqual(meta["judge"]["reasoning_effort"], "medium")
                self.assertEqual(meta["judge"]["max_output_tokens"], 16384)
                self.assertIn("openai_version", meta)
                args.resume = True
                self.assertEqual(await run(args), 0)
                self.assertEqual(judge.generate.await_count, 2)
                for field, value in (("provider", "google"), ("reasoning_effort", "high")):
                    old = getattr(args, field)
                    setattr(args, field, value)
                    with self.assertRaisesRegex(ValueError, "Resume refused"):
                        await run(args)
                    setattr(args, field, old)
                self.assertEqual(scores.read_bytes(), original)
                google.assert_not_called()


if __name__ == "__main__":
    unittest.main()
