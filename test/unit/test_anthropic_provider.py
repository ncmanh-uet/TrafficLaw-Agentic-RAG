"""Offline Claude transport and generator configuration checks."""

import json
import unittest
from unittest.mock import patch

import httpx2 as httpx
from langchain_anthropic import ChatAnthropic

from Sources.rag_core.generator import LegalAnswerGenerator
from test.evaluation_agent.config import ComparisonConfig
from test.test_generator_response import internal_chunk


class AnthropicProviderTests(unittest.TestCase):
    def test_messages_transport_and_citations(self):
        requests = []

        def handler(request):
            self.assertEqual(request.url.path, "/v1/messages")
            self.assertEqual(request.headers["x-api-key"], "offline-test")
            body = json.loads(request.content)
            requests.append(body)
            self.assertEqual(body["max_tokens"], 2048)
            self.assertNotIn("temperature", body)
            self.assertNotIn("reasoning_effort", body)
            self.assertNotIn("automatic_function_calling", body)
            self.assertTrue(body["system"])
            return httpx.Response(200, json={
                "id": "msg_test", "type": "message", "role": "assistant",
                "model": "claude-test", "content": [{"type": "text", "text": "Answer [S1]."}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 5},
            })

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            def chat(**kwargs):
                return ChatAnthropic(**kwargs, max_retries=0)

            with patch("langchain_anthropic.chat_models._get_default_httpx_client", return_value=client), patch(
                "langchain_anthropic.ChatAnthropic", side_effect=chat
            ), patch.dict(
                "os.environ", {"ANTHROPIC_API_KEY": "offline-test"}
            ):
                generator = LegalAnswerGenerator(
                    provider="anthropic", model="claude-test", max_tokens=2048,
                    reasoning_effort="low",
                )
                result = generator.generate("Question?", [internal_chunk()])
        self.assertTrue(result["citation_valid"])
        self.assertEqual(result["model"], "anthropic/claude-test")
        self.assertEqual(len(requests), 1)

    def test_missing_credentials_and_model_fail_early(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "ANTHROPIC_API_KEY"):
                LegalAnswerGenerator(provider="anthropic", model="claude-test")
            with self.assertRaisesRegex(ValueError, "explicit model"):
                LegalAnswerGenerator(provider="anthropic", api_key="offline-test")

    def test_comparison_records_claude_sampling_defaults(self):
        config = ComparisonConfig(generator_provider="anthropic", generator_model="claude-test",
                                  reasoning_effort="low")
        self.assertIsNone(config.temperature)
        self.assertIsNone(config.reasoning_effort)
