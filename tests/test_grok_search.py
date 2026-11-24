"""grok_search 渠道的解析与请求测试。

参考 GrokSearch-rs 的原理：向 OpenAI/xAI 兼容端点发带 tools:[{"type":"web_search"}]
的请求，再从 output[]/annotations/citations/正文 [[n]](url) 里收集来源。
"""

import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_custom_llm import _FakeContext, _install_astrbot_stubs  # noqa: E402

_install_astrbot_stubs()

import main  # noqa: E402


class _FakeResponse:
    def __init__(self, status: int, body):
        self.status = status
        self.body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def json(self):
        return self.body

    async def text(self):
        return self.body if isinstance(self.body, str) else json.dumps(self.body)


class _FakeSession:
    def __init__(self, capture: dict, response: _FakeResponse, **kwargs):
        capture["session_kwargs"] = kwargs
        self.capture = capture
        self.response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def post(self, url, **kwargs):
        self.capture["url"] = url
        self.capture["request_kwargs"] = kwargs
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _fake_aiohttp(capture: dict, *, status: int = 200, body=None):
    response = _FakeResponse(status, body if body is not None else {})
    module = types.ModuleType("aiohttp")
    module.ClientTimeout = lambda *, total: types.SimpleNamespace(total=total)
    module.ClientSession = lambda **kwargs: _FakeSession(capture, response, **kwargs)
    return module


def _make_plugin(**overrides):
    config = {
        "enable_web_search": True,
        "search_provider": "grok_search",
        "grok_search_base_url": "http://gateway.test:1493/v1",
        "grok_search_api_key": "test-key",
        "grok_search_model": "grok-4-fast",
        "grok_search_transport": "responses",
        "grok_search_timeout": 5,
        "grok_search_web_search": True,
        "grok_search_x_search": False,
        "max_search_chars": 2000,
    }
    config.update(overrides)
    return main.IsItTrue(_FakeContext(), config)


class GrokEndpointTests(unittest.TestCase):
    def test_responses_endpoint_appends_responses(self):
        plugin = _make_plugin()
        self.assertEqual(plugin._grok_search_endpoint(), "http://gateway.test:1493/v1/responses")

    def test_endpoint_normalises_trailing_slash_and_existing_path(self):
        plugin = _make_plugin(grok_search_base_url="http://gateway.test:1493/v1/chat/completions/")
        self.assertEqual(plugin._grok_search_endpoint(), "http://gateway.test:1493/v1/responses")

    def test_chat_completions_endpoint(self):
        plugin = _make_plugin(grok_search_transport="chat_completions")
        self.assertEqual(
            plugin._grok_search_endpoint(),
            "http://gateway.test:1493/v1/chat/completions",
        )

    def test_unknown_transport_falls_back_to_responses(self):
        plugin = _make_plugin(grok_search_transport="sse")
        self.assertEqual(plugin.grok_search_transport, "responses")
        self.assertTrue(plugin._grok_search_endpoint().endswith("/responses"))


class GrokPayloadTests(unittest.TestCase):
    def test_responses_payload_uses_input_and_tools(self):
        plugin = _make_plugin()
        payload = plugin._grok_search_payload("DeepSeek 峰谷定价是真的吗")

        self.assertEqual(payload["model"], "grok-4-fast")
        self.assertFalse(payload["stream"])
        self.assertNotIn("messages", payload)
        self.assertEqual([item["role"] for item in payload["input"]], ["system", "user"])
        self.assertEqual(payload["input"][-1]["content"], "DeepSeek 峰谷定价是真的吗")
        self.assertEqual(payload["tools"], [{"type": "web_search"}])

    def test_chat_completions_payload_uses_messages(self):
        plugin = _make_plugin(grok_search_transport="chat_completions", grok_search_x_search=True)
        payload = plugin._grok_search_payload("query")

        self.assertNotIn("input", payload)
        self.assertEqual(payload["messages"][-1], {"role": "user", "content": "query"})
        self.assertEqual(
            payload["tools"],
            [{"type": "web_search"}, {"type": "x_search"}],
        )

    def test_tools_can_be_disabled(self):
        plugin = _make_plugin(grok_search_web_search=False)
        payload = plugin._grok_search_payload("query")
        self.assertNotIn("tools", payload)

    def test_custom_system_prompt_is_used(self):
        plugin = _make_plugin(grok_search_system_prompt="只输出要点")
        payload = plugin._grok_search_payload("query")
        self.assertEqual(payload["input"][0], {"role": "system", "content": "只输出要点"})


class GrokParseTests(unittest.TestCase):
    def test_responses_parser_collects_all_source_shapes(self):
        plugin = _make_plugin()
        data = {
            "output_text": "网传说法属实，已有多家媒体确认。",
            "output": [
                {
                    "type": "web_search_call",
                    "action": {
                        "sources": [
                            {"title": "1", "url": "https://example.test/a"},
                            {"title": "新华网报道", "url": "https://example.test/b"},
                        ]
                    },
                },
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "补充说明。"},
                        {"type": "output_text", "text": "补充说明。"},
                    ],
                },
            ],
            "citations": ["https://example.test/c"],
        }
        data["output_text"] += "\n另见 [[2]](https://example.test/a)"

        text, sources = plugin._parse_grok_search(data)

        self.assertIn("网传说法属实", text)
        self.assertEqual(text.count("补充说明。"), 1)
        urls = [item["url"] for item in sources]
        self.assertEqual(
            urls,
            [
                "https://example.test/a",
                "https://example.test/b",
                "https://example.test/c",
            ],
        )
        # 纯数字标题（citation index）应被丢弃
        self.assertEqual(sources[0]["title"], "")
        self.assertEqual(sources[1]["title"], "新华网报道")

    def test_chat_completions_parser_handles_content_parts_and_nested_citation(self):
        plugin = _make_plugin(grok_search_transport="chat_completions")
        data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "第一段结论。"},
                            {"type": "text", "text": "第二段结论。"},
                        ],
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url_citation": {
                                    "url": "https://example.test/x",
                                    "title": "示例来源",
                                },
                            }
                        ],
                    }
                }
            ]
        }

        text, sources = plugin._parse_grok_search(data)

        self.assertEqual(text, "第一段结论。\n第二段结论。")
        self.assertEqual(sources, [{"url": "https://example.test/x", "title": "示例来源", "description": ""}])

    def test_inline_citation_regex_skips_malformed(self):
        out = []
        main._grok_extract_inline_citations(
            "正常 [[1]](https://example.test/ok) 畸形 [[x]](https://bad) [[2]](ftp://no)",
            out,
        )
        self.assertEqual([item["url"] for item in out], ["https://example.test/ok"])

    def test_empty_payload_returns_nothing(self):
        plugin = _make_plugin()
        text, sources = plugin._parse_grok_search({})
        self.assertEqual(text, "")
        self.assertEqual(sources, [])

    def test_format_includes_answer_and_numbered_sources(self):
        plugin = _make_plugin()
        formatted = plugin._format_grok_search_results(
            "结论正文。",
            [{"url": "https://example.test/a", "title": "来源A", "description": "摘要"}],
        )
        self.assertIn("## Grok 联网搜索结果", formatted)
        self.assertIn("结论正文。", formatted)
        self.assertIn("1. 来源A — https://example.test/a", formatted)

    def test_format_without_content_returns_empty(self):
        plugin = _make_plugin()
        self.assertEqual(plugin._format_grok_search_results("", []), "")


class GrokSearchRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_and_formatting(self):
        capture = {}
        body = {
            "output_text": "确实属实。",
            "output": [
                {
                    "type": "web_search_call",
                    "action": {"sources": [{"url": "https://example.test/a", "title": "来源A"}]},
                }
            ],
        }
        plugin = _make_plugin()

        with patch.dict(sys.modules, {"aiohttp": _fake_aiohttp(capture, body=body)}):
            result = await plugin._grok_search("DeepSeek 峰谷定价")

        self.assertEqual(capture["url"], "http://gateway.test:1493/v1/responses")
        self.assertEqual(capture["request_kwargs"]["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(capture["request_kwargs"]["json"]["model"], "grok-4-fast")
        # 配置 5 秒会被下限夹到 10 秒
        self.assertEqual(capture["session_kwargs"]["timeout"].total, 10)
        self.assertIn("确实属实。", result)
        self.assertIn("1. 来源A — https://example.test/a", result)

    async def test_missing_api_key_skips_request(self):
        capture = {}
        plugin = _make_plugin(grok_search_api_key="")

        with patch.dict(sys.modules, {"aiohttp": _fake_aiohttp(capture)}):
            result = await plugin._grok_search("query")

        self.assertEqual(result, "")
        self.assertEqual(capture, {})

    async def test_http_error_returns_empty(self):
        capture = {}
        plugin = _make_plugin()

        with patch.dict(sys.modules, {"aiohttp": _fake_aiohttp(capture, status=500, body="boom")}):
            result = await plugin._grok_search("query")

        self.assertEqual(result, "")

    async def test_transport_exception_returns_empty(self):
        capture = {}
        plugin = _make_plugin()

        with patch.dict(
            sys.modules,
            {"aiohttp": _fake_aiohttp(capture, body=RuntimeError("connect refused"))},
        ):
            result = await plugin._grok_search("query")

        self.assertEqual(result, "")

    async def test_result_is_truncated(self):
        capture = {}
        body = {"output_text": "长" * 500}
        plugin = _make_plugin(max_search_chars=200)

        with patch.dict(sys.modules, {"aiohttp": _fake_aiohttp(capture, body=body)}):
            result = await plugin._grok_search("query")

        self.assertLessEqual(len(result), 200)
        self.assertTrue(result.endswith("…(已截断)"))

    async def test_chat_completions_transport_end_to_end(self):
        capture = {}
        body = {"choices": [{"message": {"content": "chat 结论。"}}]}
        plugin = _make_plugin(grok_search_transport="chat_completions")

        with patch.dict(sys.modules, {"aiohttp": _fake_aiohttp(capture, body=body)}):
            result = await plugin._grok_search("query")

        self.assertEqual(capture["url"], "http://gateway.test:1493/v1/chat/completions")
        self.assertIn("messages", capture["request_kwargs"]["json"])
        self.assertIn("chat 结论。", result)


class GrokSchemaTests(unittest.TestCase):
    def test_schema_exposes_grok_search(self):
        schema = json.loads(
            (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text(encoding="utf-8")
        )

        self.assertIn("grok_search", schema["search_provider"]["options"])
        self.assertEqual(schema["grok_search_transport"]["options"], ["responses", "chat_completions"])
        for key in (
            "grok_search_base_url",
            "grok_search_api_key",
            "grok_search_model",
            "grok_search_web_search",
            "grok_search_x_search",
            "grok_search_timeout",
            "grok_search_system_prompt",
        ):
            self.assertIn(key, schema)


if __name__ == "__main__":
    unittest.main()