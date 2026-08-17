import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


def _install_astrbot_stubs() -> None:
    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    event = types.ModuleType("astrbot.api.event")
    components = types.ModuleType("astrbot.api.message_components")
    star = types.ModuleType("astrbot.api.star")
    core = types.ModuleType("astrbot.core")
    utils = types.ModuleType("astrbot.core.utils")
    path_module = types.ModuleType("astrbot.core.utils.astrbot_path")
    quote_module = types.ModuleType("astrbot.core.utils.quoted_message_parser")

    class _Logger:
        def __getattr__(self, _name):
            return lambda *_args, **_kwargs: None

    class _Filter:
        EventMessageType = types.SimpleNamespace(ALL="all")

        @staticmethod
        def event_message_type(*_args, **_kwargs):
            return lambda function: function

    class _Star:
        def __init__(self, context):
            self.context = context

    class _Context:
        pass

    def _register(*_args, **_kwargs):
        return lambda cls: cls

    api.logger = _Logger()
    event.AstrMessageEvent = type("AstrMessageEvent", (), {})
    event.filter = _Filter()
    for name in ("At", "Forward", "Image", "Node", "Nodes", "Plain", "Reply"):
        setattr(components, name, type(name, (), {}))
    star.Context = _Context
    star.Star = _Star
    star.register = _register
    path_module.get_astrbot_data_path = lambda: "."

    async def _empty_quote(*_args, **_kwargs):
        return []

    quote_module.extract_quoted_message_images = _empty_quote
    quote_module.extract_quoted_message_text = _empty_quote

    modules = {
        "astrbot": astrbot,
        "astrbot.api": api,
        "astrbot.api.event": event,
        "astrbot.api.message_components": components,
        "astrbot.api.star": star,
        "astrbot.core": core,
        "astrbot.core.utils": utils,
        "astrbot.core.utils.astrbot_path": path_module,
        "astrbot.core.utils.quoted_message_parser": quote_module,
    }
    sys.modules.update(modules)


_install_astrbot_stubs()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402


class _FakeResponse:
    def __init__(self, status: int, body: dict | str):
        self.status = status
        self.body = body if isinstance(body, str) else json.dumps(body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def text(self):
        return self.body


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
        return self.response


def _fake_aiohttp(capture: dict, *, status: int = 200, body=None):
    if body is None:
        body = {"choices": [{"message": {"content": "true\n测试通过"}}]}
    response = _FakeResponse(status, body)
    module = types.ModuleType("aiohttp")
    module.ClientTimeout = lambda *, total: types.SimpleNamespace(total=total)
    module.ClientSession = lambda **kwargs: _FakeSession(
        capture, response, **kwargs
    )
    return module


class OpenAICompatibleClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_builds_multimodal_chat_completions_request(self):
        capture = {}
        client = main._OpenAICompatibleClient(
            api_key="secret-key",
            base_url="https://example.test/v1/",
            model="vision-model",
            timeout=12,
        )

        with patch.dict(sys.modules, {"aiohttp": _fake_aiohttp(capture)}):
            response = await client.text_chat(
                prompt="核查这张图",
                image_urls=["https://example.test/image.png"],
                system_prompt="系统提示",
            )

        self.assertEqual(response.completion_text, "true\n测试通过")
        self.assertEqual(capture["url"], "https://example.test/v1/chat/completions")
        self.assertEqual(capture["session_kwargs"]["timeout"].total, 12)
        self.assertTrue(capture["session_kwargs"]["trust_env"])
        request = capture["request_kwargs"]
        self.assertEqual(request["headers"]["Authorization"], "Bearer secret-key")
        self.assertEqual(request["json"]["model"], "vision-model")
        self.assertFalse(request["json"]["stream"])
        user_content = request["json"]["messages"][1]["content"]
        self.assertEqual(user_content[0], {"type": "text", "text": "核查这张图"})
        self.assertEqual(
            user_content[1],
            {
                "type": "image_url",
                "image_url": {"url": "https://example.test/image.png"},
            },
        )

    async def test_accepts_full_chat_completions_url_and_no_api_key(self):
        capture = {}
        client = main._OpenAICompatibleClient(
            api_key="",
            base_url="http://127.0.0.1:8000/v1/chat/completions",
            model="local-model",
            timeout=60,
        )

        with patch.dict(sys.modules, {"aiohttp": _fake_aiohttp(capture)}):
            await client.text_chat(
                prompt="hello",
                image_urls=[],
                system_prompt="",
            )

        self.assertEqual(
            capture["url"], "http://127.0.0.1:8000/v1/chat/completions"
        )
        self.assertNotIn("Authorization", capture["request_kwargs"]["headers"])
        self.assertEqual(
            capture["request_kwargs"]["json"]["messages"],
            [{"role": "user", "content": "hello"}],
        )

    async def test_surfaces_openai_error_message(self):
        capture = {}
        client = main._OpenAICompatibleClient(
            api_key="bad-key",
            base_url="https://example.test/v1",
            model="test-model",
            timeout=60,
        )
        fake_module = _fake_aiohttp(
            capture,
            status=401,
            body={"error": {"message": "invalid api key"}},
        )

        with patch.dict(sys.modules, {"aiohttp": fake_module}):
            with self.assertRaisesRegex(RuntimeError, "401: invalid api key"):
                await client.text_chat(
                    prompt="hello",
                    image_urls=[],
                    system_prompt="",
                )

    def test_validates_base_url_and_model(self):
        invalid_url = main._OpenAICompatibleClient(
            api_key="",
            base_url="localhost:8000/v1",
            model="model",
            timeout=60,
        )
        with self.assertRaisesRegex(ValueError, r"http\(s\)"):
            invalid_url.validate_configuration()

        missing_model = main._OpenAICompatibleClient(
            api_key="",
            base_url="http://localhost:8000/v1",
            model="",
            timeout=60,
        )
        with self.assertRaisesRegex(ValueError, "模型名称"):
            missing_model.validate_configuration()


class _FakeContext:
    def __init__(self):
        self.default_provider = object()

    def get_provider_by_id(self, _provider_id):
        return object()

    def get_using_provider(self):
        return self.default_provider


class PluginLLMSelectionTests(unittest.TestCase):
    def test_custom_mode_uses_only_custom_client(self):
        plugin = main.IsItTrue(
            _FakeContext(),
            {
                "llm_source": "custom_api",
                "custom_base_url": "https://example.test/v1",
                "custom_model": "custom-model",
                "provider_id": "astrbot-provider",
                "provider_fallbacks": ["fallback-provider"],
            },
        )

        self.assertEqual(plugin._llm_configuration_error(), "")
        self.assertEqual(plugin._resolve_providers(), [plugin.custom_llm])

    def test_schema_uses_official_provider_selector(self):
        schema_path = Path(__file__).resolve().parents[1] / "_conf_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        self.assertEqual(schema["provider_id"]["_special"], "select_provider")
        self.assertEqual(
            schema["provider_id"]["condition"],
            {"llm_source": "astrbot_provider"},
        )
        self.assertIn("custom_api", schema["llm_source"]["options"])


class ConnectivityTestOptionTests(unittest.IsolatedAsyncioTestCase):
    async def test_initialize_runs_connectivity_test_when_enabled(self):
        plugin = main.IsItTrue(
            _FakeContext(),
            {
                "llm_source": "custom_api",
                "custom_base_url": "https://example.test/v1",
                "custom_model": "custom-model",
                "test_llm_on_startup": True,
            },
        )
        plugin.custom_llm.test = AsyncMock()

        await plugin.initialize()

        plugin.custom_llm.test.assert_awaited_once_with()


if __name__ == "__main__":
    unittest.main()
