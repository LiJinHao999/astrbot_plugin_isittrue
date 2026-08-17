import base64
import json
import sys
import tempfile
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

    async def test_normalizes_local_file_uri_and_base64_images(self):
        capture = {}
        png_base64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNg"
            "YAAAAAMAASsJTYQAAAAASUVORK5CYII="
        )
        client = main._OpenAICompatibleClient(
            api_key="",
            base_url="https://example.test/v1",
            model="vision-model",
            timeout=60,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "test image.png"
            image_path.write_bytes(base64.b64decode(png_base64))
            image_refs = [
                str(image_path),
                image_path.as_uri(),
                f"base64://{png_base64}",
            ]
            with patch.dict(sys.modules, {"aiohttp": _fake_aiohttp(capture)}):
                await client.text_chat(
                    prompt="核查图片",
                    image_urls=image_refs,
                    system_prompt="",
                )

        content = capture["request_kwargs"]["json"]["messages"][0]["content"]
        normalized_refs = [item["image_url"]["url"] for item in content[1:]]
        self.assertEqual(len(normalized_refs), 3)
        for normalized_ref in normalized_refs:
            self.assertTrue(normalized_ref.startswith("data:image/png;base64,"))
            self.assertEqual(normalized_ref.split(",", 1)[1], png_base64)

    async def test_rejects_unresolvable_image_before_http_request(self):
        capture = {}
        client = main._OpenAICompatibleClient(
            api_key="",
            base_url="https://example.test/v1",
            model="vision-model",
            timeout=60,
        )

        with patch.dict(sys.modules, {"aiohttp": _fake_aiohttp(capture)}):
            with self.assertRaisesRegex(ValueError, "图片引用不是有效"):
                await client.text_chat(
                    prompt="核查图片",
                    image_urls=["not-a-url-or-local-file"],
                    system_prompt="",
                )

        self.assertNotIn("url", capture)

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

    def test_formats_result_with_actual_model_name(self):
        plugin = main.IsItTrue(
            _FakeContext(),
            {
                "llm_source": "custom_api",
                "custom_base_url": "https://example.test/v1",
                "custom_model": "grok-4.5",
            },
        )

        model_name = plugin._provider_model_name(plugin.custom_llm)
        result = plugin._format_result_with_source("true\n核查内容属实。", model_name)

        self.assertEqual(result, "✅ 真的喵\n核查内容属实。\n来源：grok-4.5")

    def test_prefers_astrbot_provider_model_name_over_provider_id(self):
        class _Provider:
            provider_config = {"id": "provider-id", "model": "config-model"}

            @staticmethod
            def get_model():
                return "runtime-model"

        plugin = main.IsItTrue(_FakeContext(), {})

        self.assertEqual(plugin._provider_model_name(_Provider()), "runtime-model")

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


class ModelFallbackSourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_reports_model_that_succeeds_after_fallback(self):
        class _Provider:
            def __init__(self, model_name, error=None):
                self.model_name = model_name
                self.error = error
                self.provider_config = {
                    "id": f"provider-{model_name}",
                    "model": model_name,
                }

            def get_model(self):
                return self.model_name

            async def text_chat(self, **_kwargs):
                if self.error:
                    raise self.error
                return main._LLMResponse(completion_text="true\n回退成功。")

        plugin = main.IsItTrue(_FakeContext(), {})
        providers = [
            _Provider("primary-model", RuntimeError("primary unavailable")),
            _Provider("fallback-model"),
        ]
        plugin._resolve_providers = lambda: providers

        response, model_name, degraded = await plugin._chat_with_fallback(
            prompt_with_images="prompt",
            prompt_without_images="prompt",
            image_urls=[],
            system_prompt="system",
            stage="终判",
        )

        self.assertEqual(response.completion_text, "true\n回退成功。")
        self.assertEqual(model_name, "fallback-model")
        self.assertFalse(degraded)


if __name__ == "__main__":
    unittest.main()
