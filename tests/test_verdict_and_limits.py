"""判定词表统一、并发/超时配置、请求级缓存与 Grok 重试/降级的测试。"""

import asyncio
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


def _make_plugin(**overrides):
    config = {
        "enable_web_search": True,
        "search_provider": "grok_search",
        "grok_search_base_url": "http://gateway.test:1493/v1",
        "grok_search_api_key": "test-key",
        "grok_search_model": "grok-4-fast",
        "grok_search_transport": "responses",
        "grok_search_timeout": 120,
        "max_search_chars": 2000,
    }
    config.update(overrides)
    return main.IsItTrue(_FakeContext(), config)


class VerdictTokenTests(unittest.TestCase):
    """行首判定词解析：三处共用同一份词表。"""

    def test_decorated_labels_resolve(self):
        cases = {
            "✅ 真的喵": "true",
            "❌ 假的喵": "false",
            "⚠️ 布吉岛": "unknown",
            "**true**": "true",
            "结论：属实": "true",
            "1. false": "false",
# 「✅ 1. 真的」这类 emoji + 列表序号前缀也要能认出判定词
            "✅ 1. 真的": "true",
            "unknown（存疑）": "unknown",
            "这是假的": None,
        }
        plugin = _make_plugin()
        for token, expected in cases.items():
# 词表命中就断言判定值；不在表里的一律期望 None，别留死分支
            self.assertEqual(plugin._parse_verdict_token(token), expected, token)

    def test_single_char_alias_does_not_eat_words(self):
        plugin = _make_plugin()
        # 「真实的说法」不是判定行，单字「真」不能吞掉它
        self.assertIsNone(plugin._parse_verdict_token("真实的说法"))
        self.assertIsNone(plugin._parse_verdict_token("真的假的"))
        self.assertIsNone(plugin._parse_verdict_token(""))

    def test_strip_verdict_word_keeps_body(self):
        plugin = _make_plugin()
        self.assertEqual(plugin._strip_verdict_word("✅ 真的喵"), "")
        self.assertEqual(plugin._strip_verdict_word("❌ 假的喵"), "")
        self.assertEqual(plugin._strip_verdict_word("结论：不实。"), "")
        self.assertEqual(plugin._strip_verdict_word("真的，这是编造的"), "这是编造的")
        self.assertEqual(plugin._strip_verdict_word("这是编造的"), "这是编造的")


class DetectVerdictTests(unittest.TestCase):
    """兜底 signals 用同一份词表派生。"""

    def test_first_line_label_drives_verdict(self):
        plugin = _make_plugin()
        self.assertEqual(
            plugin._detect_verdict("❌ 假的喵\n这是编造的，官方从未发布该通知。"),
            "false",
        )
        self.assertEqual(
            plugin._detect_verdict("✅ 真的喵\n据官方公告，该消息属实。"),
            "true",
        )
        self.assertEqual(plugin._detect_verdict("⚠️ 布吉岛\n无法核实"), "unknown")

    def test_body_signals_fall_back(self):
        plugin = _make_plugin()
        self.assertEqual(plugin._detect_verdict("嗯\n官方公告确认该消息属实。"), "true")
        self.assertEqual(plugin._detect_verdict("嗯\n经核查为谣言。"), "false")
        self.assertIsNone(plugin._detect_verdict("嗯，这说法缺乏证据。"))
# 否定词守卫：「并不属实」判 false、「并非谣言」判 true，正反同现判 unknown
    def test_negated_signals_are_not_read_as_positive(self):
        plugin = _make_plugin()
        # 「并不属实」说的是假，「并非谣言」说的是真——否定词不能被丢掉
        self.assertEqual(
            plugin._detect_verdict("嗯\n官方已澄清，原文并不属实。"), "false"
        )
        self.assertEqual(
            plugin._detect_verdict("嗯\n这条消息并非谣言，是官方公告。"), "true"
        )
        # 正反两方都有说法的句子不硬挑一个，交给 unknown
        self.assertIsNone(
            plugin._detect_verdict(
                "嗯\n目前没有任何资料证明它为真，也没有证据说不实。"
            )
        )
# 「不是真的」「假消息」是口语里的否定，不能被「是真的」抢成 true
        self.assertEqual(plugin._detect_verdict("嗯\n这不是真的，官方否认了。"), "false")
        self.assertEqual(plugin._detect_verdict("嗯\n这是假消息，别信。"), "false")
        # 「真的假的？」是在问，不是断言
        self.assertIsNone(plugin._detect_verdict("嗯\n真的假的？我也想知道。"))
# 「据说 / 没有证据 / 无法证明 / …否认」这些分句讲的是别人怎么说、有没有证据，
        # 里面的判定词不能当结论——否则「据说这是真的」会被渲染成 ✅ 真的喵
        for text in (
            "他曾说过这是真的，但官方否认",
            "他说这是真的，但官方通报否认",
            "据说这是真的",
            "没有证据表明它是真的",
            "目前无法证明这是真的",
            "官方没有确认这是真的",
            "没有证据表明这是谣言",
            "目前没有任何资料证明它为真",
        ):
            self.assertIsNone(plugin._detect_verdict(text), text)
# 转折词（但/然而/不过）也算分句边界：前半句讲「他说是」，后半句才是结论
        self.assertEqual(
            plugin._detect_verdict("他说这是真的但该视频是假消息。"), "false"
        )
        self.assertEqual(
            plugin._detect_verdict("没有证据表明这是谣言但官方已通报该视频为假消息。"),
            "false",
        )
# 「网传…不实」是官方通报最常见句式，别把网传/据传这类词当屏蔽标记：
# 否则「经查，网传内容不实」会被抹成 None，丢掉本该有的 false
        self.assertEqual(plugin._detect_verdict("经查，网传内容不实。"), "false")
        self.assertEqual(plugin._detect_verdict("网传信息不实。"), "false")
        self.assertEqual(plugin._detect_verdict("官方已澄清，原文并不属实。"), "false")
        self.assertEqual(plugin._detect_verdict("经核实，该说法不实，系拼接视频。"), "false")
        # 「网传」只是转述对象，不能连结论一起抹掉：正向的也该认出来
        self.assertEqual(plugin._detect_verdict("网传截图经查证属实。"), "true")


class LimitConfigTests(unittest.TestCase):
    """并发/总超时/缓存/Grok 重试配置的默认值与夹取。"""

    def test_defaults(self):
        plugin = _make_plugin()
        self.assertEqual(plugin.max_concurrent, 2)
        self.assertEqual(plugin.total_timeout, 900)
        self.assertEqual(plugin.check_cache_ttl, 300)
        self.assertEqual(plugin.grok_search_retries, 2)
        self.assertTrue(plugin.grok_search_fallback)
        self.assertEqual(plugin.grok_search_timeout, 120)

    def test_clamps(self):
        plugin = _make_plugin(
            max_concurrent=99,
            total_timeout=5,
            check_cache_ttl=-1,
            grok_search_retries=9,
            grok_search_timeout=999,
            grok_search_fallback=False,
        )
        self.assertEqual(plugin.max_concurrent, 10)
        self.assertEqual(plugin.total_timeout, 30)
        self.assertEqual(plugin.check_cache_ttl, 0)
        self.assertEqual(plugin.grok_search_retries, 5)
        self.assertEqual(plugin.grok_search_timeout, 120)
        self.assertFalse(plugin.grok_search_fallback)
# 显式 None 要退回文档默认值；字符串 "false" 不能被当成开启
    def test_none_and_string_values_keep_documented_defaults(self):
        plugin = _make_plugin(
            max_concurrent=None,
            total_timeout=None,
            check_cache_ttl=None,
            grok_search_retries=None,
            grok_search_fallback=None,
        )
        self.assertEqual(plugin.max_concurrent, 2)
        self.assertEqual(plugin.total_timeout, 900)
        self.assertEqual(plugin.check_cache_ttl, 300)
        self.assertEqual(plugin.grok_search_retries, 2)
        self.assertTrue(plugin.grok_search_fallback)
        # 字符串 "false" 不能被当成开启（bool("false") 是 True）
        self.assertFalse(_make_plugin(grok_search_fallback="false").grok_search_fallback)
        self.assertTrue(_make_plugin(grok_search_fallback="true").grok_search_fallback)
# grok 单次超时也要走同一套夹取：非数字不能让插件在加载时就抛 ValueError
    def test_grok_search_timeout_is_clamped_not_raised(self):
        self.assertEqual(_make_plugin(grok_search_timeout="abc").grok_search_timeout, 120)
        self.assertEqual(_make_plugin(grok_search_timeout=None).grok_search_timeout, 120)
        self.assertEqual(_make_plugin(grok_search_timeout=0).grok_search_timeout, 10)
        self.assertEqual(_make_plugin(grok_search_timeout=-5).grok_search_timeout, 10)
        self.assertEqual(_make_plugin(grok_search_timeout="60").grok_search_timeout, 60)
        self.assertEqual(_make_plugin(grok_search_timeout=999).grok_search_timeout, 120)


class _FakeEvent:
    def __init__(self, messages=None):
        self._messages = list(messages or [])

    def get_messages(self):
        return self._messages


class RequestCacheTests(unittest.TestCase):
    def test_key_prefers_quoted_message_id(self):
        plugin = _make_plugin()
        event = _FakeEvent([main.Reply(id="12345")])
        bundle = {"text": "某说法", "supplement": "", "images": []}
# 键 = 被引消息 ID + 待核内容指纹：同一被引消息下的不同说法不能共用结论
        key = plugin._cache_key(event, bundle)
        self.assertTrue(key.startswith("quoted:12345:"))
        self.assertEqual(key, plugin._cache_key(event, dict(bundle)))
        self.assertNotEqual(
            key, plugin._cache_key(event, dict(bundle, text="另一个说法"))
        )
        # 空白补充问题与没有补充视为同一请求
        self.assertEqual(key, plugin._cache_key(event, dict(bundle, supplement="  ")))
        # 什么都没问的时候才退回纯消息 ID
        self.assertEqual(
            plugin._cache_key(event, {"text": "", "supplement": "", "images": []}),
            "quoted:12345",
        )

    def test_key_falls_back_to_content_fingerprint(self):
        plugin = _make_plugin()
        event = _FakeEvent()
        bundle = {"text": "某说法", "supplement": "补充", "images": ["http://a/1.png"]}
        first = plugin._cache_key(event, bundle)
        self.assertTrue(first.startswith("content:"))
        self.assertEqual(first, plugin._cache_key(event, dict(bundle)))
        changed = dict(bundle, text="另一个说法")
        self.assertNotEqual(first, plugin._cache_key(event, changed))

    def test_store_and_expire(self):
        plugin = _make_plugin()
        plugin._store_cache("quoted:1", "旧结论")
        self.assertEqual(plugin._cached_content("quoted:1"), "旧结论")

        plugin._result_cache["quoted:2"] = (0.0, "过时结论")
        self.assertIsNone(plugin._cached_content("quoted:2"))
        self.assertNotIn("quoted:2", plugin._result_cache)

# 正文里带 \x1f 也不能和「正文 + 补充」拼出同一个键
    def test_fingerprint_is_separator_safe(self):
        plugin = _make_plugin()
        event = _FakeEvent()
        joined = plugin._cache_key(
            event, {"text": "甲\x1f乙", "supplement": "", "images": []}
        )
        split = plugin._cache_key(
            event, {"text": "甲", "supplement": "乙", "images": []}
        )
        self.assertNotEqual(joined, split)
    def test_disabled_cache_returns_empty_key(self):
        plugin = _make_plugin(check_cache_ttl=0)
        event = _FakeEvent([main.Reply(id="12345")])
        bundle = {"text": "某说法", "supplement": "", "images": []}
        self.assertEqual(plugin._cache_key(event, bundle), "")
        self.assertIsNone(plugin._cached_content(""))


class _FakeResponse:
    def __init__(self, status, body):
        self.status = status
        self.body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

# 200 但响应体不是 JSON 时应直接放弃；标记值 "not-json" 用来模拟
    async def json(self):
        # 模拟「HTTP 200 但响应体不是 JSON」：正确做法是直接放弃，不要白重试
        if self.body == "not-json":
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self.body

    async def text(self):
        return self.body if isinstance(self.body, str) else json.dumps(self.body)


class _FakeSession:
    def __init__(self, state):
        self.state = state

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def post(self, url, **_kwargs):
        index = self.state["attempts"]
        self.state["attempts"] += 1
        self.state["urls"].append(url)
        item = self.state["responses"][min(index, len(self.state["responses"]) - 1)]
        if isinstance(item, Exception):
            raise item
        return _FakeResponse(*item)


def _fake_aiohttp(responses):
    state = {"attempts": 0, "urls": [], "responses": list(responses)}
    module = types.ModuleType("aiohttp")
    module.ClientTimeout = lambda *, total: types.SimpleNamespace(total=total)
# 真实代码按 aiohttp.ClientError 判断是否值得重试，假模块必须提供同类
    module.ClientError = type("ClientError", (Exception,), {})
    module.ClientSession = lambda **_kwargs: _FakeSession(state)
    return module, state


async def _noop_sleep(*_args, **_kwargs):
    return None


class GrokRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_retries_then_succeeds(self):
        plugin = _make_plugin(grok_search_retries=1)
        module, state = _fake_aiohttp([(500, "boom"), (200, {"output_text": "检索到的要点。"})])

        with patch.dict(sys.modules, {"aiohttp": module}), patch("asyncio.sleep", new=_noop_sleep):
            result = await plugin._grok_search("query")

        self.assertEqual(state["attempts"], 2)
        self.assertIn("检索到的要点。", result)

    async def test_exhausted_retries_return_empty(self):
        plugin = _make_plugin(grok_search_retries=1)
        module, state = _fake_aiohttp([(500, "boom")])

        with patch.dict(sys.modules, {"aiohttp": module}), patch("asyncio.sleep", new=_noop_sleep):
            result = await plugin._grok_search("query")

        self.assertEqual(state["attempts"], 2)
        self.assertEqual(result, "")

    async def test_client_error_is_not_retried(self):
        plugin = _make_plugin(grok_search_retries=5)
        module, state = _fake_aiohttp([(400, "bad request")])

        with patch.dict(sys.modules, {"aiohttp": module}), patch("asyncio.sleep", new=_noop_sleep):
            result = await plugin._grok_search("query")

        self.assertEqual(state["attempts"], 1)
        self.assertEqual(result, "")


class GrokFallbackTests(unittest.IsolatedAsyncioTestCase):
    def _install_fake_search(self, plugin, calls: list[str], hit: str = "tavily"):
        async def _fake_search_with(provider, _query, _event=None, _claim=""):
            calls.append(provider)
            return f"{provider} 搜索结果" if provider == hit else ""

        plugin._search_with = _fake_search_with

    async def test_explicit_grok_falls_back_to_other_channels(self):
        plugin = _make_plugin(search_provider="grok_search", grok_search_fallback=True)
        calls: list[str] = []
        self._install_fake_search(plugin, calls)

        result = await plugin._web_search("query")

        self.assertEqual(calls[0], "grok_search")
        self.assertNotIn("grok_search", calls[1:])
        self.assertEqual(result, "tavily 搜索结果")

    async def test_fallback_can_be_disabled(self):
        plugin = _make_plugin(search_provider="grok_search", grok_search_fallback=False)
        calls: list[str] = []
        self._install_fake_search(plugin, calls)

        result = await plugin._web_search("query")

        self.assertEqual(calls, ["grok_search"])
        self.assertEqual(result, "")

    async def test_fallback_skips_failed_channel(self):
        plugin = _make_plugin()
        calls: list[str] = []
        self._install_fake_search(plugin, calls, hit="bocha")

        result = await plugin._search_fallback("query", skip="grok_search")

        self.assertNotIn("grok_search", calls)
        self.assertEqual(result, "bocha 搜索结果")


class SchemaTests(unittest.TestCase):
    def test_new_options_present(self):
        schema = json.loads(
            (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text(encoding="utf-8")
        )
        for key in (
            "max_concurrent",
            "total_timeout",
            "check_cache_ttl",
            "grok_search_retries",
            "grok_search_fallback",
        ):
            self.assertIn(key, schema, key)
        self.assertEqual(schema["check_cache_ttl"]["default"], 300)
        self.assertEqual(schema["grok_search_timeout"]["default"], 120)


class ReviewRegressionTests(unittest.TestCase):
    """独立审查指出的真实缺陷：疑问句不该算判定、401 不该重试、追问算不同请求。"""

    def test_question_sentence_is_not_verdict(self):
        plugin = _make_plugin()
        for text in (
            "真的吗",
            "真的吗？",
            "是真的吗",
            "这说法属实吗",
            "真的嘛",
            "真的嘛？",
            "属实嘛？",
        ):
            self.assertIsNone(plugin._parse_verdict_token(text), text)
        self.assertIsNone(plugin._detect_verdict("这是真的吗"))
        self.assertIsNone(plugin._detect_verdict("官方说这是真的吗？"))
        # 断言句仍然要能认出来，别为了躲疑问句把正常判定也丢了
        self.assertEqual(plugin._parse_verdict_token("真的喵"), "true")
        self.assertEqual(plugin._detect_verdict("官方公告确认该消息属实。"), "true")
        self.assertEqual(plugin._detect_verdict("这条已经确认是真的。"), "true")

    def test_quoted_cache_key_includes_follow_up(self):
        plugin = _make_plugin()
        event = _FakeEvent([main.Reply(id="555")])
        plain = {"text": "真的吗", "supplement": "", "images": []}
        asked = {"text": "真的吗", "supplement": "这条作者后来澄清了吗？", "images": []}
# 键里带内容指纹：同一被引消息下，说法变了就是另一个请求
        plain_key = plugin._cache_key(event, plain)
        self.assertTrue(plain_key.startswith("quoted:555:"))
        follow_key = plugin._cache_key(event, asked)
        self.assertNotEqual(follow_key, plain_key)
        self.assertTrue(follow_key.startswith("quoted:555:"))
        self.assertEqual(follow_key, plugin._cache_key(event, dict(asked)))

    def test_decorated_prefix_is_stripped_case_insensitively(self):
        plugin = _make_plugin()
        self.assertEqual(plugin._strip_verdict_word("Verdict: unknown"), "")
        self.assertEqual(plugin._strip_verdict_word(" ANSWER: UNKNOWN "), "")
        self.assertEqual(plugin._strip_verdict_word("unknown（资料不足）"), "资料不足")

# 判定词和正文写在同一行（模型常这么写）：「真的喵，官方已公告」要能剥出正文
    def test_verdict_word_on_same_line_is_stripped(self):
        plugin = _make_plugin()
        line = "✅ 真的喵，官方已公告。"
        self.assertEqual(plugin._parse_verdict_token(line), "true")
        self.assertEqual(plugin._strip_verdict_word(line), "官方已公告。")
        self.assertEqual(plugin._detect_verdict(line), "true")
        # 实词紧跟判定词时仍不能吞（「真实的说法」不是断言）
        self.assertIsNone(plugin._parse_verdict_token("✅ 真实的说法"))

class GrokStatusRetryTests(unittest.IsolatedAsyncioTestCase):
    """401/403 是 Key 本身有问题，重试没有意义；429 属于瞬时故障，要重试。"""

    async def test_auth_error_is_not_retried(self):
        for status in (401, 403):
            plugin = _make_plugin(grok_search_retries=2)
            module, state = _fake_aiohttp([(status, "denied")])
            with patch.dict(sys.modules, {"aiohttp": module}), patch(
                "asyncio.sleep", new=_noop_sleep
            ):
                result = await plugin._grok_search("query")
            self.assertEqual(state["attempts"], 1, status)
            self.assertEqual(result, "")

    async def test_rate_limit_is_retried(self):
        plugin = _make_plugin(grok_search_retries=1)
        module, state = _fake_aiohttp(
            [(429, "slow down"), (200, {"output_text": "检索到的要点。"})]
        )
        with patch.dict(sys.modules, {"aiohttp": module}), patch(
            "asyncio.sleep", new=_noop_sleep
        ):
            result = await plugin._grok_search("query")
        self.assertEqual(state["attempts"], 2)
        self.assertIn("检索到的要点", result)

# 408（请求超时）与 432 同样是瞬时故障：要重试，不能像 401/403 那样直接放弃
    async def test_transient_statuses_are_retried(self):
        for status in (408, 432):
            plugin = _make_plugin(grok_search_retries=1)
            module, state = _fake_aiohttp(
                [(status, "flaky"), (200, {"output_text": "检索到的要点。"})]
            )
            with patch.dict(sys.modules, {"aiohttp": module}), patch(
                "asyncio.sleep", new=_noop_sleep
            ):
                result = await plugin._grok_search("query")
            self.assertEqual(state["attempts"], 2, status)
            self.assertIn("检索到的要点", result)
# 200 却不是 JSON：不是瞬时故障，重试只会再失败一次
    async def test_non_json_body_is_not_retried(self):
        plugin = _make_plugin(grok_search_retries=2)
        module, state = _fake_aiohttp([(200, "not-json")])
        with patch.dict(sys.modules, {"aiohttp": module}), patch(
            "asyncio.sleep", new=_noop_sleep
        ):
            result = await plugin._grok_search("query")
        # 200 但解析不了：不是瞬时故障，重试也只是再失败一次
        self.assertEqual(state["attempts"], 1)
        self.assertEqual(result, "")
class _FlowEvent:
    """_verify / handle_message 用的事件最小实现。"""

    def __init__(self, quoted_id: str = "", sender: str = "u1"):
        self._messages = [main.Reply(id=quoted_id)] if quoted_id else []
        self._sender = sender

    def get_messages(self):
        return self._messages

    def get_sender_id(self):
        return self._sender

    def get_group_id(self):
        return ""

    def get_platform_name(self):
        return "test-platform"

    def plain_result(self, text):
        return ("plain", text)

    def chain_result(self, chain):
        return ("chain", chain)


class VerifyFlowTests(unittest.IsolatedAsyncioTestCase):
    """_verify / handle_message 的关键路径：缓存、unknown 不缓存、空内容不吃冷却、整体超时。"""

    def _plugin(self, answer: str, **overrides):
        plugin = _make_plugin(**overrides)
        calls = {"plan": 0, "chat": 0}

        async def _extract_bundle(event, strip_keyword):
            return {
                "text": "某说法",
                "images": [],
                "supplement": "",
                "source": "text",
                "image_note": "",
            }

        async def _plan_verification(**_kwargs):
            calls["plan"] += 1
            return {"claim": "某说法", "queries": [], "need_search": False, "note": ""}

        async def _chat_with_fallback(**_kwargs):
            calls["chat"] += 1
            return types.SimpleNamespace(completion_text=answer), "stub-model", False

        async def _noop(event, *_args, **_kwargs):
            return None

        plugin._extract_bundle = _extract_bundle
        plugin._plan_verification = _plan_verification
        plugin._chat_with_fallback = _chat_with_fallback
        plugin._start_emoji = _noop
        plugin._finish_emoji = _noop
        plugin._build_result_chain = lambda content, event: ("chain", content)
        return plugin, calls

    async def test_true_verdict_is_cached_and_reused(self):
        plugin, calls = self._plugin("✅ 真的喵\n官方公告属实。")
        event = _FlowEvent(quoted_id="700")
        first = await plugin._verify(event, "")
        self.assertEqual(len(first), 1)
        self.assertTrue(
            any(key.startswith("quoted:700:") for key in plugin._result_cache)
        )

        plan_calls = calls["plan"]
        second = await plugin._verify(event, "")
        self.assertEqual(calls["plan"], plan_calls)
        self.assertEqual(first, second)

    async def test_unknown_verdict_is_not_cached(self):
        plugin, _calls = self._plugin("⚠️ 布吉岛\n无法核实，查不到可信来源。")
        event = _FlowEvent(quoted_id="701")
        await plugin._verify(event, "")
        self.assertEqual(plugin._result_cache, {})

    async def test_empty_claim_does_not_start_cooldown(self):
        plugin, _calls = self._plugin("✅ 真的")

        async def _empty_bundle(event, strip_keyword):
            return {
                "text": "",
                "images": [],
                "supplement": "",
                "source": "text",
                "image_note": "",
            }

        plugin._extract_bundle = _empty_bundle
        event = _FlowEvent(sender="u9")
        results = await plugin._verify(event, "")
        self.assertEqual(results[0][0], "plain")
        self.assertIn("请 @我", results[0][1])
        self.assertNotIn("u9", plugin._cooldowns)

    async def test_total_timeout_tells_user(self):
        plugin, _calls = self._plugin("✅ 真的喵")
        plugin.total_timeout = 0.1

        async def _slow(event, strip_keyword, emoji_state=None):
            await asyncio.sleep(1.0)
            return []

        plugin._verify = _slow
        plugin._match_trigger = lambda event: (True, "")
        event = _FlowEvent(sender="u7")
        results = [item async for item in plugin.handle_message(event)]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "plain")
# 超时提示要说明已中止，并给出可调配置的提示
        self.assertIn("已中止", results[0][1])

    async def test_normal_check_reserves_cooldown(self):
        plugin, _calls = self._plugin("✅ 真的喵，官方公告属实。")
        plugin._match_trigger = lambda event: (True, "")
        event = _FlowEvent(sender="u8")
        results = [item async for item in plugin.handle_message(event)]
        self.assertEqual(results[0][0], "chain")
        self.assertIn("u8", plugin._cooldowns)

    async def test_cancelled_check_clears_emoji(self):
        plugin, _calls = self._plugin("✅ 真的喵")
        plugin._match_trigger = lambda event: (True, "")
        emojis = []

# 真跑到一半被取消：撤掉「处理中」表情，取消继续上抛
        async def _slow(event, strip_keyword, emoji_state=None):
            # 模拟真实 _verify：贴过「处理中」表情后才开始长耗时终判
            if emoji_state is not None:
                emoji_state["started"] = True
            await asyncio.sleep(5)
        async def _record(event, verdict, *_args):
            emojis.append(verdict)

        plugin._verify = _slow
        plugin._finish_emoji = _record
        event = _FlowEvent(sender="u5")
        task = asyncio.create_task(plugin.handle_message(event).__anext__())
        await asyncio.sleep(0.05)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(emojis, ["error"])

# 排队中就被取消：没开工、没贴过表情，不该去撤表情，冷却也要退回
    async def test_queued_cancel_does_not_touch_emoji_or_cooldown(self):
        plugin, _calls = self._plugin("✅ 真的喵")
        plugin._match_trigger = lambda event: (True, "")
        emojis = []

        async def _record(event, verdict, *_args):
            emojis.append(verdict)

        plugin._finish_emoji = _record
        plugin._semaphore = asyncio.Semaphore(1)
        event = _FlowEvent(sender="u6")
        await plugin._semaphore.acquire()
        try:
            task = asyncio.create_task(plugin.handle_message(event).__anext__())
            await asyncio.sleep(0.05)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            plugin._semaphore.release()
        self.assertEqual(emojis, [])
        self.assertNotIn("u6", plugin._cooldowns)
if __name__ == "__main__":
    unittest.main()