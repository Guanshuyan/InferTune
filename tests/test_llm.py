"""LLM 模块单元测试。

测试 BaseLLM 接口、OpenAIClient 构造、GatewayClient 构造及参数合并逻辑。
"""

from unittest.mock import MagicMock, patch

from infertune.llm.base import BaseLLM, ChatCompletionResponse, ChatMessage
from infertune.llm.openai_client import OpenAIClient
from infertune.llm.gateway_client import GatewayClient


class _MockLLM(BaseLLM):
    """用于测试的 BaseLLM 模拟实现。"""

    def chat_completion(self, messages, *, model=None, temperature=None, max_tokens=None, **kwargs):
        m, t, mt = self._resolve_params(model, temperature, max_tokens)
        return ChatCompletionResponse(
            content=f"mock reply with model={m}",
            model=m,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )


class TestChatMessage:
    """ChatMessage 数据类测试。"""

    def test_to_dict(self):
        msg = ChatMessage(role="user", content="hello")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "hello"}


class TestBaseLLM:
    """BaseLLM 抽象基类测试。"""

    def test_resolve_params_defaults(self):
        llm = _MockLLM(model="test-model", temperature=0.5, max_tokens=1024)
        m, t, mt = llm._resolve_params(None, None, None)
        assert m == "test-model"
        assert t == 0.5
        assert mt == 1024

    def test_resolve_params_override(self):
        llm = _MockLLM(model="test-model", temperature=0.5, max_tokens=1024)
        m, t, mt = llm._resolve_params("other-model", 0.9, 2048)
        assert m == "other-model"
        assert t == 0.9
        assert mt == 2048

    def test_chat_returns_string(self):
        llm = _MockLLM(model="test-model")
        result = llm.chat([{"role": "user", "content": "hi"}])
        assert isinstance(result, str)
        assert "mock reply" in result

    def test_chat_completion_returns_response(self):
        llm = _MockLLM(model="test-model")
        resp = llm.chat_completion([{"role": "user", "content": "hi"}])
        assert isinstance(resp, ChatCompletionResponse)
        assert resp.content.startswith("mock reply")
        assert resp.usage["total_tokens"] == 15


class TestOpenAIClient:
    """OpenAIClient 构造测试（不实际调用 API）。"""

    def test_init_default(self):
        with patch("infertune.llm.openai_client.OpenAI"):
            client = OpenAIClient(model="gpt-4o", api_key="test-key")
            assert client.model == "gpt-4o"
            assert client.temperature == 0.7
            assert client.max_tokens == 4096

    def test_init_custom_params(self):
        with patch("infertune.llm.openai_client.OpenAI"):
            client = OpenAIClient(
                model="deepseek-chat",
                base_url="http://localhost:8000/v1",
                temperature=0.3,
                max_tokens=2048,
            )
            assert client.model == "deepseek-chat"
            assert client.temperature == 0.3


class TestGatewayClient:
    """GatewayClient 构造测试（不实际调用网关）。"""

    def test_init_default(self):
        client = GatewayClient(model="deepseek-v3", api_key="test-key")
        assert client.model == "deepseek-v3"
        assert client.temperature == 0.8

    def test_init_env_fallback(self):
        with patch.dict("os.environ", {"LLM_MODEL": "env-model", "LLM_API_KEY": "env-key"}):
            client = GatewayClient()
            assert client.model == "env-model"
            assert client.api_key == "env-key"

    @patch("infertune.llm.gateway_client.requests.post")
    def test_chat_completion(self, mock_post):
        """模拟网关响应，验证返回格式。"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "gateway reply"}}],
            "model": "deepseek-v3",
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = GatewayClient(model="deepseek-v3", api_key="test-key", qps=1000, qpm=60000)
        resp = client.chat_completion([{"role": "user", "content": "test"}])

        assert resp.content == "gateway reply"
        assert resp.model == "deepseek-v3"
        assert resp.usage["total_tokens"] == 30
