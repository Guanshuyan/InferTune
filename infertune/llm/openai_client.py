"""OpenAI 兼容 API 客户端。

支持所有兼容 OpenAI 协议的后端（OpenAI、vLLM、Ollama、各云厂商等），
通过 base_url 和 api_key 切换不同服务。
"""

from typing import Any

from openai import OpenAI

from .base import BaseLLM, ChatCompletionResponse


class OpenAIClient(BaseLLM):
    """基于 openai SDK 的标准客户端。

    参数:
        model: 模型名称，如 "gpt-4o" / "deepseek-chat"
        api_key: API 密钥
        base_url: API 地址，默认为 OpenAI 官方地址；切换为其他兼容服务时修改此项
        temperature: 默认采样温度
        max_tokens: 默认最大生成 token 数
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        """通过 openai SDK 发起聊天补全请求。

        参数:
            messages: OpenAI 格式的消息列表
            model: 覆盖默认模型名称
            temperature: 覆盖默认采样温度
            max_tokens: 覆盖默认最大生成 token 数
            **kwargs: 透传给 openai SDK 的额外参数（如 top_p、stop 等）

        返回:
            ChatCompletionResponse 响应对象
        """
        m, t, mt = self._resolve_params(model, temperature, max_tokens)

        resp = self._client.chat.completions.create(
            model=m,
            messages=messages,
            temperature=t,
            max_tokens=mt,
            **kwargs,
        )

        # 解析 usage 信息
        usage = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens or 0,
                "completion_tokens": resp.usage.completion_tokens or 0,
                "total_tokens": resp.usage.total_tokens or 0,
            }

        return ChatCompletionResponse(
            content=resp.choices[0].message.content or "",
            model=resp.model or m,
            usage=usage,
            raw=resp.model_dump(),
        )
