"""LLM 抽象基类，定义 OpenAI 协议格式的统一调用接口。

所有 LLM 后端（OpenAI API、本地网关等）都需要继承此基类，
并实现 chat_completion 方法，对外暴露统一的 OpenAI 协议格式。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatMessage:
    """单条聊天消息，对应 OpenAI 协议中的 message 对象。

    参数:
        role: 消息角色，如 "system" / "user" / "assistant"
        content: 消息文本内容
    """

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        """转换为 OpenAI 协议格式的字典。"""
        return {"role": self.role, "content": self.content}


@dataclass
class ChatCompletionResponse:
    """聊天补全响应，对应 OpenAI 协议中的 response 对象。

    参数:
        content: 助手回复的文本内容
        model: 实际使用的模型名称
        usage: token 用量信息 {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
        raw: 原始响应数据，供调试使用
    """

    content: str
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


class BaseLLM(ABC):
    """LLM 抽象基类。

    所有后端实现需要继承此类并实现 chat_completion 方法。
    对外提供 chat / chat_completion 两个调用入口。

    参数:
        model: 默认模型名称
        temperature: 默认采样温度
        max_tokens: 默认最大生成 token 数
    """

    def __init__(
        self,
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        """发起聊天补全请求。

        参数:
            messages: OpenAI 格式的消息列表 [{"role": "...", "content": "..."}]
            model: 覆盖默认模型名称
            temperature: 覆盖默认采样温度
            max_tokens: 覆盖默认最大生成 token 数
            **kwargs: 其他后端特定参数

        返回:
            ChatCompletionResponse 响应对象
        """

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """发起聊天请求，直接返回文本内容。

        参数:
            messages: OpenAI 格式的消息列表
            model: 覆盖默认模型名称
            temperature: 覆盖默认采样温度
            max_tokens: 覆盖默认最大生成 token 数
            **kwargs: 其他后端特定参数

        返回:
            助手回复的文本字符串
        """
        resp = self.chat_completion(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return resp.content

    def _resolve_params(
        self,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> tuple[str, float, int]:
        """合并调用参数与实例默认值。

        参数:
            model: 调用时传入的模型名称，None 则使用实例默认值
            temperature: 调用时传入的温度，None 则使用实例默认值
            max_tokens: 调用时传入的最大 token 数，None 则使用实例默认值

        返回:
            (model, temperature, max_tokens) 三元组
        """
        return (
            model or self.model,
            temperature if temperature is not None else self.temperature,
            max_tokens if max_tokens is not None else self.max_tokens,
        )
