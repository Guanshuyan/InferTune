"""内部网关适配客户端。

将现有 tools/llm/local_inference.py 的网关接口包装为 OpenAI 协议格式，
复用其限流器和重试逻辑，对外暴露与 OpenAIClient 一致的接口。
"""

import json
import os
import threading
import time
from collections import deque
from typing import Any

import requests

from .base import BaseLLM, ChatCompletionResponse


class GatewayClient(BaseLLM):
    """内部网关 LLM 客户端，适配为 OpenAI 协议格式。

    参数:
        model: 模型名称，默认读取环境变量 LLM_MODEL
        api_url: 网关地址，默认读取环境变量 LLM_API_URL
        api_key: 网关密钥，默认读取环境变量 LLM_API_KEY
        temperature: 默认采样温度
        max_tokens: 默认最大生成 token 数
        qps: 每秒请求数限制
        qpm: 每分钟请求数限制
        timeout: 单次请求超时秒数
    """

    def __init__(
        self,
        model: str | None = None,
        api_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        qps: int = 100,
        qpm: int = 600,
        timeout: int = 120,
    ):
        resolved_model = model or os.getenv("LLM_MODEL", "deepseek-v3.2-exp")
        super().__init__(model=resolved_model, temperature=temperature, max_tokens=max_tokens)

        self.api_url = api_url or os.getenv(
            "LLM_API_URL", "http://ai-llm-gateway.amap.com/open_api/v1/chat"
        )
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.timeout = timeout
        self._rate_limiter = _SlidingWindowRateLimiter(qps=qps, qpm=qpm)

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        """通过内部网关发起聊天补全请求。

        参数:
            messages: OpenAI 格式的消息列表
            model: 覆盖默认模型名称
            temperature: 覆盖默认采样温度
            max_tokens: 覆盖默认最大生成 token 数
            **kwargs: 透传给网关的额外参数（会合并到请求 body 中）

        返回:
            ChatCompletionResponse 响应对象
        """
        m, t, mt = self._resolve_params(model, temperature, max_tokens)

        # 限流
        self._rate_limiter.acquire()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": m,
            "messages": messages,
            "max_tokens": mt,
            "temperature": t,
        }
        if kwargs:
            data.update(kwargs)

        resp = requests.post(
            self.api_url, headers=headers, json=data, timeout=self.timeout
        )
        resp.raise_for_status()
        body = resp.json()

        # 解析为统一格式
        content = ""
        if body.get("choices"):
            content = body["choices"][0].get("message", {}).get("content", "")

        usage = {}
        if body.get("usage"):
            usage = {
                "prompt_tokens": body["usage"].get("prompt_tokens", 0),
                "completion_tokens": body["usage"].get("completion_tokens", 0),
                "total_tokens": body["usage"].get("total_tokens", 0),
            }

        return ChatCompletionResponse(
            content=content,
            model=body.get("model", m),
            usage=usage,
            raw=body,
        )


class _SlidingWindowRateLimiter:
    """线程安全的 QPS/QPM 滑动窗口限流器。

    参数:
        qps: 每秒最大请求数
        qpm: 每分钟最大请求数
    """

    def __init__(self, qps: int = 100, qpm: int = 600):
        self.qps = int(qps)
        self.qpm = int(qpm)
        self._window_sec: deque = deque()
        self._window_min: deque = deque()
        self._sum_sec = 0
        self._sum_min = 0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """阻塞直到当前请求通过速率限制。"""
        while True:
            now = time.monotonic()
            with self._lock:
                # 清理秒级窗口
                cutoff_sec = now - 1.0
                while self._window_sec and self._window_sec[0][0] <= cutoff_sec:
                    _, cnt = self._window_sec.popleft()
                    self._sum_sec -= cnt

                # 清理分钟级窗口
                cutoff_min = now - 60.0
                while self._window_min and self._window_min[0][0] <= cutoff_min:
                    _, cnt = self._window_min.popleft()
                    self._sum_min -= cnt

                if self._sum_sec < self.qps and self._sum_min < self.qpm:
                    self._window_sec.append((now, 1))
                    self._window_min.append((now, 1))
                    self._sum_sec += 1
                    self._sum_min += 1
                    return

                wait_sec = 0.01
                if self._sum_sec >= self.qps and self._window_sec:
                    wait_sec = max(wait_sec, (self._window_sec[0][0] + 1.0) - now)
                if self._sum_min >= self.qpm and self._window_min:
                    wait_sec = max(wait_sec, (self._window_min[0][0] + 60.0) - now)

            time.sleep(max(wait_sec, 0.001))
