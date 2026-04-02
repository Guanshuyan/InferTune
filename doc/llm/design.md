# LLM 模块设计

## 概述
LLM 模块提供统一的 OpenAI 协议格式接口，所有后端（OpenAI API、本地网关等）对外暴露一致的调用方式。

## 架构
```
BaseLLM (抽象基类)
├── OpenAIClient    — 标准 OpenAI / 兼容 API（vLLM、Ollama、各云厂商）
└── GatewayClient   — 内部网关适配（复用 local_inference 限流逻辑）
```

## 核心接口
- `chat_completion(messages, model, temperature, max_tokens)` → `ChatCompletionResponse`
- `chat(messages, ...)` → `str`（便捷方法，直接返回文本）

## 数据结构
- `ChatMessage`: role + content
- `ChatCompletionResponse`: content + model + usage + raw

## 配置
通过 `configs/default.yaml` 的 `llm` 段配置：
- `backend`: "openai" 或 "gateway"
- `model` / `api_key` / `base_url` / `temperature` / `max_tokens`
- `qps` / `qpm`（gateway 限流参数）

## 文件清单
- `infertune/llm/base.py` — 抽象基类
- `infertune/llm/openai_client.py` — OpenAI 客户端
- `infertune/llm/gateway_client.py` — 网关适配客户端
