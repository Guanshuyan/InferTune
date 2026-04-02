"""统一配置管理，支持 YAML 文件加载和 dataclass 结构化访问。

使用方式:
    config = load_config("configs/default.yaml")
    llm = config.llm
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LLMConfig:
    """LLM 后端配置。

    参数:
        backend: 后端类型，"openai" 或 "gateway"
        model: 模型名称
        api_key: API 密钥
        base_url: API 地址（openai 后端使用）
        api_url: 网关地址（gateway 后端使用）
        temperature: 采样温度
        max_tokens: 最大生成 token 数
        qps: 每秒请求数限制（gateway 后端使用）
        qpm: 每分钟请求数限制（gateway 后端使用）
    """

    backend: str = "openai"
    model: str = "gpt-4o"
    api_key: str = ""
    base_url: str = ""
    api_url: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    qps: int = 100
    qpm: int = 600


@dataclass
class EvalConfig:
    """评估模块配置。

    参数:
        metric: 默认评估指标名称，如 "f1" / "accuracy"
        num_samples: 评估采样数量
    """

    metric: str = "accuracy"
    num_samples: int = 50


@dataclass
class ProTeGiConfig:
    """ProTeGi 提示优化模块配置。

    参数:
        beam_width: beam search 宽度
        search_depth: 搜索深度（优化步数）
        minibatch_size: 每步采样的 minibatch 大小
        num_gradients: 每组错误生成的梯度数量
        num_edits_per_gradient: 每个梯度生成的编辑候选数
        num_monte_carlo: 每个候选的 paraphrase 扩展数
        selection_method: 候选选择算法，"ucb" 或 "successive_rejects"
    """

    beam_width: int = 4
    search_depth: int = 6
    minibatch_size: int = 64
    num_gradients: int = 4
    num_edits_per_gradient: int = 1
    num_monte_carlo: int = 2
    selection_method: str = "ucb"


@dataclass
class ACEConfig:
    """ACE 上下文工程模块配置。

    参数:
        mode: 运行模式，"offline" 或 "online"
        max_reflector_rounds: Reflector 最大迭代精炼轮数
        max_epochs: offline 模式最大 epoch 数
        dedup_threshold: 语义去重的相似度阈值
        batch_size: 每批处理的样本数
    """

    mode: str = "offline"
    max_reflector_rounds: int = 5
    max_epochs: int = 5
    dedup_threshold: float = 0.85
    batch_size: int = 1


@dataclass
class InferTuneConfig:
    """InferTune 全局配置。

    参数:
        llm: LLM 后端配置
        eval: 评估模块配置
        protegi: ProTeGi 模块配置
        ace: ACE 模块配置
    """

    llm: LLMConfig = field(default_factory=LLMConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    protegi: ProTeGiConfig = field(default_factory=ProTeGiConfig)
    ace: ACEConfig = field(default_factory=ACEConfig)


def _dict_to_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """将字典递归转换为 dataclass 实例，忽略多余字段。

    参数:
        cls: 目标 dataclass 类型
        data: 源数据字典

    返回:
        对应的 dataclass 实例
    """
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered = {}
    for k, v in data.items():
        if k not in field_names:
            continue
        f = next(f for f in dataclasses.fields(cls) if f.name == k)
        # 如果字段本身也是 dataclass，递归转换
        if dataclasses.is_dataclass(f.type) and isinstance(v, dict):
            filtered[k] = _dict_to_dataclass(f.type, v)
        else:
            filtered[k] = v
    return cls(**filtered)


def load_config(path: str | Path) -> InferTuneConfig:
    """从 YAML 文件加载配置。

    参数:
        path: YAML 配置文件路径

    返回:
        InferTuneConfig 配置对象
    """
    path = Path(path)
    if not path.exists():
        return InferTuneConfig()

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return _build_config(raw)


def _build_config(raw: dict[str, Any]) -> InferTuneConfig:
    """从字典构建配置对象。

    参数:
        raw: 原始配置字典

    返回:
        InferTuneConfig 配置对象
    """
    llm = _dict_to_dataclass(LLMConfig, raw.get("llm", {}))
    eval_cfg = _dict_to_dataclass(EvalConfig, raw.get("eval", {}))
    protegi = _dict_to_dataclass(ProTeGiConfig, raw.get("protegi", {}))
    ace = _dict_to_dataclass(ACEConfig, raw.get("ace", {}))
    return InferTuneConfig(llm=llm, eval=eval_cfg, protegi=protegi, ace=ace)
