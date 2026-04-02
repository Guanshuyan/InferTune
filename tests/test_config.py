"""配置模块单元测试。

测试 YAML 加载、默认值、dataclass 转换。
"""

import tempfile
from pathlib import Path

from infertune.config.settings import (
    InferTuneConfig,
    LLMConfig,
    load_config,
)


class TestLoadConfig:
    """配置加载测试。"""

    def test_default_config(self):
        """不存在的路径应返回全默认配置。"""
        config = load_config("/nonexistent/path.yaml")
        assert isinstance(config, InferTuneConfig)
        assert config.llm.backend == "openai"
        assert config.protegi.beam_width == 4

    def test_load_yaml(self):
        """从 YAML 文件加载配置。"""
        yaml_content = """
llm:
  backend: gateway
  model: deepseek-v3
  temperature: 0.5
protegi:
  beam_width: 8
  search_depth: 3
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = load_config(f.name)

        assert config.llm.backend == "gateway"
        assert config.llm.model == "deepseek-v3"
        assert config.llm.temperature == 0.5
        assert config.protegi.beam_width == 8
        assert config.protegi.search_depth == 3
        # 未指定的字段应保持默认值
        assert config.llm.max_tokens == 4096
        assert config.ace.mode == "offline"

    def test_partial_config(self):
        """部分配置应与默认值合并。"""
        yaml_content = """
eval:
  metric: f1
  num_samples: 100
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = load_config(f.name)

        assert config.eval.metric == "f1"
        assert config.eval.num_samples == 100
        assert config.llm.backend == "openai"  # 默认值
