# InferTune

推理时上下文优化框架：无需训练，通过优化输入上下文提升 LLM 任务表现。

## 安装

```bash
pip install -e ".[dev]"
```

## 快速开始

### ProTeGi: 自动提示优化

```python
from infertune.llm import OpenAIClient
from infertune.evaluator import AccuracyEvaluator
from infertune.prompt_optimizer import PromptOptimizer
from infertune.config import ProTeGiConfig

llm = OpenAIClient(model="gpt-4o")
optimizer = PromptOptimizer(llm, ProTeGiConfig(beam_width=2, search_depth=3))

result = optimizer.optimize(
    initial_prompt="判断文本情感，输出 positive 或 negative",
    train_data=[{"input": "好电影", "label": "positive"}, ...],
    eval_data=[{"input": "烂片", "label": "negative"}, ...],
    evaluator=AccuracyEvaluator(),
)
print(result["best_prompt"], result["improvement"])
```

### ACE: 上下文工程

```python
from infertune.llm import OpenAIClient
from infertune.context_engine import ContextEngine
from infertune.config import ACEConfig

llm = OpenAIClient(model="gpt-4o")
engine = ContextEngine(llm, ACEConfig(max_epochs=3))

# Offline: 在训练数据上优化 Playbook
result = engine.optimize_offline(train_data, system_prompt="分类助手", verbose=True)

# Online: 逐条处理并实时更新
result = engine.optimize_online("新输入文本", label="标签")
```

## 运行测试

```bash
python -m pytest tests/ -v
```

## 项目文档

详见 [doc/Agents.md](doc/Agents.md)

## License

Apache-2.0
