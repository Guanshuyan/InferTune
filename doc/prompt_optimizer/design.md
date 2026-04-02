# ProTeGi 提示优化模块设计

## 概述
基于 ProTeGi 论文实现的自动 prompt 优化模块。核心思想：用"文本梯度"描述当前 prompt 的缺陷，再反向编辑 prompt，通过 beam search + bandit 选择在 prompt 空间中高效搜索最优解。

## 流程
```
初始 prompt
  ↓
[Beam Search 外层循环] × search_depth 步
  ├── 采样 minibatch → 运行 prompt → 收集错误样本
  ├── 错误样本 → 梯度生成器(Δ) → 自然语言梯度
  ├── 梯度 + 当前 prompt → 编辑器(δ) → 候选 prompt
  ├── 候选 → paraphrase 扩展 → 更多候选
  └── 所有候选 → Bandit 选择 → top beam_width 个进入下一轮
  ↓
最优 prompt
```

## 模块组成
- `gradient.py` — 文本梯度生成（Δ prompt），将错误样本分组后让 LLM 分析 prompt 缺陷
- `editor.py` — Prompt 编辑器（δ prompt），根据梯度反向编辑 + paraphrase 扩展候选空间
- `bandit.py` — 候选选择算法：UCB（平衡探索利用）/ Successive Rejects（理论最优淘汰）
- `beam_search.py` — Beam search 外层循环，串联梯度→编辑→选择的完整流程
- `optimizer.py` — 顶层 API，封装配置和评估逻辑，一键优化

## 关键配置（ProTeGiConfig）
| 参数 | 默认值 | 说明 |
|------|--------|------|
| beam_width | 4 | beam 宽度 |
| search_depth | 6 | 优化步数 |
| minibatch_size | 64 | 每步采样大小 |
| num_gradients | 4 | 每步生成的梯度数 |
| num_edits_per_gradient | 1 | 每个梯度的编辑候选数 |
| num_monte_carlo | 2 | paraphrase 扩展数 |
| selection_method | "ucb" | 选择算法 |

## 使用方式
```python
from infertune.llm import OpenAIClient
from infertune.evaluator import AccuracyEvaluator
from infertune.prompt_optimizer.optimizer import PromptOptimizer
from infertune.config import ProTeGiConfig

llm = OpenAIClient(model="gpt-4o", api_key="...")
evaluator = AccuracyEvaluator()
config = ProTeGiConfig(beam_width=4, search_depth=3)
optimizer = PromptOptimizer(llm, config)

result = optimizer.optimize(
    initial_prompt="判断以下文本的情感倾向",
    train_data=[{"input": "好电影", "label": "positive"}, ...],
    eval_data=[{"input": "烂片", "label": "negative"}, ...],
    evaluator=evaluator,
    verbose=True,
)
print(result["best_prompt"], result["improvement"])
```
