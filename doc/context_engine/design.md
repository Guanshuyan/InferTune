# ACE 上下文工程模块设计

## 概述
基于 ACE 论文实现的 Agentic Context Engineering 模块。核心思想：将上下文视为 evolving playbook，通过 Generator/Reflector/Curator 三角色协作，用增量 delta 更新持续积累和精炼策略，避免 brevity bias 和 context collapse。

## 流程
```
[Generator] 使用当前 Playbook 执行任务 → 推理轨迹 + 策略反馈
     ↓
[Reflector] 分析轨迹 → 提取洞察 → 多轮迭代精炼
     ↓
[Curator] 洞察 → delta Bullets → 确定性合并到 Playbook
     ↓
[Dedup] 周期性语义去重 + 低质量清除
```

## 两种运行模式
- **Offline**: 在训练数据上多 epoch 迭代优化 Playbook，用于系统提示优化
- **Online**: 逐条处理输入并实时更新 Playbook，用于测试时记忆适应

## 模块组成
- `playbook.py` — Bullet 数据结构（id/content/helpful_count/harmful_count/tags）+ Playbook 容器（增删改查/渲染/序列化/delta 更新）
- `generator.py` — 使用 Playbook 上下文执行任务，生成轨迹并标注策略有用性
- `reflector.py` — 从轨迹中提取洞察，支持多轮迭代精炼
- `curator.py` — 将洞察转化为 delta Bullets，通过非 LLM 确定性逻辑合并（避免 context collapse）
- `dedup.py` — 基于词重叠/embedding 的语义去重（grow-and-refine 机制）
- `engine.py` — 顶层 API，封装 offline/online 两种模式

## 关键配置（ACEConfig）
| 参数 | 默认值 | 说明 |
|------|--------|------|
| mode | "offline" | 运行模式 |
| max_reflector_rounds | 5 | Reflector 迭代精炼轮数 |
| max_epochs | 5 | offline 模式 epoch 数 |
| dedup_threshold | 0.85 | 语义去重阈值 |
| batch_size | 1 | 每批处理样本数 |

## 使用方式
```python
from infertune.llm import OpenAIClient
from infertune.context_engine.engine import ContextEngine
from infertune.config import ACEConfig

llm = OpenAIClient(model="gpt-4o", api_key="...")
config = ACEConfig(mode="offline", max_epochs=3)
engine = ContextEngine(llm, config)

# Offline 优化
result = engine.optimize_offline(
    train_data=[{"input": "问题", "label": "答案"}, ...],
    system_prompt="你是一个分类助手",
    verbose=True,
)
print(result["context"])  # 渲染后的 Playbook 上下文

# Online 适应
for item in test_data:
    result = engine.optimize_online(item["input"], label=item["label"])
    print(result["answer"], result["playbook_size"])
```
