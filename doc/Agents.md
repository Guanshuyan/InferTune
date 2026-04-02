# 项目描述
repo目标：构建一个无需训练，仅在推理时优化上下文即可提升模型任务表现的集成框架
集成自动提示优化/上下文自动优化，skill优化等方法。

核心方法来源：
- **ProTeGi**: 文本梯度驱动的自动 prompt 优化（梯度生成 → prompt 编辑 → beam search 选择）
- **ACE**: Agentic 上下文工程（Generator/Reflector/Curator 三角色 + 增量 delta 更新）
- **ORPO**: 借鉴 odds ratio 思想用于推理时响应质量评估/排序

# 项目结构
```
infertune/
├── llm/                  # LLM 统一接口层（OpenAI 协议格式）
├── prompt_optimizer/     # ProTeGi: 自动提示优化
├── context_engine/       # ACE: 上下文工程
├── evaluator/            # 评估模块
└── config/               # 配置管理
```

## doc
- `doc/Agents.md` — 项目概览与导航（本文件）
- `doc/papers/` — 相关论文
- `doc/llm/` — LLM 模块开发文档
- `doc/prompt_optimizer/` — ProTeGi 模块开发文档
- `doc/context_engine/` — ACE 模块开发文档

# 项目要求
## 代码组织
- LLM 层采用 OpenAI 协议标准格式，所有后端统一接口
- 各优化模块（ProTeGi、ACE）独立，通过 LLM 接口和评估模块解耦
- 配置统一使用 YAML + dataclass

## 文档管理
- `doc/Agents.md` 只做项目总体描述和导航，保持简洁
- 模块开发细节放在 doc 子文件夹中
- 所有文档易读、可维护、不冗长
- `README.md` 仅做项目介绍和项目使用方式，不用添加任何实现上的介绍细节

## 代码风格
- 代码开头需要说明该代码的作用
- 每一个函数需要说明参数和作用
- 中文注释

# 开发文档导航
- [LLM 模块设计](llm/design.md) — OpenAI 协议抽象层、OpenAI 客户端、网关适配
- [ProTeGi 模块设计](prompt_optimizer/design.md) — 文本梯度、prompt 编辑、beam search
- [ACE 模块设计](context_engine/design.md) — Generator/Reflector/Curator、Playbook、增量更新

# 开发进度
- [x] Phase 1: LLM 抽象层 + 配置 + 评估基础设施
- [x] Phase 2: ProTeGi 提示优化模块
- [x] Phase 3: ACE 上下文工程模块
- [x] Phase 4: 集成与示例