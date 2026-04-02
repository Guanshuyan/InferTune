"""LLM 调用次数估算器。

在实际运行优化流程前，根据数据集大小和参数配置估算 LLM 调用次数，
帮助用户进行消费预估和成本控制。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config.settings import ProTeGiConfig, ACEConfig


@dataclass
class CostEstimate:
    """调用次数估算结果。

    参数:
        total_calls: 预估总 LLM 调用次数
        breakdown: 各阶段调用次数明细
        notes: 补充说明
    """

    total_calls: int
    breakdown: dict[str, int]
    notes: list[str]

    def summary(self) -> str:
        """生成可读的估算摘要。

        返回:
            格式化的摘要文本
        """
        lines = [f"预估 LLM 总调用次数: {self.total_calls}"]
        lines.append("")
        lines.append("调用明细:")
        for stage, count in self.breakdown.items():
            lines.append(f"  {stage}: {count}")
        if self.notes:
            lines.append("")
            lines.append("说明:")
            for note in self.notes:
                lines.append(f"  - {note}")
        return "\n".join(lines)


def estimate_protegi(
    config: ProTeGiConfig,
    train_size: int,
    eval_size: int,
) -> CostEstimate:
    """估算 ProTeGi 优化流程的 LLM 调用次数。

    调用链路分析（每步 × beam_width 个 parent）:
    1. 错误收集: minibatch_size 次 run_fn 调用（每次 1 次 LLM）
    2. 梯度生成: num_gradients 次 LLM 调用
    3. Prompt 编辑: num_gradients × num_edits_per_gradient 次编辑
    4. Paraphrase: num_gradients × num_edits_per_gradient × num_monte_carlo 次
    5. Bandit 选择: selection_budget 次 eval_fn，每次 eval_fn 遍历 eval_size 条数据
    6. 初始评估 + 最终评估: 各 eval_size 次

    参数:
        config: ProTeGi 配置
        train_size: 训练数据条数
        eval_size: 评估数据条数

    返回:
        CostEstimate 估算结果
    """
    cfg = config
    minibatch = min(cfg.minibatch_size, train_size)

    # 每个 parent prompt 每步的调用
    error_collect_per_parent = minibatch
    gradient_gen_per_parent = cfg.num_gradients
    edits_per_parent = cfg.num_gradients * cfg.num_edits_per_gradient
    paraphrase_per_parent = edits_per_parent * cfg.num_monte_carlo
    expand_per_parent = gradient_gen_per_parent + edits_per_parent + paraphrase_per_parent

    # 每步所有 parent 的扩展调用
    expand_per_step = cfg.beam_width * (error_collect_per_parent + expand_per_parent)

    # bandit 选择：每次 eval_fn 调用 = eval_size 次 LLM
    # selection_budget 默认 50（beam_search 的默认参数）
    selection_budget = 50
    select_per_step = selection_budget * eval_size

    per_step = expand_per_step + select_per_step
    beam_search_total = cfg.search_depth * per_step

    # 初始评估 + 最终评估
    init_eval = eval_size
    final_eval = eval_size

    total = init_eval + beam_search_total + final_eval

    breakdown = {
        "初始评估": init_eval,
        f"Beam Search ({cfg.search_depth} 步)": beam_search_total,
        f"  每步 - 错误收集 (×{cfg.beam_width} parents)": cfg.search_depth * cfg.beam_width * error_collect_per_parent,
        f"  每步 - 梯度生成": cfg.search_depth * cfg.beam_width * gradient_gen_per_parent,
        f"  每步 - Prompt 编辑": cfg.search_depth * cfg.beam_width * edits_per_parent,
        f"  每步 - Paraphrase 扩展": cfg.search_depth * cfg.beam_width * paraphrase_per_parent,
        f"  每步 - Bandit 选择": cfg.search_depth * select_per_step,
        "最终评估": final_eval,
    }

    notes = [
        f"beam_width={cfg.beam_width}, search_depth={cfg.search_depth}",
        f"minibatch_size={minibatch}（实际取 min(配置值, 训练集大小)）",
        f"每次 eval_fn 调用 = {eval_size} 次 LLM（遍历评估集）",
        "实际调用可能略少：若某步无错误样本则跳过梯度生成和编辑",
    ]

    return CostEstimate(total_calls=total, breakdown=breakdown, notes=notes)


def estimate_ace_offline(
    config: ACEConfig,
    train_size: int,
    eval_size: int = 0,
    has_evaluator: bool = False,
) -> CostEstimate:
    """估算 ACE offline 优化流程的 LLM 调用次数。

    每条训练数据的调用链路:
    1. Generator: 1 次 LLM
    2. Reflector: 1 次提取 + (max_reflector_rounds - 1) 次精炼
    3. Curator: 纯逻辑，无 LLM 调用

    每个 epoch 结束后若有 evaluator:
    4. 评估: eval_size 次 LLM（Generator 跑评估集）

    参数:
        config: ACE 配置
        train_size: 训练数据条数
        eval_size: 评估数据条数
        has_evaluator: 是否配置了评估器

    返回:
        CostEstimate 估算结果
    """
    cfg = config

    # 每条数据的调用
    generator_per_item = 1
    reflector_per_item = 1 + max(cfg.max_reflector_rounds - 1, 0)  # 初始提取 + 精炼轮数
    per_item = generator_per_item + reflector_per_item

    # 每个 epoch
    train_per_epoch = train_size * per_item
    eval_per_epoch = eval_size if has_evaluator else 0
    per_epoch = train_per_epoch + eval_per_epoch

    total = cfg.max_epochs * per_epoch

    breakdown = {
        f"训练迭代 ({cfg.max_epochs} epochs × {train_size} 条)": cfg.max_epochs * train_per_epoch,
        f"  每条 - Generator": cfg.max_epochs * train_size * generator_per_item,
        f"  每条 - Reflector ({cfg.max_reflector_rounds} 轮)": cfg.max_epochs * train_size * reflector_per_item,
    }

    if has_evaluator:
        breakdown[f"阶段评估 ({cfg.max_epochs} epochs × {eval_size} 条)"] = cfg.max_epochs * eval_per_epoch

    notes = [
        f"max_epochs={cfg.max_epochs}, max_reflector_rounds={cfg.max_reflector_rounds}",
        "Curator 为纯逻辑操作，不消耗 LLM 调用",
    ]
    if not has_evaluator:
        notes.append("未配置评估器，无阶段评估调用")

    return CostEstimate(total_calls=total, breakdown=breakdown, notes=notes)


def estimate_ace_online(
    config: ACEConfig,
    num_inputs: int,
) -> CostEstimate:
    """估算 ACE online 模式的 LLM 调用次数。

    参数:
        config: ACE 配置
        num_inputs: 待处理的输入条数

    返回:
        CostEstimate 估算结果
    """
    cfg = config

    generator_per_item = 1
    reflector_per_item = 1 + max(cfg.max_reflector_rounds - 1, 0)
    per_item = generator_per_item + reflector_per_item

    total = num_inputs * per_item

    breakdown = {
        f"逐条处理 ({num_inputs} 条)": total,
        f"  每条 - Generator": num_inputs * generator_per_item,
        f"  每条 - Reflector ({cfg.max_reflector_rounds} 轮)": num_inputs * reflector_per_item,
    }

    notes = [
        f"max_reflector_rounds={cfg.max_reflector_rounds}",
        "Online 模式逐条处理，Playbook 实时更新",
    ]

    return CostEstimate(total_calls=total, breakdown=breakdown, notes=notes)
