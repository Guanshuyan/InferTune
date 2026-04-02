"""文本梯度生成器。

基于 ProTeGi 论文的核心思想：用 minibatch 中的错误样本生成自然语言"梯度"，
即描述当前 prompt 缺陷的文本，类似数值梯度指向误差上升方向。
"""

from typing import Any

from ..llm.base import BaseLLM


# 梯度生成 prompt 模板（对应论文中的 Δ prompt）
_GRADIENT_PROMPT_TEMPLATE = """我正在优化一个用于特定任务的 prompt。当前 prompt 在一些样本上产生了错误。
请分析这些错误，找出当前 prompt 的缺陷，并给出改进方向。

当前 prompt：
{current_prompt}

以下是产生错误的样本：
{error_examples}

请简洁地描述当前 prompt 存在的问题，以及应该如何改进。只输出分析结果，不要输出新的 prompt。"""


def generate_gradients(
    llm: BaseLLM,
    current_prompt: str,
    error_examples: list[dict[str, Any]],
    *,
    num_gradients: int = 4,
    errors_per_group: int = 4,
    gradient_template: str | None = None,
) -> list[str]:
    """从错误样本中生成自然语言梯度。

    将错误样本分组，每组生成一个梯度（对当前 prompt 缺陷的描述）。

    参数:
        llm: LLM 实例
        current_prompt: 当前待优化的 prompt
        error_examples: 错误样本列表，每个元素为 {"input": str, "label": str, "prediction": str}
        num_gradients: 要生成的梯度数量
        errors_per_group: 每组包含的错误样本数
        gradient_template: 自定义梯度生成 prompt 模板，需包含 {current_prompt} 和 {error_examples}

    返回:
        自然语言梯度列表
    """
    template = gradient_template or _GRADIENT_PROMPT_TEMPLATE

    # 将错误样本分组
    groups = _split_into_groups(error_examples, errors_per_group, num_gradients)

    gradients = []
    for group in groups:
        # 格式化错误样本为可读文本
        examples_text = _format_error_examples(group)

        prompt = template.format(
            current_prompt=current_prompt,
            error_examples=examples_text,
        )

        messages = [{"role": "user", "content": prompt}]
        gradient = llm.chat(messages, temperature=1.0)
        gradients.append(gradient.strip())

    return gradients


def _split_into_groups(
    items: list[Any],
    group_size: int,
    num_groups: int,
) -> list[list[Any]]:
    """将列表分成指定数量的组。

    如果样本不足以填满所有组，会循环复用样本。

    参数:
        items: 待分组的列表
        group_size: 每组大小
        num_groups: 目标组数

    返回:
        分组后的嵌套列表
    """
    if not items:
        return []

    groups = []
    for i in range(num_groups):
        start = (i * group_size) % len(items)
        group = []
        for j in range(group_size):
            idx = (start + j) % len(items)
            group.append(items[idx])
        groups.append(group)
    return groups


def _format_error_examples(examples: list[dict[str, Any]]) -> str:
    """将错误样本格式化为可读文本。

    参数:
        examples: 错误样本列表

    返回:
        格式化后的文本
    """
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(f"样本 {i}:")
        lines.append(f"  输入: {ex.get('input', '')}")
        lines.append(f"  期望输出: {ex.get('label', '')}")
        lines.append(f"  实际输出: {ex.get('prediction', '')}")
        lines.append("")
    return "\n".join(lines)
