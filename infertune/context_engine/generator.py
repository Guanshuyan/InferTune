"""Generator: 生成推理轨迹。

ACE 三角色架构中的第一个角色，负责使用当前 Playbook 上下文执行任务，
生成推理轨迹（trajectory），并标注哪些 Bullet 在过程中有用或有害。
"""

from typing import Any, Callable

from ..llm.base import BaseLLM
from .playbook import Playbook


# Generator 执行任务时的 prompt 模板
_GENERATOR_PROMPT_TEMPLATE = """你是一个任务执行助手。请根据以下策略和知识来完成任务。

{playbook_context}

任务输入：
{input_text}

请完成任务并输出结果。在回答末尾，请用以下格式标注策略的有用性：
[有用策略]: 列出帮助你完成任务的策略编号（逗号分隔），如无则写"无"
[有害策略]: 列出误导你或不相关的策略编号（逗号分隔），如无则写"无"
[回答]: 你的最终回答"""


def generate_trajectory(
    llm: BaseLLM,
    playbook: Playbook,
    input_text: str,
    *,
    prompt_template: str | None = None,
    system_prompt: str = "",
) -> dict[str, Any]:
    """使用当前 Playbook 执行任务，生成推理轨迹。

    参数:
        llm: LLM 实例
        playbook: 当前 Playbook
        input_text: 任务输入文本
        prompt_template: 自定义 prompt 模板，需包含 {playbook_context} 和 {input_text}
        system_prompt: 额外的系统提示（如任务描述）

    返回:
        轨迹字典，包含:
        - "input": 原始输入
        - "output": LLM 输出的完整文本
        - "answer": 提取的最终回答
        - "helpful_ids": 被标记为有用的 Bullet 编号列表
        - "harmful_ids": 被标记为有害的 Bullet 编号列表
    """
    template = prompt_template or _GENERATOR_PROMPT_TEMPLATE
    playbook_context = playbook.render() if len(playbook) > 0 else "（暂无可用策略）"

    content = template.format(
        playbook_context=playbook_context,
        input_text=input_text,
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})

    output = llm.chat(messages, temperature=0.7)

    # 解析输出，提取回答和策略标注
    answer, helpful_ids, harmful_ids = _parse_trajectory_output(output, playbook)

    return {
        "input": input_text,
        "output": output,
        "answer": answer,
        "helpful_ids": helpful_ids,
        "harmful_ids": harmful_ids,
    }


def generate_batch_trajectories(
    llm: BaseLLM,
    playbook: Playbook,
    inputs: list[str],
    *,
    labels: list[str] | None = None,
    system_prompt: str = "",
) -> list[dict[str, Any]]:
    """批量生成推理轨迹。

    参数:
        llm: LLM 实例
        playbook: 当前 Playbook
        inputs: 任务输入列表
        labels: 真实标签列表（可选），用于标注轨迹是否正确
        system_prompt: 额外的系统提示

    返回:
        轨迹字典列表
    """
    trajectories = []
    for i, input_text in enumerate(inputs):
        traj = generate_trajectory(
            llm, playbook, input_text, system_prompt=system_prompt,
        )
        if labels is not None and i < len(labels):
            traj["label"] = labels[i]
            traj["correct"] = _check_correct(traj["answer"], labels[i])
        trajectories.append(traj)
    return trajectories


def _parse_trajectory_output(
    output: str,
    playbook: Playbook,
) -> tuple[str, list[str], list[str]]:
    """解析 Generator 输出，提取回答和策略标注。

    参数:
        output: LLM 的完整输出文本
        playbook: 当前 Playbook（用于将编号映射回 Bullet ID）

    返回:
        (回答文本, 有用 Bullet ID 列表, 有害 Bullet ID 列表)
    """
    answer = output
    helpful_ids = []
    harmful_ids = []

    bullets = playbook.bullets

    # 尝试解析结构化输出
    lines = output.split("\n")
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith("[回答]"):
            answer = line_stripped.replace("[回答]", "").strip().lstrip(":").strip()
        elif line_stripped.startswith("[有用策略]"):
            nums = _extract_numbers(line_stripped)
            helpful_ids = _map_numbers_to_ids(nums, bullets)
        elif line_stripped.startswith("[有害策略]"):
            nums = _extract_numbers(line_stripped)
            harmful_ids = _map_numbers_to_ids(nums, bullets)

    return answer, helpful_ids, harmful_ids


def _extract_numbers(text: str) -> list[int]:
    """从文本中提取数字列表。

    参数:
        text: 包含数字的文本

    返回:
        整数列表
    """
    import re
    return [int(n) for n in re.findall(r"\d+", text)]


def _map_numbers_to_ids(numbers: list[int], bullets: list) -> list[str]:
    """将编号映射为 Bullet ID。

    参数:
        numbers: 编号列表（从 1 开始）
        bullets: 按顺序排列的 Bullet 列表

    返回:
        对应的 Bullet ID 列表
    """
    ids = []
    for n in numbers:
        idx = n - 1  # 编号从 1 开始
        if 0 <= idx < len(bullets):
            ids.append(bullets[idx].id)
    return ids


def _check_correct(answer: str, label: str) -> bool:
    """简单判断回答是否正确。

    参数:
        answer: 模型回答
        label: 真实标签

    返回:
        是否正确
    """
    a = answer.strip().lower()
    l = label.strip().lower()
    return a == l or l in a
