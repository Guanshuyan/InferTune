"""Reflector: 从推理轨迹中提取洞察。

ACE 三角色架构中的第二个角色，负责分析 Generator 产生的轨迹，
从成功和失败中提取可复用的策略洞察，支持多轮迭代精炼。
"""

from typing import Any

from ..llm.base import BaseLLM


# Reflector 分析轨迹的 prompt 模板
_REFLECT_PROMPT_TEMPLATE = """你是一个策略分析专家。请分析以下任务执行轨迹，提取可复用的策略洞察。

任务输入：{input_text}
执行结果：{output}
{correctness_info}

请提取 1-3 条具体、可操作的策略洞察。每条洞察应该是：
- 一个可复用的策略（如果执行成功）
- 一个需要避免的失败模式（如果执行失败）
- 一个领域特定的知识点

请用以下 JSON 格式输出（不要输出其他内容）：
[
  {{"content": "策略描述", "tags": ["标签1", "标签2"]}},
  ...
]"""

# 迭代精炼 prompt 模板
_REFINE_PROMPT_TEMPLATE = """你是一个策略分析专家。请对以下洞察进行精炼和改进。

原始洞察：
{insights_text}

基于的任务轨迹：
输入：{input_text}
输出：{output}

请改进这些洞察，使其更加具体、准确、可操作。
保持 JSON 格式输出：
[
  {{"content": "改进后的策略描述", "tags": ["标签1", "标签2"]}},
  ...
]"""


def reflect_on_trajectory(
    llm: BaseLLM,
    trajectory: dict[str, Any],
    *,
    prompt_template: str | None = None,
) -> list[dict[str, Any]]:
    """分析单条轨迹，提取策略洞察。

    参数:
        llm: LLM 实例
        trajectory: Generator 产生的轨迹字典，包含 input/output/correct 等字段
        prompt_template: 自定义分析 prompt 模板

    返回:
        洞察列表，每个元素为 {"content": str, "tags": list[str]}
    """
    template = prompt_template or _REFLECT_PROMPT_TEMPLATE

    # 构建正确性信息
    correctness_info = ""
    if "correct" in trajectory:
        if trajectory["correct"]:
            correctness_info = "执行状态：成功（回答正确）"
        else:
            correctness_info = f"执行状态：失败（期望: {trajectory.get('label', '未知')}，实际: {trajectory.get('answer', '未知')}）"

    content = template.format(
        input_text=trajectory.get("input", ""),
        output=trajectory.get("output", ""),
        correctness_info=correctness_info,
    )

    messages = [{"role": "user", "content": content}]
    response = llm.chat(messages, temperature=0.7)

    return _parse_insights(response)


def reflect_on_batch(
    llm: BaseLLM,
    trajectories: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """批量分析轨迹，汇总提取洞察。

    参数:
        llm: LLM 实例
        trajectories: 轨迹列表

    返回:
        所有洞察的合并列表
    """
    all_insights = []
    for traj in trajectories:
        insights = reflect_on_trajectory(llm, traj)
        all_insights.extend(insights)
    return all_insights


def refine_insights(
    llm: BaseLLM,
    insights: list[dict[str, Any]],
    trajectory: dict[str, Any],
    *,
    max_rounds: int = 3,
    refine_template: str | None = None,
) -> list[dict[str, Any]]:
    """对洞察进行多轮迭代精炼。

    对应 ACE 论文中 Reflector 的迭代精炼机制，
    每轮让 LLM 基于轨迹上下文改进洞察的质量。

    参数:
        llm: LLM 实例
        insights: 初始洞察列表
        trajectory: 关联的轨迹（提供上下文）
        max_rounds: 最大精炼轮数
        refine_template: 自定义精炼 prompt 模板

    返回:
        精炼后的洞察列表
    """
    template = refine_template or _REFINE_PROMPT_TEMPLATE
    current_insights = insights

    for _ in range(max_rounds):
        insights_text = "\n".join(
            f"- {ins.get('content', '')}" for ins in current_insights
        )

        content = template.format(
            insights_text=insights_text,
            input_text=trajectory.get("input", ""),
            output=trajectory.get("output", ""),
        )

        messages = [{"role": "user", "content": content}]
        response = llm.chat(messages, temperature=0.5)
        refined = _parse_insights(response)

        if refined:
            current_insights = refined

    return current_insights


def _parse_insights(response: str) -> list[dict[str, Any]]:
    """从 LLM 响应中解析洞察 JSON。

    参数:
        response: LLM 的文本响应

    返回:
        洞察列表，解析失败时返回空列表
    """
    import json
    import re

    # 尝试提取 JSON 数组
    # 先尝试直接解析
    text = response.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return _validate_insights(result)
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown 代码块中提取
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1).strip())
            if isinstance(result, list):
                return _validate_insights(result)
        except json.JSONDecodeError:
            pass

    # 尝试找到第一个 [ 和最后一个 ] 之间的内容
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, list):
                return _validate_insights(result)
        except json.JSONDecodeError:
            pass

    return []


def _validate_insights(items: list) -> list[dict[str, Any]]:
    """校验并规范化洞察列表。

    参数:
        items: 原始解析结果

    返回:
        规范化后的洞察列表
    """
    valid = []
    for item in items:
        if isinstance(item, dict) and "content" in item:
            valid.append({
                "content": str(item["content"]),
                "tags": item.get("tags", []),
            })
    return valid
