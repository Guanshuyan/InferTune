"""Prompt 编辑器。

基于 ProTeGi 论文的 δ prompt：根据文本梯度（缺陷描述）反向编辑当前 prompt，
生成改进后的候选 prompt，并通过 paraphrase 扩展候选空间。
"""

from ..llm.base import BaseLLM


# 编辑 prompt 模板（对应论文中的 δ prompt）
_EDIT_PROMPT_TEMPLATE = """请根据以下分析改进当前 prompt。

当前 prompt：
{current_prompt}

分析发现的问题：
{gradient}

请直接输出改进后的完整 prompt，不要输出其他解释内容。"""

# Paraphrase 模板，用于扩展候选空间
_PARAPHRASE_PROMPT_TEMPLATE = """请用不同的措辞重写以下 prompt，保持语义不变但表达方式不同。

原始 prompt：
{prompt}

请直接输出重写后的完整 prompt，不要输出其他内容。"""


def edit_prompt(
    llm: BaseLLM,
    current_prompt: str,
    gradient: str,
    *,
    edit_template: str | None = None,
) -> str:
    """根据文本梯度编辑 prompt。

    对应论文中的"反向传播"步骤：根据梯度指出的缺陷方向，
    编辑当前 prompt 以修复问题。

    参数:
        llm: LLM 实例
        current_prompt: 当前待编辑的 prompt
        gradient: 文本梯度（缺陷描述）
        edit_template: 自定义编辑 prompt 模板

    返回:
        编辑后的新 prompt
    """
    template = edit_template or _EDIT_PROMPT_TEMPLATE
    prompt = template.format(current_prompt=current_prompt, gradient=gradient)
    messages = [{"role": "user", "content": prompt}]
    return llm.chat(messages, temperature=1.0).strip()


def paraphrase_prompt(
    llm: BaseLLM,
    prompt_text: str,
    *,
    num_paraphrases: int = 2,
    paraphrase_template: str | None = None,
) -> list[str]:
    """对 prompt 进行 paraphrase 扩展，生成语义等价但表达不同的变体。

    对应论文中的 Monte Carlo 采样步骤，用于扩展候选空间。

    参数:
        llm: LLM 实例
        prompt_text: 待 paraphrase 的 prompt
        num_paraphrases: 生成的 paraphrase 数量
        paraphrase_template: 自定义 paraphrase 模板

    返回:
        paraphrase 后的 prompt 列表
    """
    template = paraphrase_template or _PARAPHRASE_PROMPT_TEMPLATE
    content = template.format(prompt=prompt_text)
    messages = [{"role": "user", "content": content}]

    results = []
    for _ in range(num_paraphrases):
        result = llm.chat(messages, temperature=1.0).strip()
        results.append(result)
    return results


def expand_prompt(
    llm: BaseLLM,
    current_prompt: str,
    gradients: list[str],
    *,
    num_edits_per_gradient: int = 1,
    num_monte_carlo: int = 2,
) -> list[str]:
    """从梯度列表生成完整的候选 prompt 集合。

    流程：每个梯度 → 编辑生成候选 → paraphrase 扩展 → 汇总去重。

    参数:
        llm: LLM 实例
        current_prompt: 当前 prompt
        gradients: 文本梯度列表
        num_edits_per_gradient: 每个梯度生成的编辑候选数
        num_monte_carlo: 每个候选的 paraphrase 扩展数

    返回:
        所有候选 prompt 列表（已去重）
    """
    candidates = set()

    for gradient in gradients:
        for _ in range(num_edits_per_gradient):
            edited = edit_prompt(llm, current_prompt, gradient)
            candidates.add(edited)

            # Paraphrase 扩展
            paraphrases = paraphrase_prompt(llm, edited, num_paraphrases=num_monte_carlo)
            candidates.update(paraphrases)

    return list(candidates)
