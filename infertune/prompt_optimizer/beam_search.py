"""Beam Search 搜索框架。

实现 ProTeGi 论文的外层循环：在 prompt 空间上进行 beam search，
每步通过梯度生成 + 编辑扩展候选，再用 bandit 算法选出最优子集进入下一轮。
"""

import random
from typing import Any, Callable

from ..llm.base import BaseLLM
from .gradient import generate_gradients
from .editor import expand_prompt
from .bandit import select_by_ucb, select_by_successive_rejects


def beam_search(
    llm: BaseLLM,
    initial_prompt: str,
    train_data: list[dict[str, Any]],
    eval_fn: Callable[[str], float],
    *,
    beam_width: int = 4,
    search_depth: int = 6,
    minibatch_size: int = 64,
    num_gradients: int = 4,
    errors_per_group: int = 4,
    num_edits_per_gradient: int = 1,
    num_monte_carlo: int = 2,
    selection_method: str = "ucb",
    selection_budget: int = 50,
    max_candidates_per_parent: int = 8,
    run_fn: Callable[[str, str], str] | None = None,
    verbose: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """在 prompt 空间上执行 beam search 优化。

    对应 ProTeGi 论文 Algorithm 1 的完整流程。

    参数:
        llm: LLM 实例
        initial_prompt: 初始 prompt
        train_data: 训练数据列表，每个元素为 {"input": str, "label": str}
        eval_fn: 评估函数，输入 prompt 文本，返回 0~1 分数
        beam_width: beam 宽度（每轮保留的候选数）
        search_depth: 搜索深度（优化步数）
        minibatch_size: 每步采样的 minibatch 大小
        num_gradients: 每组错误生成的梯度数量
        errors_per_group: 每个梯度使用的错误样本数
        num_edits_per_gradient: 每个梯度生成的编辑候选数
        num_monte_carlo: 每个候选的 paraphrase 扩展数
        selection_method: 候选选择算法，"ucb" 或 "successive_rejects"
        selection_budget: bandit 选择的评估预算
        max_candidates_per_parent: 每个父 prompt 最多保留的候选数（避免计算爆炸）
        run_fn: 运行函数，输入 (prompt, input_text)，返回预测结果；
                默认使用 llm.chat 构造 system+user 消息
        verbose: 是否输出中间过程信息

    返回:
        (最优 prompt, 搜索历史记录列表)
    """
    # 默认运行函数：将 prompt 作为 system message
    if run_fn is None:
        def run_fn(prompt: str, input_text: str) -> str:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_text},
            ]
            return llm.chat(messages, temperature=0.0)

    beam = [initial_prompt]
    history = []

    for step in range(search_depth):
        if verbose:
            print(f"[BeamSearch] 第 {step + 1}/{search_depth} 步，当前 beam 大小: {len(beam)}")

        all_candidates = []

        for parent_prompt in beam:
            # 1. 采样 minibatch 并收集错误
            minibatch = _sample_minibatch(train_data, minibatch_size)
            errors = _collect_errors(run_fn, parent_prompt, minibatch)

            if not errors:
                # 没有错误，保留当前 prompt
                all_candidates.append(parent_prompt)
                continue

            # 2. 生成文本梯度
            gradients = generate_gradients(
                llm, parent_prompt, errors,
                num_gradients=num_gradients,
                errors_per_group=errors_per_group,
            )

            # 3. 根据梯度扩展候选
            expanded = expand_prompt(
                llm, parent_prompt, gradients,
                num_edits_per_gradient=num_edits_per_gradient,
                num_monte_carlo=num_monte_carlo,
            )

            # 限制每个父 prompt 的候选数
            if len(expanded) > max_candidates_per_parent:
                expanded = random.sample(expanded, max_candidates_per_parent)

            all_candidates.extend(expanded)
            # 保留父 prompt 作为候选
            all_candidates.append(parent_prompt)

        # 4. 去重
        all_candidates = list(set(all_candidates))

        if verbose:
            print(f"[BeamSearch] 候选数: {len(all_candidates)}，开始选择...")

        # 5. Bandit 选择 top beam_width 个候选
        if selection_method == "successive_rejects":
            selected = select_by_successive_rejects(
                all_candidates, eval_fn,
                budget=selection_budget, top_k=beam_width,
            )
        else:
            selected = select_by_ucb(
                all_candidates, eval_fn,
                budget=selection_budget, top_k=beam_width,
                exploration_c=2.0,
            )

        beam = [prompt for prompt, _ in selected]

        # 记录历史
        step_record = {
            "step": step + 1,
            "num_candidates": len(all_candidates),
            "selected": selected,
            "best_prompt": selected[0][0] if selected else "",
            "best_score": selected[0][1] if selected else 0.0,
        }
        history.append(step_record)

        if verbose:
            print(f"[BeamSearch] 最优分数: {step_record['best_score']:.4f}")

    # 返回最终 beam 中得分最高的 prompt
    best_prompt = beam[0] if beam else initial_prompt
    return best_prompt, history


def _sample_minibatch(
    data: list[dict[str, Any]],
    size: int,
) -> list[dict[str, Any]]:
    """从训练数据中随机采样 minibatch。

    参数:
        data: 完整训练数据
        size: 采样大小

    返回:
        采样后的子集
    """
    if len(data) <= size:
        return data
    return random.sample(data, size)


def _collect_errors(
    run_fn: Callable[[str, str], str],
    prompt: str,
    data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """用当前 prompt 在数据上运行，收集预测错误的样本。

    参数:
        run_fn: 运行函数 (prompt, input) -> prediction
        prompt: 当前 prompt
        data: 数据样本列表

    返回:
        错误样本列表，每个元素包含 input/label/prediction
    """
    errors = []
    for item in data:
        prediction = run_fn(prompt, item["input"])
        # 简单的字符串匹配判断是否正确
        label_norm = item["label"].strip().lower()
        pred_norm = prediction.strip().lower()
        if label_norm not in pred_norm and pred_norm != label_norm:
            errors.append({
                "input": item["input"],
                "label": item["label"],
                "prediction": prediction,
            })
    return errors
