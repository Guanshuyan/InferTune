"""Bandit 候选选择算法。

实现 ProTeGi 论文中的候选 prompt 选择策略：
- UCB (Upper Confidence Bound): 平衡探索与利用
- Successive Rejects: 逐轮淘汰最差候选，理论最优的 best-arm identification 算法
"""

import math
import random
from typing import Any, Callable


def select_by_ucb(
    candidates: list[str],
    eval_fn: Callable[[str], float],
    *,
    budget: int = 50,
    top_k: int = 4,
    exploration_c: float = 2.0,
) -> list[tuple[str, float]]:
    """使用 UCB 算法从候选中选出 top_k 个最优 prompt。

    每轮选择置信上界最高的候选进行评估，逐步收敛到最优。

    参数:
        candidates: 候选 prompt 列表
        eval_fn: 评估函数，输入 prompt 文本，返回 0~1 分数
        budget: 总评估预算（调用 eval_fn 的次数上限）
        top_k: 最终选出的候选数量
        exploration_c: 探索参数，越大越倾向探索未充分评估的候选

    返回:
        按分数降序排列的 (prompt, 平均分数) 列表，长度为 top_k
    """
    n = len(candidates)
    if n == 0:
        return []
    if n <= top_k:
        scores = [(c, eval_fn(c)) for c in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    # 初始化：每个候选至少评估一次
    counts = [0] * n
    totals = [0.0] * n

    for i in range(min(n, budget)):
        score = eval_fn(candidates[i])
        counts[i] = 1
        totals[i] = score

    remaining_budget = budget - min(n, budget)

    # UCB 迭代
    for t in range(remaining_budget):
        step = n + t + 1
        best_idx = -1
        best_ucb = -float("inf")

        for i in range(n):
            if counts[i] == 0:
                ucb_val = float("inf")
            else:
                avg = totals[i] / counts[i]
                ucb_val = avg + exploration_c * math.sqrt(math.log(step) / counts[i])

            if ucb_val > best_ucb:
                best_ucb = ucb_val
                best_idx = i

        score = eval_fn(candidates[best_idx])
        counts[best_idx] += 1
        totals[best_idx] += score

    # 按平均分排序，选出 top_k
    avg_scores = []
    for i in range(n):
        avg = totals[i] / counts[i] if counts[i] > 0 else 0.0
        avg_scores.append((candidates[i], avg))

    avg_scores.sort(key=lambda x: x[1], reverse=True)
    return avg_scores[:top_k]


def select_by_successive_rejects(
    candidates: list[str],
    eval_fn: Callable[[str], float],
    *,
    budget: int = 50,
    top_k: int = 4,
) -> list[tuple[str, float]]:
    """使用 Successive Rejects 算法从候选中选出 top_k 个最优 prompt。

    逐轮淘汰得分最低的候选，每轮分配的评估预算逐步增加。
    该算法是 best-arm identification 问题的理论最优解，无需超参数调优。

    参数:
        candidates: 候选 prompt 列表
        eval_fn: 评估函数，输入 prompt 文本，返回 0~1 分数
        budget: 总评估预算
        top_k: 最终选出的候选数量

    返回:
        按分数降序排列的 (prompt, 平均分数) 列表，长度为 top_k
    """
    n = len(candidates)
    if n == 0:
        return []
    if n <= top_k:
        scores = [(c, eval_fn(c)) for c in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    # 存活候选索引集合
    surviving = list(range(n))
    counts = [0] * n
    totals = [0.0] * n

    num_phases = n - top_k  # 需要淘汰的轮数

    for phase in range(num_phases):
        # 计算本轮每个候选的评估次数（论文公式 1）
        n_k = _compute_phase_budget(budget, n, num_phases, phase)

        for idx in surviving:
            # 补充评估到 n_k 次
            while counts[idx] < n_k:
                score = eval_fn(candidates[idx])
                counts[idx] += 1
                totals[idx] += score

        # 淘汰本轮得分最低的候选
        worst_idx = min(surviving, key=lambda i: totals[i] / max(counts[i], 1))
        surviving.remove(worst_idx)

    # 返回存活候选
    result = []
    for idx in surviving:
        avg = totals[idx] / max(counts[idx], 1)
        result.append((candidates[idx], avg))

    result.sort(key=lambda x: x[1], reverse=True)
    return result


def _compute_phase_budget(total_budget: int, n: int, num_phases: int, phase: int) -> int:
    """计算 Successive Rejects 算法中每个 phase 的累计评估次数。

    对应论文公式：n_k = ceil(1 / (0.5 + sum(1/i for i=2..T)) * (B - T) / (T + 1 - k))

    参数:
        total_budget: 总评估预算
        n: 候选总数
        num_phases: 总淘汰轮数
        phase: 当前轮次（从 0 开始）

    返回:
        本轮每个候选应达到的累计评估次数
    """
    T = n
    log_bar = 0.5 + sum(1.0 / i for i in range(2, T + 1))
    if log_bar == 0:
        return 1
    n_k = math.ceil((1.0 / log_bar) * (total_budget - T) / (T + 1 - (phase + 1)))
    return max(n_k, 1)
