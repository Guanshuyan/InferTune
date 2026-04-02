"""ProTeGi 主流程编排。

将文本梯度生成、prompt 编辑、beam search、bandit 选择等组件串联，
提供简洁的顶层 API 供用户一键优化 prompt。
"""

import random
from typing import Any, Callable

from ..llm.base import BaseLLM
from ..evaluator.base import BaseEvaluator
from ..evaluator.cost_estimator import estimate_protegi
from ..config.settings import ProTeGiConfig
from .beam_search import beam_search


class PromptOptimizer:
    """ProTeGi 提示优化器。

    封装完整的 prompt 优化流程，用户只需提供初始 prompt、训练数据和评估方式。

    参数:
        llm: LLM 实例，用于梯度生成和 prompt 编辑
        config: ProTeGi 配置，为 None 时使用默认值
    """

    def __init__(self, llm: BaseLLM, config: ProTeGiConfig | None = None):
        self.llm = llm
        self.config = config or ProTeGiConfig()

    def optimize(
        self,
        initial_prompt: str,
        train_data: list[dict[str, Any]],
        eval_data: list[dict[str, Any]],
        evaluator: BaseEvaluator,
        *,
        run_fn: Callable[[str, str], str] | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """执行完整的 prompt 优化流程。

        参数:
            initial_prompt: 初始 prompt
            train_data: 训练数据，每个元素为 {"input": str, "label": str}
            eval_data: 评估数据，格式同 train_data，用于 bandit 选择时评分
            evaluator: 评估器实例
            run_fn: 自定义运行函数 (prompt, input_text) -> prediction；
                    为 None 时使用 LLM 的 system+user 消息模式
            verbose: 是否输出中间过程

        返回:
            优化结果字典，包含:
            - "best_prompt": 最优 prompt
            - "initial_score": 初始 prompt 的评估分数
            - "final_score": 最优 prompt 的评估分数
            - "improvement": 分数提升量
            - "history": beam search 搜索历史
            - "cost_estimate": 调用次数估算结果
        """
        cfg = self.config

        # 运行前估算 LLM 调用次数
        cost = estimate_protegi(cfg, len(train_data), len(eval_data))
        if verbose:
            print(f"\n{cost.summary()}\n")

        # 默认运行函数
        if run_fn is None:
            def run_fn(prompt: str, input_text: str) -> str:
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": input_text},
                ]
                return self.llm.chat(messages, temperature=0.0)

        # 构建评估函数：在 eval_data 上运行 prompt 并计算分数
        def eval_fn(prompt: str) -> float:
            return self._evaluate_prompt(prompt, eval_data, evaluator, run_fn)

        # 计算初始分数
        initial_score = eval_fn(initial_prompt)
        if verbose:
            print(f"[ProTeGi] 初始 prompt 分数: {initial_score:.4f}")

        # 执行 beam search
        best_prompt, history = beam_search(
            self.llm,
            initial_prompt,
            train_data,
            eval_fn,
            beam_width=cfg.beam_width,
            search_depth=cfg.search_depth,
            minibatch_size=cfg.minibatch_size,
            num_gradients=cfg.num_gradients,
            num_edits_per_gradient=cfg.num_edits_per_gradient,
            num_monte_carlo=cfg.num_monte_carlo,
            selection_method=cfg.selection_method,
            run_fn=run_fn,
            verbose=verbose,
        )

        # 计算最终分数
        final_score = eval_fn(best_prompt)
        if verbose:
            print(f"[ProTeGi] 最优 prompt 分数: {final_score:.4f}")
            print(f"[ProTeGi] 提升: {final_score - initial_score:+.4f}")

        return {
            "best_prompt": best_prompt,
            "initial_score": initial_score,
            "final_score": final_score,
            "improvement": final_score - initial_score,
            "history": history,
            "cost_estimate": cost,
        }

    def _evaluate_prompt(
        self,
        prompt: str,
        data: list[dict[str, Any]],
        evaluator: BaseEvaluator,
        run_fn: Callable[[str, str], str],
    ) -> float:
        """在数据集上评估 prompt 的表现。

        参数:
            prompt: 待评估的 prompt
            data: 评估数据
            evaluator: 评估器
            run_fn: 运行函数

        返回:
            评估分数
        """
        predictions = []
        labels = []
        for item in data:
            pred = run_fn(prompt, item["input"])
            predictions.append(pred)
            labels.append(item["label"])
        return evaluator.score(predictions, labels)
