"""响应对比与排序模块。

借鉴 ORPO 的 odds ratio 思想，在推理时对多个候选响应进行质量评估和排序。
支持基于 LLM 打分和基于评估指标两种对比方式。
"""

from typing import Any

from ..llm.base import BaseLLM
from .metrics import odds_ratio_score


# LLM 评分时使用的默认 prompt 模板
_SCORE_PROMPT_TEMPLATE = """请对以下回答的质量进行评分（1-10分）。
评分标准：准确性、完整性、清晰度、相关性。

问题：{question}

回答：{response}

请只输出一个整数分数（1-10），不要输出其他内容。"""


class ResponseComparator:
    """响应对比器，用于在多个候选响应中选择最优。

    参数:
        llm: LLM 实例，用于 LLM-as-judge 评分
    """

    def __init__(self, llm: BaseLLM | None = None):
        self._llm = llm

    def rank_by_evaluator(
        self,
        responses: list[str],
        label: str,
        evaluator: Any,
    ) -> list[tuple[int, float]]:
        """基于评估指标对候选响应排序。

        参数:
            responses: 候选响应列表
            label: 真实标签
            evaluator: 评估器实例（需实现 score_single 方法）

        返回:
            按分数降序排列的 (索引, 分数) 列表
        """
        scored = []
        for i, resp in enumerate(responses):
            s = evaluator.score_single(resp, label)
            scored.append((i, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def rank_by_llm(
        self,
        question: str,
        responses: list[str],
        prompt_template: str | None = None,
    ) -> list[tuple[int, float]]:
        """基于 LLM-as-judge 对候选响应评分排序。

        参数:
            question: 原始问题
            responses: 候选响应列表
            prompt_template: 评分 prompt 模板，需包含 {question} 和 {response} 占位符

        返回:
            按分数降序排列的 (索引, 分数) 列表

        异常:
            ValueError: 未设置 LLM 实例时调用
        """
        if self._llm is None:
            raise ValueError("rank_by_llm 需要提供 LLM 实例")

        template = prompt_template or _SCORE_PROMPT_TEMPLATE
        scored = []

        for i, resp in enumerate(responses):
            prompt = template.format(question=question, response=resp)
            messages = [{"role": "user", "content": prompt}]
            result = self._llm.chat(messages, temperature=0.0)
            # 从回复中提取数字分数
            try:
                s = float(result.strip())
            except ValueError:
                # 提取第一个数字
                import re
                nums = re.findall(r"\d+(?:\.\d+)?", result)
                s = float(nums[0]) if nums else 0.0
            scored.append((i, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def best_response(
        self,
        responses: list[str],
        label: str,
        evaluator: Any,
    ) -> tuple[str, float]:
        """选出最优候选响应。

        参数:
            responses: 候选响应列表
            label: 真实标签
            evaluator: 评估器实例

        返回:
            (最优响应文本, 对应分数)
        """
        ranked = self.rank_by_evaluator(responses, label, evaluator)
        best_idx, best_score = ranked[0]
        return responses[best_idx], best_score
