"""通用评估指标集合。

提供 Accuracy、F1、ExactMatch 等常用指标实现，
以及基于 ORPO odds ratio 思想的响应质量评分。
"""

import math
from collections import Counter

from .base import BaseEvaluator


class AccuracyEvaluator(BaseEvaluator):
    """准确率评估器，计算预测与标签完全匹配的比例。

    参数:
        normalize: 是否对预测和标签做小写 + strip 归一化
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def score(self, predictions: list[str], labels: list[str]) -> float:
        """计算准确率。

        参数:
            predictions: 模型预测结果列表
            labels: 真实标签列表

        返回:
            准确率，0~1 之间
        """
        if not predictions:
            return 0.0
        correct = 0
        for pred, label in zip(predictions, labels):
            p = pred.strip().lower() if self.normalize else pred
            l = label.strip().lower() if self.normalize else label
            if p == l:
                correct += 1
        return correct / len(predictions)


class F1Evaluator(BaseEvaluator):
    """二分类 F1 评估器。

    参数:
        positive_label: 正类标签文本
        normalize: 是否对预测和标签做小写 + strip 归一化
    """

    def __init__(self, positive_label: str = "yes", normalize: bool = True):
        self.positive_label = positive_label.strip().lower() if normalize else positive_label
        self.normalize = normalize

    def score(self, predictions: list[str], labels: list[str]) -> float:
        """计算 F1 分数。

        参数:
            predictions: 模型预测结果列表
            labels: 真实标签列表

        返回:
            F1 分数，0~1 之间
        """
        if not predictions:
            return 0.0

        tp = fp = fn = 0
        for pred, label in zip(predictions, labels):
            p = pred.strip().lower() if self.normalize else pred
            l = label.strip().lower() if self.normalize else label
            if p == self.positive_label and l == self.positive_label:
                tp += 1
            elif p == self.positive_label and l != self.positive_label:
                fp += 1
            elif p != self.positive_label and l == self.positive_label:
                fn += 1

        if tp == 0:
            return 0.0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * precision * recall / (precision + recall)


class ExactMatchEvaluator(BaseEvaluator):
    """精确匹配评估器，判断预测文本是否包含标签内容。

    参数:
        mode: 匹配模式，"exact" 完全匹配 / "contains" 包含匹配
        normalize: 是否做小写 + strip 归一化
    """

    def __init__(self, mode: str = "exact", normalize: bool = True):
        self.mode = mode
        self.normalize = normalize

    def score(self, predictions: list[str], labels: list[str]) -> float:
        """计算匹配率。

        参数:
            predictions: 模型预测结果列表
            labels: 真实标签列表

        返回:
            匹配率，0~1 之间
        """
        if not predictions:
            return 0.0
        correct = 0
        for pred, label in zip(predictions, labels):
            p = pred.strip().lower() if self.normalize else pred
            l = label.strip().lower() if self.normalize else label
            if self.mode == "contains":
                if l in p:
                    correct += 1
            else:
                if p == l:
                    correct += 1
        return correct / len(predictions)


def odds_ratio_score(prob_preferred: float, prob_rejected: float) -> float:
    """计算 odds ratio 分数，借鉴 ORPO 论文思想。

    用于在推理时对比两个候选响应的质量差异。
    odds(y|x) = P(y|x) / (1 - P(y|x))
    OR = odds(preferred) / odds(rejected)

    参数:
        prob_preferred: 偏好响应的生成概率（0~1）
        prob_rejected: 拒绝响应的生成概率（0~1）

    返回:
        log odds ratio 分数，正值表示偏好响应更优
    """
    # 裁剪概率避免除零
    eps = 1e-10
    p_w = max(min(prob_preferred, 1.0 - eps), eps)
    p_l = max(min(prob_rejected, 1.0 - eps), eps)

    odds_w = p_w / (1.0 - p_w)
    odds_l = p_l / (1.0 - p_l)

    return math.log(odds_w / odds_l)
