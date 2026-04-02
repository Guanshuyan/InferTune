"""评估器基类，定义统一的评估接口。

所有评估指标需要继承 BaseEvaluator 并实现 score 方法。
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseEvaluator(ABC):
    """评估器抽象基类。

    子类需实现 score 方法，接收预测值和标签，返回 0~1 之间的分数。
    """

    @abstractmethod
    def score(self, predictions: list[str], labels: list[str]) -> float:
        """计算评估分数。

        参数:
            predictions: 模型预测结果列表
            labels: 真实标签列表

        返回:
            0~1 之间的评估分数
        """

    def score_single(self, prediction: str, label: str) -> float:
        """评估单条样本，默认委托给 score 方法。

        参数:
            prediction: 单条模型预测
            label: 单条真实标签

        返回:
            0~1 之间的评估分数
        """
        return self.score([prediction], [label])


class FunctionEvaluator(BaseEvaluator):
    """基于自定义函数的评估器，方便快速接入任意指标。

    参数:
        fn: 评估函数，签名为 (predictions, labels) -> float
    """

    def __init__(self, fn: Any):
        self._fn = fn

    def score(self, predictions: list[str], labels: list[str]) -> float:
        """调用自定义函数计算分数。

        参数:
            predictions: 模型预测结果列表
            labels: 真实标签列表

        返回:
            评估分数
        """
        return self._fn(predictions, labels)
