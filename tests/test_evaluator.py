"""评估模块单元测试。

测试 Accuracy、F1、ExactMatch 评估器及 odds_ratio_score 函数。
"""

import math

from infertune.evaluator.metrics import (
    AccuracyEvaluator,
    F1Evaluator,
    ExactMatchEvaluator,
    odds_ratio_score,
)
from infertune.evaluator.base import FunctionEvaluator
from infertune.evaluator.comparator import ResponseComparator


class TestAccuracyEvaluator:
    """准确率评估器测试。"""

    def test_perfect_score(self):
        ev = AccuracyEvaluator()
        assert ev.score(["yes", "no"], ["yes", "no"]) == 1.0

    def test_zero_score(self):
        ev = AccuracyEvaluator()
        assert ev.score(["yes", "yes"], ["no", "no"]) == 0.0

    def test_normalize(self):
        ev = AccuracyEvaluator(normalize=True)
        assert ev.score(["YES ", " No"], ["yes", "no"]) == 1.0

    def test_empty(self):
        ev = AccuracyEvaluator()
        assert ev.score([], []) == 0.0


class TestF1Evaluator:
    """F1 评估器测试。"""

    def test_perfect_f1(self):
        ev = F1Evaluator(positive_label="yes")
        assert ev.score(["yes", "no"], ["yes", "no"]) == 1.0

    def test_all_false_positive(self):
        ev = F1Evaluator(positive_label="yes")
        assert ev.score(["yes", "yes"], ["no", "no"]) == 0.0

    def test_partial(self):
        ev = F1Evaluator(positive_label="yes")
        # tp=1, fp=1, fn=0 → precision=0.5, recall=1.0, f1=2/3
        score = ev.score(["yes", "yes", "no"], ["yes", "no", "no"])
        assert abs(score - 2 / 3) < 1e-6


class TestExactMatchEvaluator:
    """精确匹配评估器测试。"""

    def test_exact_mode(self):
        ev = ExactMatchEvaluator(mode="exact")
        assert ev.score(["hello"], ["hello"]) == 1.0
        assert ev.score(["hello world"], ["hello"]) == 0.0

    def test_contains_mode(self):
        ev = ExactMatchEvaluator(mode="contains")
        assert ev.score(["the answer is yes"], ["yes"]) == 1.0
        assert ev.score(["no way"], ["yes"]) == 0.0


class TestOddsRatioScore:
    """odds_ratio_score 函数测试。"""

    def test_preferred_higher(self):
        # 偏好概率更高时，分数应为正
        score = odds_ratio_score(0.8, 0.2)
        assert score > 0

    def test_equal_probs(self):
        # 概率相等时，分数应接近 0
        score = odds_ratio_score(0.5, 0.5)
        assert abs(score) < 1e-6

    def test_rejected_higher(self):
        # 拒绝概率更高时，分数应为负
        score = odds_ratio_score(0.2, 0.8)
        assert score < 0


class TestFunctionEvaluator:
    """自定义函数评估器测试。"""

    def test_custom_fn(self):
        fn = lambda preds, labels: sum(p == l for p, l in zip(preds, labels)) / len(preds)
        ev = FunctionEvaluator(fn)
        assert ev.score(["a", "b", "c"], ["a", "b", "x"]) == 2 / 3


class TestResponseComparator:
    """响应对比器测试。"""

    def test_rank_by_evaluator(self):
        ev = AccuracyEvaluator()
        comp = ResponseComparator()
        ranked = comp.rank_by_evaluator(["yes", "no", "maybe"], "yes", ev)
        # "yes" 应排第一
        assert ranked[0][0] == 0
        assert ranked[0][1] == 1.0

    def test_best_response(self):
        ev = AccuracyEvaluator()
        comp = ResponseComparator()
        best, score = comp.best_response(["no", "yes", "maybe"], "yes", ev)
        assert best == "yes"
        assert score == 1.0
