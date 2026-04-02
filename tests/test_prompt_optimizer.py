"""ProTeGi 提示优化模块单元测试。

使用 Mock LLM 测试各组件的逻辑正确性，不依赖真实 API 调用。
"""

from unittest.mock import MagicMock, patch

from infertune.llm.base import BaseLLM, ChatCompletionResponse
from infertune.evaluator.metrics import AccuracyEvaluator
from infertune.prompt_optimizer.gradient import (
    generate_gradients,
    _split_into_groups,
    _format_error_examples,
)
from infertune.prompt_optimizer.editor import edit_prompt, paraphrase_prompt, expand_prompt
from infertune.prompt_optimizer.bandit import select_by_ucb, select_by_successive_rejects
from infertune.prompt_optimizer.beam_search import _sample_minibatch, _collect_errors
from infertune.prompt_optimizer.optimizer import PromptOptimizer
from infertune.config.settings import ProTeGiConfig


class _MockLLM(BaseLLM):
    """测试用 Mock LLM，返回可控的固定响应。"""

    def __init__(self, responses: list[str] | None = None):
        super().__init__(model="mock")
        self._responses = responses or ["mock response"]
        self._call_count = 0

    def chat_completion(self, messages, *, model=None, temperature=None, max_tokens=None, **kwargs):
        resp_text = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return ChatCompletionResponse(content=resp_text, model="mock")


# ---- gradient.py 测试 ----

class TestSplitIntoGroups:
    """_split_into_groups 分组逻辑测试。"""

    def test_basic_split(self):
        items = [1, 2, 3, 4, 5, 6, 7, 8]
        groups = _split_into_groups(items, group_size=4, num_groups=2)
        assert len(groups) == 2
        assert len(groups[0]) == 4
        assert len(groups[1]) == 4

    def test_cyclic_reuse(self):
        """样本不足时应循环复用。"""
        items = [1, 2]
        groups = _split_into_groups(items, group_size=4, num_groups=1)
        assert len(groups) == 1
        assert len(groups[0]) == 4

    def test_empty_items(self):
        groups = _split_into_groups([], group_size=4, num_groups=2)
        assert groups == []


class TestFormatErrorExamples:
    """_format_error_examples 格式化测试。"""

    def test_format(self):
        examples = [
            {"input": "hello", "label": "greeting", "prediction": "farewell"},
        ]
        text = _format_error_examples(examples)
        assert "hello" in text
        assert "greeting" in text
        assert "farewell" in text


class TestGenerateGradients:
    """generate_gradients 梯度生成测试。"""

    def test_generates_correct_count(self):
        llm = _MockLLM(["这个 prompt 缺少具体的分类标准"])
        errors = [
            {"input": "text1", "label": "yes", "prediction": "no"},
            {"input": "text2", "label": "no", "prediction": "yes"},
        ]
        gradients = generate_gradients(llm, "classify this", errors, num_gradients=3)
        assert len(gradients) == 3

    def test_empty_errors(self):
        llm = _MockLLM()
        gradients = generate_gradients(llm, "test", [], num_gradients=2)
        assert gradients == []


# ---- editor.py 测试 ----

class TestEditPrompt:
    """edit_prompt 编辑测试。"""

    def test_returns_string(self):
        llm = _MockLLM(["improved prompt text"])
        result = edit_prompt(llm, "old prompt", "needs more detail")
        assert result == "improved prompt text"


class TestParaphrasePrompt:
    """paraphrase_prompt 测试。"""

    def test_returns_correct_count(self):
        llm = _MockLLM(["paraphrase A", "paraphrase B"])
        results = paraphrase_prompt(llm, "some prompt", num_paraphrases=2)
        assert len(results) == 2


class TestExpandPrompt:
    """expand_prompt 候选扩展测试。"""

    def test_generates_candidates(self):
        # 每次调用返回不同文本，确保去重后仍有候选
        llm = _MockLLM(["edited v1", "para v1", "para v2", "edited v2", "para v3", "para v4"])
        candidates = expand_prompt(
            llm, "original", ["gradient1", "gradient2"],
            num_edits_per_gradient=1, num_monte_carlo=1,
        )
        assert len(candidates) > 0


# ---- bandit.py 测试 ----

class TestSelectByUCB:
    """UCB 选择算法测试。"""

    def test_selects_best(self):
        """固定分数的候选，应选出分数最高的。"""
        scores = {"good": 0.9, "medium": 0.5, "bad": 0.1}
        result = select_by_ucb(
            list(scores.keys()),
            lambda c: scores[c],
            budget=30, top_k=1,
        )
        assert len(result) == 1
        assert result[0][0] == "good"

    def test_top_k(self):
        scores = {"a": 0.9, "b": 0.7, "c": 0.5, "d": 0.3}
        result = select_by_ucb(
            list(scores.keys()),
            lambda c: scores[c],
            budget=40, top_k=2,
        )
        assert len(result) == 2

    def test_empty_candidates(self):
        result = select_by_ucb([], lambda c: 0.5, budget=10, top_k=2)
        assert result == []

    def test_fewer_than_top_k(self):
        result = select_by_ucb(["only_one"], lambda c: 0.8, budget=10, top_k=3)
        assert len(result) == 1


class TestSelectBySuccessiveRejects:
    """Successive Rejects 选择算法测试。"""

    def test_selects_best(self):
        scores = {"good": 0.9, "medium": 0.5, "bad": 0.1}
        result = select_by_successive_rejects(
            list(scores.keys()),
            lambda c: scores[c],
            budget=30, top_k=1,
        )
        assert len(result) == 1
        assert result[0][0] == "good"

    def test_empty_candidates(self):
        result = select_by_successive_rejects([], lambda c: 0.5, budget=10, top_k=2)
        assert result == []


# ---- beam_search.py 辅助函数测试 ----

class TestSampleMinibatch:
    """_sample_minibatch 采样测试。"""

    def test_full_data(self):
        data = [{"input": str(i)} for i in range(5)]
        batch = _sample_minibatch(data, 10)
        assert len(batch) == 5

    def test_subsample(self):
        data = [{"input": str(i)} for i in range(100)]
        batch = _sample_minibatch(data, 10)
        assert len(batch) == 10


class TestCollectErrors:
    """_collect_errors 错误收集测试。"""

    def test_collects_wrong_predictions(self):
        def run_fn(prompt, input_text):
            return "wrong" if input_text == "q1" else "correct"

        data = [
            {"input": "q1", "label": "correct"},
            {"input": "q2", "label": "correct"},
        ]
        errors = _collect_errors(run_fn, "test prompt", data)
        assert len(errors) == 1
        assert errors[0]["input"] == "q1"

    def test_no_errors(self):
        def run_fn(prompt, input_text):
            return "yes"

        data = [{"input": "q1", "label": "yes"}]
        errors = _collect_errors(run_fn, "test", data)
        assert len(errors) == 0


# ---- optimizer.py 集成测试 ----

class TestPromptOptimizer:
    """PromptOptimizer 集成测试（使用 Mock LLM）。"""

    def test_optimize_returns_result(self):
        """验证优化流程能正常运行并返回结果结构。"""
        # Mock LLM 交替返回梯度、编辑结果、paraphrase
        llm = _MockLLM([
            "prompt 缺少分类标准",  # 梯度
            "请判断以下文本是否为正面情感",  # 编辑
            "判断下面的文本情感倾向",  # paraphrase
        ])

        config = ProTeGiConfig(
            beam_width=2,
            search_depth=1,
            minibatch_size=4,
            num_gradients=1,
            num_edits_per_gradient=1,
            num_monte_carlo=1,
            selection_method="ucb",
        )

        optimizer = PromptOptimizer(llm, config)

        train_data = [
            {"input": "great movie", "label": "positive"},
            {"input": "terrible film", "label": "negative"},
        ]
        eval_data = train_data

        # 自定义 run_fn 返回固定结果
        def mock_run(prompt, input_text):
            return "positive"

        evaluator = AccuracyEvaluator()
        result = optimizer.optimize(
            "classify sentiment",
            train_data, eval_data, evaluator,
            run_fn=mock_run,
        )

        assert "best_prompt" in result
        assert "initial_score" in result
        assert "final_score" in result
        assert "improvement" in result
        assert "history" in result
        assert isinstance(result["best_prompt"], str)
