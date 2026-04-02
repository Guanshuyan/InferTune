"""ACE 上下文工程模块单元测试。

使用 Mock LLM 测试 Playbook、Generator、Reflector、Curator、Dedup、Engine 的逻辑正确性。
"""

import json
import tempfile
from pathlib import Path

from infertune.llm.base import BaseLLM, ChatCompletionResponse
from infertune.evaluator.metrics import AccuracyEvaluator
from infertune.config.settings import ACEConfig
from infertune.context_engine.playbook import Bullet, Playbook
from infertune.context_engine.generator import (
    generate_trajectory,
    _parse_trajectory_output,
    _extract_numbers,
    _check_correct,
)
from infertune.context_engine.reflector import (
    reflect_on_trajectory,
    refine_insights,
    _parse_insights,
)
from infertune.context_engine.curator import curate_delta, apply_feedback, _find_similar_bullet
from infertune.context_engine.dedup import deduplicate, _cosine_similarity
from infertune.context_engine.engine import ContextEngine


class _MockLLM(BaseLLM):
    """测试用 Mock LLM。"""

    def __init__(self, responses: list[str] | None = None):
        super().__init__(model="mock")
        self._responses = responses or ["mock response"]
        self._call_count = 0

    def chat_completion(self, messages, *, model=None, temperature=None, max_tokens=None, **kwargs):
        resp_text = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return ChatCompletionResponse(content=resp_text, model="mock")


# ---- Playbook 测试 ----

class TestBullet:
    """Bullet 数据结构测试。"""

    def test_auto_id(self):
        b = Bullet(content="test")
        assert len(b.id) == 12

    def test_net_score(self):
        b = Bullet(helpful_count=5, harmful_count=2)
        assert b.net_score == 3

    def test_serialization(self):
        b = Bullet(id="abc", content="策略1", helpful_count=3, tags=["tag1"])
        d = b.to_dict()
        b2 = Bullet.from_dict(d)
        assert b2.id == "abc"
        assert b2.content == "策略1"
        assert b2.helpful_count == 3


class TestPlaybook:
    """Playbook 测试。"""

    def test_add_and_len(self):
        pb = Playbook()
        pb.add(Bullet(id="a", content="策略A"))
        pb.add(Bullet(id="b", content="策略B"))
        assert len(pb) == 2

    def test_add_duplicate_merges(self):
        pb = Playbook()
        pb.add(Bullet(id="a", content="v1", helpful_count=1))
        pb.add(Bullet(id="a", content="v2", helpful_count=2))
        assert len(pb) == 1
        assert pb.get("a").content == "v2"
        assert pb.get("a").helpful_count == 3

    def test_remove(self):
        pb = Playbook()
        pb.add(Bullet(id="a", content="test"))
        assert pb.remove("a") is True
        assert len(pb) == 0
        assert pb.remove("nonexistent") is False

    def test_mark_helpful_harmful(self):
        pb = Playbook()
        pb.add(Bullet(id="a", content="test"))
        pb.mark_helpful("a")
        pb.mark_helpful("a")
        pb.mark_harmful("a")
        assert pb.get("a").helpful_count == 2
        assert pb.get("a").harmful_count == 1

    def test_prune(self):
        pb = Playbook()
        pb.add(Bullet(id="good", content="好策略", helpful_count=5))
        pb.add(Bullet(id="bad", content="坏策略", harmful_count=5))
        removed = pb.prune(min_net_score=-2)
        assert removed == 1
        assert len(pb) == 1
        assert pb.get("good") is not None

    def test_render(self):
        pb = Playbook()
        pb.add(Bullet(id="a", content="使用具体的分类标准", helpful_count=3))
        text = pb.render()
        assert "使用具体的分类标准" in text
        assert "[+3/-0]" in text

    def test_render_max_bullets(self):
        pb = Playbook()
        for i in range(10):
            pb.add(Bullet(content=f"策略{i}"))
        text = pb.render(max_bullets=3)
        # 应该只包含 3 条
        assert text.count(". [") == 3

    def test_apply_delta(self):
        pb = Playbook()
        pb.add(Bullet(id="existing", content="已有策略"))
        delta = [
            Bullet(id="existing", content="更新策略", helpful_count=1),
            Bullet(id="new1", content="新策略"),
        ]
        pb.apply_delta(delta)
        assert len(pb) == 2
        assert pb.get("existing").helpful_count == 1

    def test_json_serialization(self):
        pb = Playbook()
        pb.add(Bullet(id="a", content="策略A", helpful_count=2))
        pb.add(Bullet(id="b", content="策略B", harmful_count=1))

        json_str = pb.to_json()
        pb2 = Playbook.from_json(json_str)
        assert len(pb2) == 2
        assert pb2.get("a").content == "策略A"

    def test_save_load(self):
        pb = Playbook()
        pb.add(Bullet(id="x", content="持久化测试"))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        pb.save(path)
        pb2 = Playbook.load(path)
        assert len(pb2) == 1
        assert pb2.get("x").content == "持久化测试"

    def test_bullets_sorted_by_net_score(self):
        pb = Playbook()
        pb.add(Bullet(id="low", content="低分", helpful_count=1, harmful_count=3))
        pb.add(Bullet(id="high", content="高分", helpful_count=5))
        pb.add(Bullet(id="mid", content="中分", helpful_count=2))
        bullets = pb.bullets
        assert bullets[0].id == "high"
        assert bullets[-1].id == "low"


# ---- Generator 测试 ----

class TestGenerator:
    """Generator 模块测试。"""

    def test_generate_trajectory(self):
        llm = _MockLLM(["[有用策略]: 无\n[有害策略]: 无\n[回答]: positive"])
        pb = Playbook()
        traj = generate_trajectory(llm, pb, "great movie")
        assert traj["input"] == "great movie"
        assert traj["answer"] == "positive"

    def test_extract_numbers(self):
        assert _extract_numbers("1, 3, 5") == [1, 3, 5]
        assert _extract_numbers("无") == []
        assert _extract_numbers("策略 2 和 4") == [2, 4]

    def test_check_correct(self):
        assert _check_correct("positive", "positive") is True
        assert _check_correct("the answer is positive", "positive") is True
        assert _check_correct("negative", "positive") is False


# ---- Reflector 测试 ----

class TestReflector:
    """Reflector 模块测试。"""

    def test_parse_insights_json(self):
        response = '[{"content": "使用情感词汇判断", "tags": ["情感分析"]}]'
        insights = _parse_insights(response)
        assert len(insights) == 1
        assert insights[0]["content"] == "使用情感词汇判断"

    def test_parse_insights_markdown(self):
        response = '```json\n[{"content": "test", "tags": []}]\n```'
        insights = _parse_insights(response)
        assert len(insights) == 1

    def test_parse_insights_invalid(self):
        insights = _parse_insights("这不是 JSON")
        assert insights == []

    def test_reflect_on_trajectory(self):
        llm = _MockLLM(['[{"content": "注意否定词", "tags": ["语义"]}]'])
        traj = {"input": "not good", "output": "positive", "correct": False, "label": "negative"}
        insights = reflect_on_trajectory(llm, traj)
        assert len(insights) == 1

    def test_refine_insights(self):
        llm = _MockLLM(['[{"content": "精炼后的策略", "tags": ["改进"]}]'])
        initial = [{"content": "初始策略", "tags": []}]
        traj = {"input": "test", "output": "result"}
        refined = refine_insights(llm, initial, traj, max_rounds=1)
        assert len(refined) >= 1


# ---- Curator 测试 ----

class TestCurator:
    """Curator 模块测试。"""

    def test_curate_delta_new(self):
        pb = Playbook()
        insights = [{"content": "新策略", "tags": ["test"]}]
        delta = curate_delta(insights, pb)
        assert len(delta) == 1
        assert delta[0].content == "新策略"

    def test_curate_delta_existing(self):
        pb = Playbook()
        pb.add(Bullet(id="exist", content="已有 策略 内容"))
        insights = [{"content": "已有 策略 内容 更新", "tags": []}]
        delta = curate_delta(insights, pb)
        # 应该匹配到已有 Bullet（Jaccard 相似度高）
        assert len(delta) == 1

    def test_apply_feedback(self):
        pb = Playbook()
        pb.add(Bullet(id="a", content="test"))
        pb.add(Bullet(id="b", content="test2"))
        apply_feedback(pb, helpful_ids=["a"], harmful_ids=["b"])
        assert pb.get("a").helpful_count == 1
        assert pb.get("b").harmful_count == 1

    def test_find_similar_bullet(self):
        pb = Playbook()
        pb.add(Bullet(id="x", content="使用 情感 词汇 来 判断 文本 情感"))
        # 高相似度
        result = _find_similar_bullet("使用 情感 词汇 来 判断 文本 倾向", pb)
        assert result == "x"
        # 低相似度
        result = _find_similar_bullet("完全不同的内容关于天气预报", pb)
        assert result is None


# ---- Dedup 测试 ----

class TestDedup:
    """去重模块测试。"""

    def test_deduplicate_similar(self):
        pb = Playbook()
        pb.add(Bullet(id="a", content="使用 情感 词汇 判断 文本 情感 倾向", helpful_count=3))
        pb.add(Bullet(id="b", content="使用 情感 词汇 判断 文本 情感 方向", helpful_count=1))
        removed = deduplicate(pb, threshold=0.7)
        assert removed == 1
        assert len(pb) == 1
        # 应保留高分的
        assert pb.get("a") is not None

    def test_deduplicate_different(self):
        pb = Playbook()
        pb.add(Bullet(id="a", content="情感分析策略"))
        pb.add(Bullet(id="b", content="代码生成技巧"))
        removed = deduplicate(pb, threshold=0.7)
        assert removed == 0
        assert len(pb) == 2

    def test_cosine_similarity(self):
        assert abs(_cosine_similarity([1, 0], [1, 0]) - 1.0) < 1e-6
        assert abs(_cosine_similarity([1, 0], [0, 1]) - 0.0) < 1e-6
        assert abs(_cosine_similarity([0, 0], [1, 1]) - 0.0) < 1e-6

    def test_deduplicate_empty(self):
        pb = Playbook()
        removed = deduplicate(pb)
        assert removed == 0


# ---- Engine 集成测试 ----

class TestContextEngine:
    """ContextEngine 集成测试。"""

    def test_optimize_online(self):
        llm = _MockLLM([
            # Generator 输出
            "[有用策略]: 无\n[有害策略]: 无\n[回答]: positive",
            # Reflector 输出
            '[{"content": "关注积极词汇", "tags": ["情感"]}]',
            # Reflector refine 输出
            '[{"content": "关注积极和消极词汇", "tags": ["情感"]}]',
        ])

        config = ACEConfig(max_reflector_rounds=2)
        engine = ContextEngine(llm, config)

        result = engine.optimize_online("great movie", label="positive")
        assert "answer" in result
        assert result["playbook_size"] > 0

    def test_optimize_offline(self):
        llm = _MockLLM([
            # 每条数据的 Generator + Reflector + Refine 循环
            "[回答]: positive",
            '[{"content": "策略1", "tags": []}]',
            '[{"content": "策略1改进", "tags": []}]',
            "[回答]: negative",
            '[{"content": "策略2", "tags": []}]',
            '[{"content": "策略2改进", "tags": []}]',
        ])

        config = ACEConfig(max_epochs=1, max_reflector_rounds=2)
        engine = ContextEngine(llm, config)

        train_data = [
            {"input": "good", "label": "positive"},
            {"input": "bad", "label": "negative"},
        ]

        result = engine.optimize_offline(train_data)
        assert result["num_bullets"] > 0
        assert isinstance(result["context"], str)

    def test_get_context(self):
        engine = ContextEngine(_MockLLM())
        engine.playbook.add(Bullet(content="测试策略", helpful_count=1))
        ctx = engine.get_context()
        assert "测试策略" in ctx

    def test_offline_with_evaluator(self):
        """验证 offline 模式带评估器时能正常运行。"""
        llm = _MockLLM([
            "[回答]: yes",
            '[{"content": "s1", "tags": []}]',
            '[{"content": "s1v2", "tags": []}]',
            # 评估阶段的 Generator 调用
            "[回答]: yes",
        ])

        config = ACEConfig(max_epochs=1, max_reflector_rounds=2)
        engine = ContextEngine(llm, config)

        train_data = [{"input": "q1", "label": "yes"}]
        eval_data = [{"input": "q1", "label": "yes"}]
        evaluator = AccuracyEvaluator()

        result = engine.optimize_offline(
            train_data, evaluator=evaluator, eval_data=eval_data,
        )
        assert len(result["epoch_scores"]) == 1
