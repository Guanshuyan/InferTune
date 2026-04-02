"""ACE 主流程编排。

将 Generator/Reflector/Curator 三角色串联，
提供 offline（系统提示优化）和 online（测试时记忆适应）两种运行模式。
"""

from typing import Any, Callable

from ..llm.base import BaseLLM
from ..evaluator.base import BaseEvaluator
from ..evaluator.cost_estimator import estimate_ace_offline
from ..config.settings import ACEConfig
from .playbook import Playbook
from .generator import generate_trajectory, generate_batch_trajectories
from .reflector import reflect_on_trajectory, refine_insights
from .curator import curate_delta, apply_feedback
from .dedup import deduplicate


class ContextEngine:
    """ACE 上下文工程引擎。

    封装完整的上下文优化流程，支持 offline 和 online 两种模式。

    参数:
        llm: LLM 实例，三个角色共用同一个 LLM
        config: ACE 配置，为 None 时使用默认值
        playbook: 初始 Playbook，为 None 时创建空 Playbook
    """

    def __init__(
        self,
        llm: BaseLLM,
        config: ACEConfig | None = None,
        playbook: Playbook | None = None,
    ):
        self.llm = llm
        self.config = config or ACEConfig()
        self.playbook = playbook or Playbook()

    def optimize_offline(
        self,
        train_data: list[dict[str, Any]],
        *,
        system_prompt: str = "",
        evaluator: BaseEvaluator | None = None,
        eval_data: list[dict[str, Any]] | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Offline 模式：在训练数据上多轮迭代优化 Playbook。

        对应 ACE 论文的 offline adaptation，用于系统提示优化场景。
        支持多 epoch 反复遍历训练数据以逐步强化上下文。

        参数:
            train_data: 训练数据列表，每个元素为 {"input": str, "label": str}
            system_prompt: 任务描述系统提示
            evaluator: 评估器（可选），用于跟踪优化进度
            eval_data: 评估数据（可选），用于计算阶段性分数
            verbose: 是否输出中间过程

        返回:
            优化结果字典，包含:
            - "playbook": 优化后的 Playbook
            - "context": 渲染后的上下文文本
            - "num_bullets": 最终 Bullet 数量
            - "epoch_scores": 每个 epoch 的评估分数（如提供了 evaluator）
            - "history": 详细历史记录
            - "cost_estimate": 调用次数估算结果
        """
        cfg = self.config

        # 运行前估算 LLM 调用次数
        eval_size = len(eval_data) if eval_data else 0
        cost = estimate_ace_offline(cfg, len(train_data), eval_size, has_evaluator=evaluator is not None)
        if verbose:
            print(f"\n{cost.summary()}\n")

        history = []
        epoch_scores = []

        for epoch in range(cfg.max_epochs):
            if verbose:
                print(f"[ACE-Offline] Epoch {epoch + 1}/{cfg.max_epochs}，"
                      f"当前 Playbook 大小: {len(self.playbook)}")

            epoch_record = {"epoch": epoch + 1, "steps": []}

            # 遍历训练数据，逐条处理
            for i, item in enumerate(train_data):
                step_result = self._process_single(
                    item["input"],
                    label=item.get("label"),
                    system_prompt=system_prompt,
                )
                epoch_record["steps"].append(step_result)

            # 去重精炼
            removed = deduplicate(self.playbook, threshold=cfg.dedup_threshold)
            epoch_record["dedup_removed"] = removed

            # 清除低质量 Bullet
            pruned = self.playbook.prune(min_net_score=-2)
            epoch_record["pruned"] = pruned

            # 阶段性评估
            if evaluator and eval_data:
                score = self._evaluate(eval_data, evaluator, system_prompt)
                epoch_scores.append(score)
                epoch_record["eval_score"] = score
                if verbose:
                    print(f"[ACE-Offline] Epoch {epoch + 1} 评估分数: {score:.4f}")

            history.append(epoch_record)

            if verbose:
                print(f"[ACE-Offline] Epoch {epoch + 1} 完成，"
                      f"Playbook 大小: {len(self.playbook)}，"
                      f"去重移除: {removed}，清除: {pruned}")

        return {
            "playbook": self.playbook,
            "context": self.playbook.render(),
            "num_bullets": len(self.playbook),
            "epoch_scores": epoch_scores,
            "history": history,
            "cost_estimate": cost,
        }

    def optimize_online(
        self,
        input_text: str,
        *,
        label: str | None = None,
        system_prompt: str = "",
    ) -> dict[str, Any]:
        """Online 模式：处理单条输入并实时更新 Playbook。

        对应 ACE 论文的 online adaptation，用于测试时记忆适应场景。
        每处理一条输入，Playbook 就会增量更新。

        参数:
            input_text: 任务输入文本
            label: 真实标签（可选），有标签时可提供更精确的反馈
            system_prompt: 任务描述系统提示

        返回:
            处理结果字典，包含:
            - "answer": 模型回答
            - "correct": 是否正确（仅在提供 label 时有值）
            - "new_bullets": 本次新增的 Bullet 数量
            - "playbook_size": 更新后的 Playbook 大小
        """
        result = self._process_single(input_text, label=label, system_prompt=system_prompt)

        # 懒去重：当 Playbook 超过一定大小时触发
        if len(self.playbook) > 50:
            deduplicate(self.playbook, threshold=self.config.dedup_threshold)

        return result

    def get_context(self, max_bullets: int | None = None) -> str:
        """获取当前 Playbook 渲染的上下文文本。

        参数:
            max_bullets: 最多返回的条目数

        返回:
            格式化后的上下文文本
        """
        return self.playbook.render(max_bullets=max_bullets)

    def _process_single(
        self,
        input_text: str,
        *,
        label: str | None = None,
        system_prompt: str = "",
    ) -> dict[str, Any]:
        """处理单条输入的完整 Generator → Reflector → Curator 流程。

        参数:
            input_text: 任务输入
            label: 真实标签（可选）
            system_prompt: 系统提示

        返回:
            处理结果字典
        """
        # 1. Generator: 生成推理轨迹
        trajectory = generate_trajectory(
            self.llm, self.playbook, input_text,
            system_prompt=system_prompt,
        )

        # 补充标签信息
        if label is not None:
            trajectory["label"] = label
            answer = trajectory.get("answer", "")
            trajectory["correct"] = (
                answer.strip().lower() == label.strip().lower()
                or label.strip().lower() in answer.strip().lower()
            )

        # 更新 Bullet 反馈计数
        apply_feedback(
            self.playbook,
            trajectory.get("helpful_ids", []),
            trajectory.get("harmful_ids", []),
        )

        # 2. Reflector: 提取洞察
        insights = reflect_on_trajectory(self.llm, trajectory)

        # 迭代精炼
        if insights and self.config.max_reflector_rounds > 1:
            insights = refine_insights(
                self.llm, insights, trajectory,
                max_rounds=self.config.max_reflector_rounds - 1,
            )

        # 3. Curator: 生成 delta 并应用
        delta = curate_delta(insights, self.playbook)
        self.playbook.apply_delta(delta)

        return {
            "answer": trajectory.get("answer", ""),
            "correct": trajectory.get("correct"),
            "new_bullets": len(delta),
            "playbook_size": len(self.playbook),
        }

    def _evaluate(
        self,
        data: list[dict[str, Any]],
        evaluator: BaseEvaluator,
        system_prompt: str = "",
    ) -> float:
        """在数据集上评估当前 Playbook 的效果。

        参数:
            data: 评估数据
            evaluator: 评估器
            system_prompt: 系统提示

        返回:
            评估分数
        """
        predictions = []
        labels = []

        for item in data:
            trajectory = generate_trajectory(
                self.llm, self.playbook, item["input"],
                system_prompt=system_prompt,
            )
            predictions.append(trajectory.get("answer", ""))
            labels.append(item.get("label", ""))

        return evaluator.score(predictions, labels)
