"""ProTeGi: 基于文本梯度的自动提示优化模块。

核心流程：minibatch 错误样本 → 生成自然语言梯度 → 反向编辑 prompt → beam search 选择最优候选。
"""

from .optimizer import PromptOptimizer

__all__ = ["PromptOptimizer"]
