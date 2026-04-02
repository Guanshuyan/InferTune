"""ACE: Agentic Context Engineering 上下文工程模块。

核心架构：Generator/Reflector/Curator 三角色协作，
通过增量 delta 更新维护 evolving playbook，避免 context collapse。
"""

from .engine import ContextEngine
from .playbook import Bullet, Playbook

__all__ = ["ContextEngine", "Bullet", "Playbook"]
