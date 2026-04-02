"""Playbook 数据结构。

Playbook 是 ACE 框架的核心数据结构，由结构化的 Bullet 条目组成。
每个 Bullet 是一个可复用的策略、领域知识或常见失败模式，
附带元数据（ID、有用/有害计数）用于增量更新和去重。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Bullet:
    """Playbook 中的单条策略条目。

    参数:
        id: 唯一标识符
        content: 策略/知识/失败模式的文本内容
        helpful_count: 被标记为有用的次数
        harmful_count: 被标记为有害的次数
        tags: 可选标签列表，用于分类检索
    """

    id: str = ""
    content: str = ""
    helpful_count: int = 0
    harmful_count: int = 0
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex[:12]

    @property
    def net_score(self) -> int:
        """净有用分数 = helpful - harmful。"""
        return self.helpful_count - self.harmful_count

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "id": self.id,
            "content": self.content,
            "helpful_count": self.helpful_count,
            "harmful_count": self.harmful_count,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Bullet:
        """从字典反序列化。

        参数:
            data: 字典数据

        返回:
            Bullet 实例
        """
        return cls(
            id=data.get("id", ""),
            content=data.get("content", ""),
            helpful_count=data.get("helpful_count", 0),
            harmful_count=data.get("harmful_count", 0),
            tags=data.get("tags", []),
        )


class Playbook:
    """Evolving Playbook，由 Bullet 条目组成的可演化上下文。

    支持增量添加、更新、删除、序列化，以及渲染为 LLM 可用的上下文文本。
    """

    def __init__(self, bullets: list[Bullet] | None = None):
        """初始化 Playbook。

        参数:
            bullets: 初始 Bullet 列表
        """
        self._bullets: dict[str, Bullet] = {}
        if bullets:
            for b in bullets:
                self._bullets[b.id] = b

    @property
    def bullets(self) -> list[Bullet]:
        """返回所有 Bullet 列表（按 net_score 降序）。"""
        return sorted(self._bullets.values(), key=lambda b: b.net_score, reverse=True)

    def __len__(self) -> int:
        return len(self._bullets)

    def get(self, bullet_id: str) -> Bullet | None:
        """根据 ID 获取 Bullet。

        参数:
            bullet_id: Bullet 唯一标识符

        返回:
            Bullet 实例，不存在则返回 None
        """
        return self._bullets.get(bullet_id)

    def add(self, bullet: Bullet) -> None:
        """添加新 Bullet。如果 ID 已存在则更新内容。

        参数:
            bullet: 待添加的 Bullet
        """
        if bullet.id in self._bullets:
            existing = self._bullets[bullet.id]
            existing.content = bullet.content
            existing.helpful_count += bullet.helpful_count
            existing.harmful_count += bullet.harmful_count
            existing.tags = list(set(existing.tags + bullet.tags))
        else:
            self._bullets[bullet.id] = bullet

    def remove(self, bullet_id: str) -> bool:
        """删除指定 Bullet。

        参数:
            bullet_id: 待删除的 Bullet ID

        返回:
            是否成功删除
        """
        if bullet_id in self._bullets:
            del self._bullets[bullet_id]
            return True
        return False

    def mark_helpful(self, bullet_id: str) -> None:
        """标记某 Bullet 为有用。

        参数:
            bullet_id: Bullet ID
        """
        if bullet_id in self._bullets:
            self._bullets[bullet_id].helpful_count += 1

    def mark_harmful(self, bullet_id: str) -> None:
        """标记某 Bullet 为有害。

        参数:
            bullet_id: Bullet ID
        """
        if bullet_id in self._bullets:
            self._bullets[bullet_id].harmful_count += 1

    def prune(self, min_net_score: int = -2) -> int:
        """清除净分数过低的 Bullet。

        参数:
            min_net_score: 最低净分数阈值，低于此值的 Bullet 将被移除

        返回:
            被移除的 Bullet 数量
        """
        to_remove = [bid for bid, b in self._bullets.items() if b.net_score < min_net_score]
        for bid in to_remove:
            del self._bullets[bid]
        return len(to_remove)

    def render(self, max_bullets: int | None = None) -> str:
        """将 Playbook 渲染为 LLM 可用的上下文文本。

        按 net_score 降序排列，每条 Bullet 作为一个编号条目。

        参数:
            max_bullets: 最多渲染的条目数，None 表示全部

        返回:
            格式化后的上下文文本
        """
        sorted_bullets = self.bullets
        if max_bullets is not None:
            sorted_bullets = sorted_bullets[:max_bullets]

        if not sorted_bullets:
            return ""

        lines = ["以下是经过验证的策略和知识：", ""]
        for i, b in enumerate(sorted_bullets, 1):
            score_info = f"[+{b.helpful_count}/-{b.harmful_count}]"
            lines.append(f"{i}. {score_info} {b.content}")
        return "\n".join(lines)

    def apply_delta(self, delta_bullets: list[Bullet]) -> None:
        """应用增量 delta 更新。

        新 ID 的 Bullet 直接添加，已有 ID 的 Bullet 合并计数。
        这是 ACE 避免 context collapse 的核心机制。

        参数:
            delta_bullets: 增量 Bullet 列表
        """
        for b in delta_bullets:
            self.add(b)

    def to_json(self) -> str:
        """序列化为 JSON 字符串。"""
        return json.dumps(
            [b.to_dict() for b in self.bullets],
            ensure_ascii=False,
            indent=2,
        )

    @classmethod
    def from_json(cls, json_str: str) -> Playbook:
        """从 JSON 字符串反序列化。

        参数:
            json_str: JSON 字符串

        返回:
            Playbook 实例
        """
        data = json.loads(json_str)
        bullets = [Bullet.from_dict(d) for d in data]
        return cls(bullets)

    def save(self, path: str | Path) -> None:
        """保存到文件。

        参数:
            path: 文件路径
        """
        path = Path(path)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Playbook:
        """从文件加载。

        参数:
            path: 文件路径

        返回:
            Playbook 实例
        """
        path = Path(path)
        return cls.from_json(path.read_text(encoding="utf-8"))
