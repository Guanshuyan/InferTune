"""Curator: 将洞察整合为结构化 delta 更新。

ACE 三角色架构中的第三个角色，负责将 Reflector 提取的洞察
转化为 Playbook 的增量 delta Bullet，并通过非 LLM 的确定性逻辑
合并到现有 Playbook 中，避免 context collapse。
"""

from typing import Any

from .playbook import Bullet, Playbook


def curate_delta(
    insights: list[dict[str, Any]],
    playbook: Playbook,
) -> list[Bullet]:
    """将洞察转化为 delta Bullet 列表。

    对于每条洞察，检查是否与现有 Bullet 语义重复：
    - 如果重复，生成更新现有 Bullet 计数的 delta
    - 如果不重复，生成新 Bullet

    参数:
        insights: Reflector 提取的洞察列表，每个元素为 {"content": str, "tags": list}
        playbook: 当前 Playbook（用于检查重复）

    返回:
        delta Bullet 列表，可直接通过 playbook.apply_delta() 应用
    """
    delta = []

    for insight in insights:
        content = insight.get("content", "").strip()
        if not content:
            continue

        tags = insight.get("tags", [])

        # 检查是否与现有 Bullet 内容高度相似（简单文本匹配）
        existing_id = _find_similar_bullet(content, playbook)

        if existing_id:
            # 已有类似 Bullet，生成更新 delta（增加 helpful 计数）
            delta.append(Bullet(
                id=existing_id,
                content=content,
                helpful_count=1,
                tags=tags,
            ))
        else:
            # 新洞察，生成新 Bullet
            delta.append(Bullet(
                content=content,
                helpful_count=1,
                tags=tags,
            ))

    return delta


def apply_feedback(
    playbook: Playbook,
    helpful_ids: list[str],
    harmful_ids: list[str],
) -> None:
    """根据 Generator 的反馈更新 Bullet 计数。

    参数:
        playbook: 当前 Playbook
        helpful_ids: 被标记为有用的 Bullet ID 列表
        harmful_ids: 被标记为有害的 Bullet ID 列表
    """
    for bid in helpful_ids:
        playbook.mark_helpful(bid)
    for bid in harmful_ids:
        playbook.mark_harmful(bid)


def _find_similar_bullet(content: str, playbook: Playbook) -> str | None:
    """在 Playbook 中查找与给定内容相似的 Bullet。

    使用简单的关键词重叠度判断相似性，避免引入 embedding 依赖。
    后续可通过 dedup.py 的语义去重进一步精炼。

    参数:
        content: 待匹配的文本内容
        playbook: 当前 Playbook

    返回:
        最相似 Bullet 的 ID，未找到则返回 None
    """
    content_words = set(content.lower().split())
    if not content_words:
        return None

    best_id = None
    best_overlap = 0.0

    for bullet in playbook.bullets:
        bullet_words = set(bullet.content.lower().split())
        if not bullet_words:
            continue

        # Jaccard 相似度
        intersection = len(content_words & bullet_words)
        union = len(content_words | bullet_words)
        overlap = intersection / union if union > 0 else 0.0

        if overlap > best_overlap:
            best_overlap = overlap
            best_id = bullet.id

    # 阈值：超过 0.6 认为是相似内容
    if best_overlap >= 0.6:
        return best_id
    return None
