"""语义去重与精炼模块。

实现 ACE 论文的 grow-and-refine 机制：
在 Playbook 持续增长的过程中，通过语义相似度检测合并冗余 Bullet，
保持上下文紧凑且不丢失关键信息。
"""

from __future__ import annotations

from typing import Sequence

from .playbook import Bullet, Playbook


def deduplicate(
    playbook: Playbook,
    *,
    threshold: float = 0.85,
    use_embedding: bool = False,
    embed_fn: object | None = None,
) -> int:
    """对 Playbook 中的 Bullet 进行去重。

    检测语义相似的 Bullet 对，将低分的合并到高分的中。
    默认使用基于词重叠的轻量方法，可选接入 embedding 模型。

    参数:
        playbook: 待去重的 Playbook
        threshold: 相似度阈值，超过此值的 Bullet 对将被合并
        use_embedding: 是否使用 embedding 计算相似度
        embed_fn: embedding 函数，签名为 (texts: list[str]) -> list[list[float]]；
                  仅在 use_embedding=True 时需要

    返回:
        被合并移除的 Bullet 数量
    """
    bullets = playbook.bullets
    if len(bullets) < 2:
        return 0

    if use_embedding and embed_fn is not None:
        return _deduplicate_by_embedding(playbook, bullets, threshold, embed_fn)
    else:
        return _deduplicate_by_words(playbook, bullets, threshold)


def _deduplicate_by_words(
    playbook: Playbook,
    bullets: list[Bullet],
    threshold: float,
) -> int:
    """基于词重叠度的去重。

    使用 Jaccard 相似度衡量两条 Bullet 的文本相似性。

    参数:
        playbook: Playbook 实例
        bullets: 按 net_score 降序排列的 Bullet 列表
        threshold: 相似度阈值

    返回:
        被移除的 Bullet 数量
    """
    removed = set()
    n = len(bullets)

    for i in range(n):
        if bullets[i].id in removed:
            continue
        words_i = set(bullets[i].content.lower().split())
        if not words_i:
            continue

        for j in range(i + 1, n):
            if bullets[j].id in removed:
                continue
            words_j = set(bullets[j].content.lower().split())
            if not words_j:
                continue

            # Jaccard 相似度
            intersection = len(words_i & words_j)
            union = len(words_i | words_j)
            sim = intersection / union if union > 0 else 0.0

            if sim >= threshold:
                # 保留 net_score 更高的（bullets 已按 net_score 降序排列）
                # 将低分 Bullet 的计数合并到高分 Bullet
                _merge_bullet(playbook, keep_id=bullets[i].id, remove_id=bullets[j].id)
                removed.add(bullets[j].id)

    return len(removed)


def _deduplicate_by_embedding(
    playbook: Playbook,
    bullets: list[Bullet],
    threshold: float,
    embed_fn: object,
) -> int:
    """基于 embedding 余弦相似度的去重。

    参数:
        playbook: Playbook 实例
        bullets: Bullet 列表
        threshold: 余弦相似度阈值
        embed_fn: embedding 函数

    返回:
        被移除的 Bullet 数量
    """
    texts = [b.content for b in bullets]
    embeddings = embed_fn(texts)

    removed = set()
    n = len(bullets)

    for i in range(n):
        if bullets[i].id in removed:
            continue
        for j in range(i + 1, n):
            if bullets[j].id in removed:
                continue

            sim = _cosine_similarity(embeddings[i], embeddings[j])
            if sim >= threshold:
                _merge_bullet(playbook, keep_id=bullets[i].id, remove_id=bullets[j].id)
                removed.add(bullets[j].id)

    return len(removed)


def _merge_bullet(playbook: Playbook, keep_id: str, remove_id: str) -> None:
    """将一个 Bullet 合并到另一个中。

    将被移除 Bullet 的计数累加到保留的 Bullet 上，然后删除。

    参数:
        playbook: Playbook 实例
        keep_id: 保留的 Bullet ID
        remove_id: 待移除的 Bullet ID
    """
    keep = playbook.get(keep_id)
    remove = playbook.get(remove_id)
    if keep and remove:
        keep.helpful_count += remove.helpful_count
        keep.harmful_count += remove.harmful_count
        keep.tags = list(set(keep.tags + remove.tags))
        playbook.remove(remove_id)


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """计算两个向量的余弦相似度。

    参数:
        vec_a: 向量 A
        vec_b: 向量 B

    返回:
        余弦相似度，范围 [-1, 1]
    """
    import math

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
