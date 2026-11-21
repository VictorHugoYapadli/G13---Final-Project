"""无向共现图及 TextRank 迭代。"""

from __future__ import annotations


def rank_words(
    tokens: list[str],
    eligible: list[bool],
    window: int,
    damping: float = 0.85,
    tolerance: float = 0.0001,
    max_iterations: int = 100,
    edge_weighting: str = "binary",
) -> tuple[dict[str, float], int, bool]:
    """构造局部共现图并按阻尼递归公式迭代节点分数。"""
    if len(tokens) != len(eligible):
        raise ValueError("tokens 与 eligible 的长度必须相同")
    if window < 2:
        raise ValueError("window 必须至少为 2")
    if not 0.0 <= damping < 1.0:
        raise ValueError("damping 必须在 [0, 1) 范围内")
    if tolerance <= 0 or max_iterations < 1:
        raise ValueError("tolerance 和 max_iterations 必须为正数")
    if edge_weighting not in {"binary", "distance_decay"}:
        raise ValueError("edge_weighting 必须为 binary 或 distance_decay")

    graph: dict[str, dict[str, float]] = {}
    for token, keep in zip(tokens, eligible):
        if keep:
            graph.setdefault(token, {})

    # 窗口按原文 token 位置计算，故 window=2 只连接相邻位置。
    for left_index, left_token in enumerate(tokens):
        if not eligible[left_index]:
            continue
        right_limit = min(len(tokens), left_index + window)
        for right_index in range(left_index + 1, right_limit):
            if not eligible[right_index]:
                continue
            right_token = tokens[right_index]
            if left_token != right_token:
                weight = (
                    1.0
                    if edge_weighting == "binary"
                    else 1.0 / (right_index - left_index)
                )
                if edge_weighting == "binary":
                    graph[left_token][right_token] = weight
                    graph[right_token][left_token] = weight
                else:
                    graph[left_token][right_token] = (
                        graph[left_token].get(right_token, 0.0) + weight
                    )
                    graph[right_token][left_token] = (
                        graph[right_token].get(left_token, 0.0) + weight
                    )

    if not graph:
        return {}, 0, True

    strengths = {word: sum(neighbors.values()) for word, neighbors in graph.items()}
    scores = {word: 1.0 for word in graph}
    for iteration in range(1, max_iterations + 1):
        updated: dict[str, float] = {}
        for word, neighbors in graph.items():
            recommendation = sum(
                scores[neighbor] * neighbors[neighbor] / strengths[neighbor]
                for neighbor in sorted(neighbors)
                if strengths[neighbor]
            )
            updated[word] = (1.0 - damping) + damping * recommendation

        # 论文以任意节点相邻两轮分数差低于阈值作为收敛判断。
        error = max(abs(updated[word] - scores[word]) for word in graph)
        scores = updated
        if error < tolerance:
            return scores, iteration, True

    return scores, max_iterations, False
