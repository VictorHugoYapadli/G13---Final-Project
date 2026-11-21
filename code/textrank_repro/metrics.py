"""关键词排序指标与候选覆盖率。"""

from __future__ import annotations

from collections.abc import Iterable

from .text import normalize_phrase


def document_metrics(predictions: list[str], references: list[str], cutoff: int) -> dict[str, float]:
    """计算单文档精确匹配的 P/R/F1@k。"""
    gold = {normalize_phrase(item) for item in references if normalize_phrase(item)}
    predicted = [normalize_phrase(item) for item in predictions[:cutoff]]
    correct = len(set(predicted) & gold)
    precision = correct / cutoff
    recall = correct / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "correct": float(correct)}


def candidate_coverage(candidates: Iterable[str], references: list[str]) -> tuple[float, int, int]:
    """返回单文档覆盖率、可匹配参考词数和参考词总数。"""
    gold = {normalize_phrase(item) for item in references if normalize_phrase(item)}
    available = {normalize_phrase(item) for item in candidates if normalize_phrase(item)}
    matched = len(gold & available)
    return (matched / len(gold) if gold else 0.0, matched, len(gold))