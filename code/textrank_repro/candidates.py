"""从连续名词/形容词序列生成共享短语候选集。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    phrase: str
    start: int
    end: int
    occurrences: int


def eligible_mask(tagged_tokens: list[tuple[str, str]], allowed_tags: set[str]) -> list[bool]:
    """逐词判断是否通过配置中的词性筛选。"""
    return [tag in allowed_tags for _, tag in tagged_tokens]


def exclude_single_character_tokens(tokens: list[str], eligible: list[bool]) -> list[bool]:
    """Create an ablation mask that drops eligible single-letter tokens."""
    if len(tokens) != len(eligible):
        raise ValueError("tokens 与 eligible 的长度必须相同")
    return [keep and len(token) > 1 for token, keep in zip(tokens, eligible)]


def extract_candidates(
    tokens: list[str], eligible: list[bool], break_before: list[bool] | None = None
) -> list[Candidate]:
    """生成所有连续合格词序列的子短语，并按文本位置合并重复项。"""
    if len(tokens) != len(eligible):
        raise ValueError("tokens 与 eligible 的长度必须相同")
    if break_before is None:
        break_before = [False] * len(tokens)
    if len(tokens) != len(break_before):
        raise ValueError("tokens 与 break_before 的长度必须相同")

    found: dict[str, list[int]] = {}
    run_start = 0
    while run_start < len(tokens):
        if not eligible[run_start]:
            run_start += 1
            continue

        run_end = run_start + 1
        while run_end < len(tokens) and eligible[run_end] and not break_before[run_end]:
            run_end += 1

        # 保留连续子短语，使短语候选召回不依赖某个排序器先选中哪些单词。
        for start in range(run_start, run_end):
            for end in range(start + 1, run_end + 1):
                phrase = " ".join(tokens[start:end])
                found.setdefault(phrase, []).append(start)
        run_start = run_end

    return [
        Candidate(phrase=phrase, start=min(positions), end=min(positions) + len(phrase.split()), occurrences=len(positions))
        for phrase, positions in found.items()
    ]


def merge_selected_words(
    tokens: list[str],
    eligible: list[bool],
    scores: dict[str, float],
    top_fraction: float = 1 / 3,
    break_before: list[bool] | None = None,
) -> list[tuple[Candidate, float]]:
    """选择排名靠前的词节点，再把原文中相邻节点合并为短语。"""
    if len(tokens) != len(eligible):
        raise ValueError("tokens 与 eligible 的长度必须相同")
    if break_before is None:
        break_before = [False] * len(tokens)
    if len(tokens) != len(break_before):
        raise ValueError("tokens 与 break_before 的长度必须相同")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction 必须在 (0, 1] 范围内")
    if not scores:
        return []

    selected_count = max(1, int(len(scores) * top_fraction))
    first_position: dict[str, int] = {}
    for index, (token, keep) in enumerate(zip(tokens, eligible)):
        if keep and token in scores:
            first_position.setdefault(token, index)
    ranked_words = sorted(
        scores,
        key=lambda word: (-scores[word], first_position.get(word, len(tokens)), word),
    )
    selected_words = set(ranked_words[:selected_count])
    merged: dict[str, tuple[Candidate, float]] = {}
    run_start: int | None = None

    def add_run(start: int, end: int) -> None:
        phrase = " ".join(tokens[start:end])
        phrase_score = sum(scores[tokens[index]] for index in range(start, end))
        previous = merged.get(phrase)
        candidate = Candidate(phrase=phrase, start=start, end=end, occurrences=1)
        if previous is None:
            merged[phrase] = (candidate, phrase_score)
        else:
            old_candidate, old_score = previous
            merged[phrase] = (
                Candidate(
                    phrase=phrase,
                    start=min(old_candidate.start, start),
                    end=min(old_candidate.start, start) + len(phrase.split()),
                    occurrences=old_candidate.occurrences + 1,
                ),
                max(old_score, phrase_score),
            )

    for index, (token, keep) in enumerate(zip(tokens, eligible)):
        if keep and token in selected_words:
            if run_start is not None and break_before[index]:
                add_run(run_start, index)
                run_start = index
            elif run_start is None:
                run_start = index
        elif run_start is not None:
            add_run(run_start, index)
            run_start = None
    if run_start is not None:
        add_run(run_start, len(tokens))

    return sorted(
        merged.values(),
        key=lambda item: (-item[1], item[0].start, -len(item[0].phrase.split()), item[0].phrase),
    )