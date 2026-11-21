"""文本预处理与候选词识别。"""

from __future__ import annotations

import re
from functools import lru_cache

TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")


def tokenize(text: str) -> list[str]:
    """使用固定正则规则抽取英文词项。"""
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def phrase_boundaries(text: str) -> list[bool]:
    """标记词项之间是否有标点，供短语合并时阻断跨标点连接。"""
    matches = list(TOKEN_PATTERN.finditer(text))
    boundaries = [False]
    for previous, current in zip(matches, matches[1:]):
        gap = text[previous.end() : current.start()]
        boundaries.append(any(not character.isspace() for character in gap))
    return boundaries


def tag_text(text: str) -> list[tuple[str, str]]:
    """用固定版本的 NLTK 英文模型标注词性；模型缺失时由 NLTK 给出提示。"""
    try:
        from nltk import pos_tag
    except ImportError as error:
        raise RuntimeError("请先执行 python -m pip install -r code/requirements.txt") from error

    tokens = tokenize(text)
    if not tokens:
        return []
    try:
        return pos_tag(tokens, lang="eng")
    except LookupError as error:
        raise RuntimeError(
            "缺少 NLTK 英文词性模型；请执行 "
            "python -m nltk.downloader averaged_perceptron_tagger_eng"
        ) from error


def tag_text_spacy(text: str, model_name: str = "en_core_web_sm") -> list[tuple[str, str]]:
    """用 spaCy 的 Penn Treebank 标签标注与本项目相同的正则词元。"""
    tokens = tokenize(text)
    if not tokens:
        return []

    nlp = _get_spacy_model(model_name)
    try:
        from spacy.tokens import Doc
    except ImportError as error:
        raise RuntimeError(
            "spaCy POS 标注需要安装 spaCy；请执行 python -m pip install spacy"
        ) from error

    doc = Doc(nlp.vocab, words=tokens, spaces=[True] * len(tokens))
    doc = nlp(doc)
    if not doc.has_annotation("TAG"):
        raise RuntimeError(f"spaCy 模型 {model_name!r} 未生成 Penn Treebank POS 标签")
    return [(token.text, token.tag_) for token in doc]


@lru_cache(maxsize=2)
def _get_spacy_model(model_name: str):
    try:
        import spacy
    except ImportError as error:
        raise RuntimeError(
            "spaCy POS 标注需要安装 spaCy；请执行 python -m pip install spacy"
        ) from error

    try:
        nlp = spacy.load(
            model_name,
            disable=["parser", "ner", "lemmatizer", "attribute_ruler"],
        )
    except OSError as error:
        raise RuntimeError(
            f"找不到 spaCy 模型 {model_name!r}；请执行 "
            f"python -m spacy download {model_name}"
        ) from error
    if "tagger" not in nlp.pipe_names:
        raise RuntimeError(f"spaCy 模型 {model_name!r} 不包含 Penn Treebank tagger")
    return nlp


def spacy_model_info(model_name: str) -> dict[str, str]:
    """返回当前运行实际加载的 spaCy 与模型版本。"""
    import spacy

    nlp = _get_spacy_model(model_name)
    return {
        "spacy_version": spacy.__version__,
        "model_name": model_name,
        "model_version": str(nlp.meta.get("version", "unknown")),
    }


@lru_cache(maxsize=50000)
def normalize_phrase(phrase: str) -> str:
    """按小写、正则分词和 Porter 词干化统一短语匹配口径。"""
    tokens = tokenize(phrase)
    if not tokens:
        return ""
    try:
        from nltk.stem import PorterStemmer
    except ImportError as error:
        raise RuntimeError("请先执行 python -m pip install -r code/requirements.txt") from error

    stemmer = _get_stemmer()
    return " ".join(stemmer.stem(token) for token in tokens)


@lru_cache(maxsize=1)
def _get_stemmer():
    from nltk.stem import PorterStemmer

    return PorterStemmer()