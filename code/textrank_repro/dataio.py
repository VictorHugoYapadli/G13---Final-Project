"""固定格式 JSONL 数据读取与完整性校验。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取并校验每行一篇文档的输入文件。"""
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"第 {line_number} 行不是有效 JSON") from error
            if not isinstance(record, dict):
                raise ValueError(f"第 {line_number} 行必须是 JSON 对象")
            sample_id = record.get("id")
            text = record.get("text")
            keywords = record.get("keywords")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(f"第 {line_number} 行缺少非空字符串 id")
            if sample_id in seen_ids:
                raise ValueError(f"样本 id 重复：{sample_id}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"样本 {sample_id} 缺少非空 text")
            if not isinstance(keywords, list) or not keywords or not all(
                isinstance(item, str) and item.strip() for item in keywords
            ):
                raise ValueError(f"样本 {sample_id} 的 keywords 必须是非空字符串列表")
            seen_ids.add(sample_id)
            records.append({"id": sample_id, "text": text, "keywords": keywords})

    if not records:
        raise ValueError(f"数据文件没有有效样本：{path}")
    return records


def sha256_file(path: Path) -> str:
    """流式计算数据文件 SHA-256，便于固定数据版本核验。"""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()