"""下载固定版本的 Inspec 测试集并生成可追溯的固定子集。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

DATASET_ID = "midas/inspec"
DATASET_REVISION = "9617780e99705df88a4cb239174c86b9d8d8300f"
SOURCE_URL = (
    "https://huggingface.co/datasets/midas/inspec/resolve/"
    f"{DATASET_REVISION}/test.jsonl"
)
ROOT = Path(__file__).resolve().parent
TREEBANK_MARKERS = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LSB-": "[",
    "-RSB-": "]",
    "-LCB-": "{",
    "-RCB-": "}",
    "``": '"',
    "''": '"',
}


def sha256_bytes(content: bytes) -> str:
    """计算下载文件的 SHA-256，供数据版本核验。"""
    return hashlib.sha256(content).hexdigest()


def document_text(tokens: list[str]) -> str:
    """把 Penn Treebank 标点占位符还原为标点，避免其被误认成词。"""
    return " ".join(TREEBANK_MARKERS.get(token, token) for token in tokens)


def read_or_download(source_path: Path) -> bytes:
    """优先复用本地原始文件；缺失时从固定提交下载。"""
    if source_path.is_file():
        return source_path.read_bytes()

    source_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(SOURCE_URL, headers={"User-Agent": "TextRank-course-reproduction/1.0"})
    with urlopen(request, timeout=60) as response:
        content = response.read()
    if not content:
        raise RuntimeError("下载到的 Inspec 文件为空")
    source_path.write_bytes(content)
    return content


def parse_source(content: bytes) -> list[dict[str, object]]:
    """校验 MIDAS JSONL 字段并合并存在/缺席关键词作为原始参考集。"""
    records: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(content.decode("utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(f"源文件第 {line_number} 行不是有效 JSON") from error

        source_id = record.get("paper_id")
        document = record.get("document")
        extractive = record.get("extractive_keyphrases") or []
        abstractive = record.get("abstractive_keyphrases") or []
        if source_id is None or not isinstance(document, list) or not all(
            isinstance(token, str) for token in document
        ):
            raise ValueError(f"源文件第 {line_number} 行的 ID 或 document 格式不正确")
        if not isinstance(extractive, list) or not isinstance(abstractive, list):
            raise ValueError(f"源文件第 {line_number} 行的关键词必须是列表")

        sample_id = str(source_id)
        if sample_id in seen_ids:
            raise ValueError(f"源数据 ID 重复：{sample_id}")
        seen_ids.add(sample_id)
        keywords = list(dict.fromkeys(extractive + abstractive))
        if keywords and all(isinstance(keyword, str) and keyword.strip() for keyword in keywords):
            records.append({"id": sample_id, "text": document_text(document), "keywords": keywords})

    if not records:
        raise ValueError("源文件中没有带有效关键词标注的文档")
    return records


def prepare_subset(
    count: int,
    seed: int,
    source_path: Path,
    output_path: Path,
    metadata_path: Path,
    ids_path: Path,
) -> None:
    """固定抽样、写出模型输入，并保存足以审计来源的元数据。"""
    source_bytes = read_or_download(source_path)
    records = parse_source(source_bytes)
    if count < 1 or count > len(records):
        raise ValueError(f"抽样数量必须在 1 到 {len(records)} 之间，当前为 {count}")

    source_rows = [json.loads(line) for line in source_bytes.decode("utf-8").splitlines() if line.strip()]
    selected = random.Random(seed).sample(records, count)
    source_order = {str(row["paper_id"]): index for index, row in enumerate(source_rows)}
    selected.sort(key=lambda row: source_order[str(row["id"])])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as destination:
        for record in selected:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")

    sample_ids = [str(record["id"]) for record in selected]
    ids_path.write_text("\n".join(sample_ids) + "\n", encoding="utf-8")
    metadata = {
        "dataset": DATASET_ID,
        "dataset_config": "generation",
        "source_split": "test",
        "source_url": SOURCE_URL,
        "source_revision": DATASET_REVISION,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(source_path.resolve()),
        "source_sha256": sha256_bytes(source_bytes),
        "source_document_count": len(source_rows),
        "eligible_document_count": len(records),
        "eligible_rule": "at least one present or absent reference keyphrase",
        "reference_rule": "ordered union of extractive_keyphrases and abstractive_keyphrases",
        "document_conversion": "join tokens after restoring Penn Treebank bracket and quote markers",
        "sample_count": count,
        "sampling_seed": seed,
        "sampling_rule": "uniform sample without replacement, restored to source order",
        "sample_ids_in_order": sample_ids,
        "output_file": str(output_path.resolve()),
        "output_sha256": sha256_bytes(output_path.read_bytes()),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"原始测试文档：{len(source_rows)}")
    print(f"有参考关键词的文档：{len(records)}")
    print(f"固定子集：{count} 篇；seed={seed}")
    print(f"JSONL：{output_path.resolve()}")
    print(f"样本 ID：{ids_path.resolve()}")
    print(f"元数据：{metadata_path.resolve()}")
    print(f"源文件 SHA-256：{metadata['source_sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="准备固定版本的 MIDAS/Inspec 关键词数据")
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--source", type=Path, default=ROOT / "source" / "inspec_test.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "fixed_subset.jsonl")
    parser.add_argument("--metadata", type=Path, default=ROOT / "dataset_metadata.json")
    parser.add_argument("--ids", type=Path, default=ROOT / "fixed_subset_ids.txt")
    arguments = parser.parse_args()
    prepare_subset(
        count=arguments.count,
        seed=arguments.seed,
        source_path=arguments.source,
        output_path=arguments.output,
        metadata_path=arguments.metadata,
        ids_path=arguments.ids,
    )


if __name__ == "__main__":
    main()