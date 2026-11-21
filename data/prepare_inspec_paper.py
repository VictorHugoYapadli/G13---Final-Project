"""Build the complete Inspec test set with the original uncontrolled labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from prepare_inspec import document_text

ROOT = Path(__file__).resolve().parent
TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")
REFERENCE_URL = (
    "https://raw.githubusercontent.com/boudinfl/ake-datasets/"
    "master/datasets/Inspec/references/test.uncontr.json"
)
SOURCE_URL = (
    "https://huggingface.co/datasets/midas/inspec/resolve/"
    "9617780e99705df88a4cb239174c86b9d8d8300f/test.jsonl"
)


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def normalized(phrase: str) -> str:
    return " ".join(token.lower() for token in TOKEN_RE.findall(phrase))


def build_dataset(source_path: Path, references_path: Path, output_path: Path, metadata_path: Path) -> None:
    source_bytes = source_path.read_bytes()
    reference_bytes = references_path.read_bytes()
    source_rows = [
        json.loads(line)
        for line in source_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    references = json.loads(reference_bytes.decode("utf-8-sig"))

    source_by_id: dict[str, dict] = {}
    for row in source_rows:
        sample_id = str(row["paper_id"])
        if sample_id in source_by_id:
            raise ValueError(f"duplicate source document ID: {sample_id}")
        source_by_id[sample_id] = row

    if len(source_rows) != 500 or len(references) != 500:
        raise ValueError(
            f"expected 500 source documents and references; got {len(source_rows)} and {len(references)}"
        )
    source_ids = set(source_by_id)
    reference_ids = set(references)
    if source_ids != reference_ids:
        raise ValueError(
            f"source/reference ID mismatch: source-only={len(source_ids-reference_ids)}, "
            f"reference-only={len(reference_ids-source_ids)}"
        )

    output_rows: list[dict[str, object]] = []
    for row in source_rows:
        sample_id = str(row["paper_id"])
        original_labels = row.get("extractive_keyphrases", []) + row.get("abstractive_keyphrases", [])
        groups = references[sample_id]
        if any(not isinstance(group, list) or len(group) != 1 for group in groups):
            raise ValueError(f"unexpected alternative-label group format for document {sample_id}")
        uncontrolled = [group[0] for group in groups]
        source_set = {normalized(phrase) for phrase in original_labels if normalized(phrase)}
        reference_set = {normalized(phrase) for phrase in uncontrolled if normalized(phrase)}
        if source_set != reference_set:
            raise ValueError(f"MIDAS labels do not match official uncontrolled labels for {sample_id}")
        output_rows.append(
            {
                "id": sample_id,
                "text": document_text(row["document"]),
                "keywords": uncontrolled,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows).encode("utf-8")
    output_path.write_bytes(output_content)
    metadata = {
        "dataset": "Inspec",
        "source_dataset": "midas/inspec generation test split",
        "source_revision": "9617780e99705df88a4cb239174c86b9d8d8300f",
        "source_url": SOURCE_URL,
        "source_file": str(source_path.resolve()),
        "source_sha256": sha256(source_bytes),
        "reference_source": "boudinfl/ake-datasets Inspec test.uncontr.json",
        "reference_url": REFERENCE_URL,
        "reference_file": str(references_path.resolve()),
        "reference_sha256": sha256(reference_bytes),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "test",
        "document_count": len(output_rows),
        "sampling": "none; all 500 test documents in source order",
        "label_rule": "original uncontrolled keyphrases only",
        "label_validation": "all 500 IDs and normalized label sets match MIDAS present+absent labels",
        "sample_ids_in_order": [row["id"] for row in output_rows],
        "output_file": str(output_path.resolve()),
        "output_sha256": sha256(output_content),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Documents: {len(output_rows)}")
    print("Uncontrolled-label validation: 500/500 IDs and label sets match")
    print(f"Output file written: {output_path.name}")
    print(f"SHA-256: {metadata['output_sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "source" / "inspec_test.jsonl")
    parser.add_argument("--references", type=Path, default=ROOT / "source" / "inspec_test_uncontrolled.json")
    parser.add_argument("--output", type=Path, default=ROOT / "inspec_paper_test.jsonl")
    parser.add_argument("--metadata", type=Path, default=ROOT / "inspec_paper_metadata.json")
    args = parser.parse_args()
    build_dataset(args.source, args.references, args.output, args.metadata)


if __name__ == "__main__":
    main()
