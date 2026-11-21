"""TextRank 关键词对照实验命令行入口。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from textrank_repro.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TextRank keyword extraction experiment")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment_paper_aligned.json"))
    parser.add_argument("--data", type=Path, default=Path("data/inspec_paper_test.jsonl"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, help="Run only the first N documents as a smoke test")
    arguments = parser.parse_args()

    output_dir = arguments.output
    if output_dir is None:
        from datetime import datetime

        run_id = datetime.now().strftime("textrank_%Y%m%d_%H%M%S")
        output_dir = Path("results") / run_id

    logger = logging.getLogger("textrank_repro")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)
    try:
        output_dir = run_experiment(arguments.config, arguments.data, output_dir, arguments.limit)
    finally:
        logger.removeHandler(console_handler)
    print(f"Experiment complete. Evidence directory: {output_dir}")


if __name__ == "__main__":
    main()
