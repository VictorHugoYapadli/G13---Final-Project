"""固定子集上的基线、TextRank 消融与证据导出。"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import platform
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .candidates import (
    Candidate,
    eligible_mask,
    exclude_single_character_tokens,
    extract_candidates,
    merge_selected_words,
)
from .dataio import load_jsonl, sha256_file
from .graph import rank_words
from .metrics import candidate_coverage, document_metrics
from .text import (
    normalize_phrase,
    phrase_boundaries,
    spacy_model_info,
    tag_text,
    tag_text_spacy,
)

LOGGER = logging.getLogger("textrank_repro")


def _read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        config = json.load(source)
    required_sections = {"dataset", "preprocessing", "textrank", "evaluation"}
    missing = required_sections - config.keys()
    if missing:
        raise ValueError(f"Configuration is missing required sections: {', '.join(sorted(missing))}")
    return config


def _tfidf_scores(prepared: list[dict[str, Any]]) -> list[dict[str, float]]:
    document_frequency: Counter[str] = Counter()
    for item in prepared:
        document_frequency.update(item["word_counts"].keys())

    document_count = len(prepared)
    results: list[dict[str, float]] = []
    for item in prepared:
        scores = {}
        for word, count in item["word_counts"].items():
            inverse_frequency = math.log((1 + document_count) / (1 + document_frequency[word])) + 1.0
            scores[word] = count * inverse_frequency
        results.append(scores)
    return results


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_experiment(
    config_path: Path,
    data_path: Path,
    output_dir: Path,
    limit: int | None = None,
) -> Path:
    """Run all configured comparisons and return the evidence directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()

    LOGGER.info("Starting TextRank experiment.")
    LOGGER.info("Loading configuration: %s", config_path.resolve())
    config = _read_config(config_path)
    LOGGER.info("Loading dataset: %s", data_path.resolve())
    records = load_jsonl(data_path)
    LOGGER.info("Loaded %d documents.", len(records))
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be a positive integer.")
        records = records[:limit]
    else:
        minimum = config["dataset"]["expected_min_documents"]
        maximum = config["dataset"]["expected_max_documents"]
        if not minimum <= len(records) <= maximum:
            raise ValueError(
                f"A full run requires {minimum}–{maximum} documents; got {len(records)}. "
                "Use --limit for a separately saved smoke test."
            )

    LOGGER.info(
        "Validated %d documents; smoke-test limit: %s.",
        len(records),
        limit if limit is not None else "none",
    )
    LOGGER.info("Evidence will be written to: %s", output_dir.resolve())

    preprocessing_config = config["preprocessing"]
    pos_tagger = preprocessing_config.get("pos_tagger", "nltk")
    if pos_tagger not in {"nltk", "spacy"}:
        raise ValueError("preprocessing.pos_tagger must be either 'nltk' or 'spacy'.")
    spacy_model = preprocessing_config.get("spacy_model", "en_core_web_sm")
    if pos_tagger == "spacy" and not isinstance(spacy_model, str):
        raise ValueError("preprocessing.spacy_model must be a model name string.")
    allowed_tags = set(preprocessing_config["pos_tags"])
    edge_weighting = config["textrank"].get("edge_weighting", "binary")
    if edge_weighting not in {"binary", "distance_decay"}:
        raise ValueError("textrank.edge_weighting must be either 'binary' or 'distance_decay'.")
    prepared: list[dict[str, Any]] = []
    LOGGER.info(
        "Preprocessing documents with the %s POS tagger (%s).",
        pos_tagger,
        spacy_model if pos_tagger == "spacy" else "NLTK English tagger",
    )
    preprocessing_started = time.perf_counter()
    progress_interval = max(1, math.ceil(len(records) / 10))
    for document_index, record in enumerate(records, start=1):
        if pos_tagger == "spacy":
            tagged = tag_text_spacy(record["text"], spacy_model)
        else:
            tagged = tag_text(record["text"])
        tokens = [token for token, _ in tagged]
        break_before = phrase_boundaries(record["text"])
        if len(tokens) != len(break_before):
            raise ValueError(
                f"Token count and phrase-boundary count do not match for sample {record['id']}."
            )
        keep = eligible_mask(tagged, allowed_tags)
        no_single_character = exclude_single_character_tokens(tokens, keep)
        candidates = extract_candidates(tokens, keep, break_before)
        word_counts = Counter(token for token, is_eligible in zip(tokens, keep) if is_eligible)
        prepared.append(
            {
                "id": record["id"],
                "text": record["text"],
                "gold": record["keywords"],
                "tokens": tokens,
                "break_before": break_before,
                "eligible": keep,
                "no_single_character_eligible": no_single_character,
                "all_tokens_eligible": [True] * len(tokens),
                "candidates": candidates,
                "no_single_character_candidates": extract_candidates(
                    tokens, no_single_character, break_before
                ),
                "all_token_candidates": extract_candidates(
                    tokens, [True] * len(tokens), break_before
                ),
                "word_counts": word_counts,
            }
        )
        if document_index % progress_interval == 0 or document_index == len(records):
            LOGGER.info(
                "Preprocessing progress: %d/%d documents (%.0f%%).",
                document_index,
                len(records),
                100 * document_index / len(records),
            )
    preprocessing_seconds = time.perf_counter() - preprocessing_started
    LOGGER.info("Preprocessing complete in %.2f seconds.", preprocessing_seconds)

    methods = ["tfidf"] + [f"textrank_w{window}" for window in config["textrank"]["windows"]]
    ablation = config.get("ablations", {}).get("no_pos_filter", {})
    ablation_enabled = bool(ablation.get("enabled", False))
    ablation_method = None
    if ablation_enabled:
        ablation_window = int(ablation.get("comparison_window", 2))
        if ablation_window not in config["textrank"]["windows"]:
            raise ValueError("no_pos_filter comparison_window must also appear in textrank.windows")
        ablation_method = f"textrank_w{ablation_window}_no_pos_filter"
        methods.append(ablation_method)
    single_character_ablation = config.get("ablations", {}).get(
        "exclude_single_character_tokens", {}
    )
    single_character_ablation_enabled = bool(single_character_ablation.get("enabled", False))
    single_character_method = None
    if single_character_ablation_enabled:
        single_character_window = int(single_character_ablation.get("comparison_window", 2))
        if single_character_window not in config["textrank"]["windows"]:
            raise ValueError(
                "exclude_single_character_tokens comparison_window must also appear "
                "in textrank.windows"
            )
        single_character_method = f"textrank_w{single_character_window}_no_single_char"
        methods.append(single_character_method)
    per_document: list[dict[str, Any]] = []
    method_seconds = Counter()
    tfidf_per_document = _tfidf_scores(prepared)
    ranking_details: dict[str, list[tuple[list[tuple[Candidate, float]], int | None, bool | None]]] = {
        method: [] for method in methods
    }

    LOGGER.info(
        "Ranking keywords for %d documents using %d methods: %s.",
        len(prepared),
        len(methods),
        ", ".join(methods),
    )
    ranking_progress_interval = max(1, math.ceil(len(prepared) / 10))
    for index, item in enumerate(prepared):
        document_methods: dict[str, list[tuple[Candidate, float]]] = {}
        for method, word_scores in [("tfidf", tfidf_per_document[index])]:
            start = time.perf_counter()
            ranked = merge_selected_words(
                item["tokens"],
                item["eligible"],
                word_scores,
                top_fraction=config["textrank"]["top_word_fraction"],
                break_before=item["break_before"],
            )
            method_seconds[method] += time.perf_counter() - start
            document_methods[method] = ranked
            ranking_details[method].append((ranked, None, None))

        for window in config["textrank"]["windows"]:
            method = f"textrank_w{window}"
            start = time.perf_counter()
            word_scores, iterations, converged = rank_words(
                item["tokens"],
                item["eligible"],
                window=window,
                damping=config["textrank"]["damping"],
                tolerance=config["textrank"]["tolerance"],
                max_iterations=config["textrank"]["max_iterations"],
                edge_weighting=edge_weighting,
            )
            ranked = merge_selected_words(
                item["tokens"],
                item["eligible"],
                word_scores,
                top_fraction=config["textrank"]["top_word_fraction"],
                break_before=item["break_before"],
            )
            method_seconds[method] += time.perf_counter() - start
            document_methods[method] = ranked
            ranking_details[method].append((ranked, iterations, converged))

        if ablation_method is not None:
            start = time.perf_counter()
            ablation_scores, iterations, converged = rank_words(
                item["tokens"],
                item["all_tokens_eligible"],
                window=ablation_window,
                damping=config["textrank"]["damping"],
                tolerance=config["textrank"]["tolerance"],
                max_iterations=config["textrank"]["max_iterations"],
                edge_weighting=edge_weighting,
            )
            ablation_ranked = merge_selected_words(
                item["tokens"],
                item["all_tokens_eligible"],
                ablation_scores,
                top_fraction=config["textrank"]["top_word_fraction"],
                break_before=item["break_before"],
            )
            method_seconds[ablation_method] += time.perf_counter() - start
            document_methods[ablation_method] = ablation_ranked
            ranking_details[ablation_method].append((ablation_ranked, iterations, converged))

        if single_character_method is not None:
            start = time.perf_counter()
            single_character_scores, iterations, converged = rank_words(
                item["tokens"],
                item["no_single_character_eligible"],
                window=single_character_window,
                damping=config["textrank"]["damping"],
                tolerance=config["textrank"]["tolerance"],
                max_iterations=config["textrank"]["max_iterations"],
                edge_weighting=edge_weighting,
            )
            single_character_ranked = merge_selected_words(
                item["tokens"],
                item["no_single_character_eligible"],
                single_character_scores,
                top_fraction=config["textrank"]["top_word_fraction"],
                break_before=item["break_before"],
            )
            method_seconds[single_character_method] += time.perf_counter() - start
            document_methods[single_character_method] = single_character_ranked
            ranking_details[single_character_method].append(
                (single_character_ranked, iterations, converged)
            )

        coverage, matched, gold_count = candidate_coverage(
            (candidate.phrase for candidate in item["candidates"]), item["gold"]
        )
        single_character_coverage, single_character_matched, single_character_gold_count = (
            candidate_coverage(
                (
                    candidate.phrase
                    for candidate in item["no_single_character_candidates"]
                ),
                item["gold"],
            )
        )
        all_coverage, all_matched, all_gold_count = candidate_coverage(
            (candidate.phrase for candidate in item["all_token_candidates"]), item["gold"]
        )
        coverage_by_method = {
            method: (all_coverage, all_matched, all_gold_count)
            if method == ablation_method
            else (
                single_character_coverage,
                single_character_matched,
                single_character_gold_count,
            )
            if method == single_character_method
            else (coverage, matched, gold_count)
            for method in methods
        }
        candidate_count_by_method = {
            method: len(item["all_token_candidates"])
            if method == ablation_method
            else len(item["no_single_character_candidates"])
            if method == single_character_method
            else len(item["candidates"])
            for method in methods
        }
        per_document.append(
            {
                "sample_id": item["id"],
                "gold_keywords": item["gold"],
                "candidate_count": len(item["candidates"]),
                "candidate_count_by_method": candidate_count_by_method,
                "candidate_coverage": coverage,
                "candidate_gold_matched": matched,
                "gold_count": gold_count,
                "candidate_coverage_by_method": {
                    method: {
                        "coverage": values[0],
                        "matched": values[1],
                        "gold_count": values[2],
                    }
                    for method, values in coverage_by_method.items()
                },
                "predictions": {
                    method: [candidate.phrase for candidate, _ in ranked]
                    for method, ranked in document_methods.items()
                },
                "top_scores": {
                    method: [{"phrase": candidate.phrase, "score": score} for candidate, score in ranked[:10]]
                    for method, ranked in document_methods.items()
                },
                "ranking_diagnostics": {
                    method: {
                        "iterations": ranking_details[method][index][1],
                        "converged": ranking_details[method][index][2],
                    }
                    for method in methods
                    if ranking_details[method][index][1] is not None
                },
            }
        )
        completed_documents = index + 1
        if (
            completed_documents % ranking_progress_interval == 0
            or completed_documents == len(prepared)
        ):
            LOGGER.info(
                "Ranking progress: %d/%d documents (%.0f%%).",
                completed_documents,
                len(prepared),
                100 * completed_documents / len(prepared),
            )

    metric_rows: list[dict[str, Any]] = []
    LOGGER.info("Calculating evaluation metrics for %d methods.", len(methods))
    for method in methods:
        for cutoff in config["evaluation"]["cutoffs"]:
            values = [
                document_metrics(row["predictions"][method], row["gold_keywords"], cutoff)
                for row in per_document
            ]
            metric_rows.append(
                {
                    "method": method,
                    "cutoff": cutoff,
                    "precision_macro": _mean([value["precision"] for value in values]),
                    "recall_macro": _mean([value["recall"] for value in values]),
                    "f1_macro": _mean([value["f1"] for value in values]),
                    "candidate_coverage_macro": _mean(
                        [row["candidate_coverage_by_method"][method]["coverage"] for row in per_document]
                    ),
                    "candidate_coverage_micro": sum(
                        row["candidate_coverage_by_method"][method]["matched"] for row in per_document
                    )
                    / max(1, sum(row["candidate_coverage_by_method"][method]["gold_count"] for row in per_document)),
                    "candidate_count_mean": _mean(
                        [float(row["candidate_count_by_method"][method]) for row in per_document]
                    ),
                    "ranking_seconds_total": method_seconds[method],
                    "ranking_seconds_per_document": method_seconds[method] / len(per_document),
                    "documents": len(per_document),
                }
            )

        if config["evaluation"].get("paper_all_predictions", False):
            assigned_total = sum(len(row["predictions"][method]) for row in per_document)
            references_total = sum(
                len({normalize_phrase(item) for item in row["gold_keywords"] if normalize_phrase(item)})
                for row in per_document
            )
            correct_total = sum(
                int(
                    document_metrics(
                        row["predictions"][method],
                        row["gold_keywords"],
                        len(row["predictions"][method]),
                    )["correct"]
                )
                for row in per_document
            )
            precision_micro = correct_total / assigned_total if assigned_total else 0.0
            recall_micro = correct_total / references_total if references_total else 0.0
            f1_micro = (
                2 * precision_micro * recall_micro / (precision_micro + recall_micro)
                if precision_micro + recall_micro
                else 0.0
            )
            metric_rows.append(
                {
                    "method": method,
                    "cutoff": "all",
                    "precision_macro": "",
                    "recall_macro": "",
                    "f1_macro": "",
                    "precision_micro": precision_micro,
                    "recall_micro": recall_micro,
                    "f1_micro": f1_micro,
                    "assigned_total": assigned_total,
                    "assigned_mean": assigned_total / len(per_document),
                    "correct_total": correct_total,
                    "correct_mean": correct_total / len(per_document),
                    "references_total": references_total,
                    "candidate_coverage_macro": _mean(
                        [row["candidate_coverage_by_method"][method]["coverage"] for row in per_document]
                    ),
                    "candidate_coverage_micro": sum(
                        row["candidate_coverage_by_method"][method]["matched"] for row in per_document
                    )
                    / max(1, sum(row["candidate_coverage_by_method"][method]["gold_count"] for row in per_document)),
                    "candidate_count_mean": _mean(
                        [float(row["candidate_count_by_method"][method]) for row in per_document]
                    ),
                    "ranking_seconds_total": method_seconds[method],
                    "ranking_seconds_per_document": method_seconds[method] / len(per_document),
                    "documents": len(per_document),
                }
            )

    LOGGER.info("Writing predictions.jsonl and metrics.csv.")
    _write_jsonl(output_dir / "predictions.jsonl", per_document)
    _write_csv(output_dir / "metrics.csv", metric_rows)

    seed = int(config["evaluation"]["case_seed"])
    case_count = min(int(config["evaluation"]["case_count"]), len(records))
    case_indices = set(random.Random(seed).sample(range(len(records)), case_count))
    include_text = bool(config["evaluation"].get("include_text_in_cases", False))
    excerpt_length = int(config["evaluation"].get("text_excerpt_characters", 500))
    cases = []
    for index, row in enumerate(per_document):
        if index not in case_indices:
            continue
        case = dict(row)
        case["case_seed"] = seed
        if include_text:
            case["text_excerpt"] = prepared[index]["text"][:excerpt_length]
        cases.append(case)
    LOGGER.info("Writing %d sampled cases to cases_10.jsonl.", len(cases))
    _write_jsonl(output_dir / "cases_10.jsonl", cases)

    duration = time.perf_counter() - started
    metadata = {
        "experiment_name": config.get("experiment_name", "experiment"),
        "code_version": __version__,
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "code_files_sha256": {
            source.name: sha256_file(source)
            for source in [
                Path(__file__).with_name("__init__.py"),
                Path(__file__).with_name("candidates.py"),
                Path(__file__).with_name("dataio.py"),
                Path(__file__).with_name("experiment.py"),
                Path(__file__).with_name("graph.py"),
                Path(__file__).with_name("metrics.py"),
                Path(__file__).with_name("text.py"),
                Path(__file__).parents[1] / "run_experiment.py",
            ]
        },
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration,
        "preprocessing_seconds": preprocessing_seconds,
        "document_count": len(records),
        "smoke_limit": limit,
        "data_path": str(data_path.resolve()),
        "data_sha256": sha256_file(data_path),
        "sample_ids_in_order": [record["id"] for record in records],
        "case_sample_ids": [cases_item["sample_id"] for cases_item in cases],
        "configuration": config,
        "python_version": platform.python_version(),
        "nltk_version": _nltk_version(),
        "pos_tagger_resource": _pos_tagger_resource(),
        "spacy_model_info": (
            spacy_model_info(spacy_model) if pos_tagger == "spacy" else None
        ),
        "method_ranking_seconds": dict(method_seconds),
        "notes": "无随机算法；evaluation.case_seed 仅固定案例样本 ID，不控制数据子集。",
    }
    LOGGER.info("Writing run_metadata.json and finalizing evidence files.")
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as destination:
        json.dump(metadata, destination, ensure_ascii=False, indent=2)
        destination.write("\n")

    LOGGER.info(
        "Experiment finished in %.2f seconds. Evidence directory: %s",
        duration,
        output_dir.resolve(),
    )
    LOGGER.removeHandler(handler)
    handler.close()
    return output_dir


def _nltk_version() -> str | None:
    try:
        import nltk
    except ImportError:
        return None
    return nltk.__version__


def _pos_tagger_resource() -> dict[str, Any] | None:
    """Fingerprint the NLTK English tagger data used by ``pos_tag``."""
    try:
        import nltk

        resource = Path(str(nltk.data.find("taggers/averaged_perceptron_tagger_eng/")))
    except (ImportError, LookupError):
        return None
    files = sorted(resource.glob("*"))
    return {
        "resource": "averaged_perceptron_tagger_eng",
        "files_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
            if path.is_file()
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
