from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from textrank_repro.candidates import (
    exclude_single_character_tokens,
    extract_candidates,
    merge_selected_words,
)
from textrank_repro.experiment import run_experiment
from textrank_repro.graph import rank_words
from textrank_repro.metrics import candidate_coverage, document_metrics
from textrank_repro.text import normalize_phrase, phrase_boundaries, tokenize
from data.prepare_inspec import document_text


class TextRankCoreTests(unittest.TestCase):
    def test_graph_scores_connected_and_isolated_nodes(self) -> None:
        scores, iterations, converged = rank_words(
            ["linear", "equations", "the", "isolated"], [True, True, False, True], window=2
        )
        self.assertTrue(converged)
        self.assertGreaterEqual(iterations, 1)
        self.assertAlmostEqual(scores["linear"], scores["equations"], places=6)
        self.assertLess(scores["isolated"], scores["linear"])

    def test_graph_window_counts_original_token_positions(self) -> None:
        tokens = ["alpha", "the", "beta"]
        eligible = [True, False, True]

        adjacent_scores, _, _ = rank_words(tokens, eligible, window=2)
        wider_scores, _, _ = rank_words(tokens, eligible, window=3)

        self.assertAlmostEqual(adjacent_scores["alpha"], 0.15)
        self.assertAlmostEqual(adjacent_scores["beta"], 0.15)
        self.assertAlmostEqual(wider_scores["alpha"], 1.0)
        self.assertAlmostEqual(wider_scores["beta"], 1.0)

    def test_graph_ranking_is_repeatable(self) -> None:
        arguments = (["alpha", "beta", "gamma", "delta", "alpha"], [True] * 5)
        first = rank_words(*arguments, window=3)
        second = rank_words(*arguments, window=3)
        self.assertEqual(first, second)

    def test_distance_decay_weights_longer_cooccurrences_less(self) -> None:
        tokens = ["alpha", "beta", "gamma"]
        eligible = [True] * 3

        binary_scores, _, _ = rank_words(tokens, eligible, window=3)
        decay_scores, _, _ = rank_words(
            tokens, eligible, window=3, edge_weighting="distance_decay"
        )
        adjacent_decay_scores, _, _ = rank_words(
            tokens, eligible, window=2, edge_weighting="distance_decay"
        )

        self.assertAlmostEqual(binary_scores["alpha"], binary_scores["beta"])
        self.assertGreater(decay_scores["beta"], decay_scores["alpha"])
        self.assertEqual(
            adjacent_decay_scores,
            rank_words(tokens, eligible, window=2)[0],
        )

    def test_unknown_edge_weighting_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rank_words(["alpha"], [True], window=2, edge_weighting="unknown")

    def test_no_candidate_words_returns_empty_scores(self) -> None:
        self.assertEqual(rank_words(["the"], [False], window=2), ({}, 0, True))

    def test_candidate_phrases_only_join_adjacent_eligible_tokens(self) -> None:
        candidates = extract_candidates(
            ["linear", "and", "diophantine", "equations"], [True, False, True, True]
        )
        self.assertEqual(
            {candidate.phrase for candidate in candidates},
            {"linear", "diophantine", "equations", "diophantine equations"},
        )

    def test_single_character_ablation_preserves_longer_eligible_tokens(self) -> None:
        self.assertEqual(
            exclude_single_character_tokens(
                ["x", "eu", "x-ray", "system"],
                [True, True, True, False],
            ),
            [False, True, True, False],
        )

    def test_phrase_merging_does_not_cross_punctuation(self) -> None:
        text = "linear equations, graph method"
        tokens = tokenize(text)
        boundaries = phrase_boundaries(text)
        scores = {"equations": 4.0, "graph": 3.0, "linear": 2.0, "method": 1.0}
        ranked = merge_selected_words(tokens, [True] * len(tokens), scores, top_fraction=0.5, break_before=boundaries)
        self.assertEqual([candidate.phrase for candidate, _ in ranked], ["equations", "graph"])
        candidates = extract_candidates(tokens, [True] * len(tokens), boundaries)
        self.assertNotIn("equations graph", {candidate.phrase for candidate in candidates})

    def test_top_word_selection_merges_adjacent_occurrences(self) -> None:
        tokens = ["linear", "equations", "method", "graph", "ranking", "algorithm"]
        scores = {"linear": 6.0, "equations": 5.0, "method": 4.0, "graph": 3.0, "ranking": 2.0, "algorithm": 1.0}
        ranked = merge_selected_words(tokens, [True] * len(tokens), scores, top_fraction=1 / 3)
        self.assertEqual([candidate.phrase for candidate, _ in ranked], ["linear equations"])

    def test_equal_word_scores_use_first_text_occurrence(self) -> None:
        tokens = ["zeta", "alpha", "mu", "beta", "zeta"]
        scores = {"alpha": 1.0, "beta": 1.0, "mu": 1.0, "zeta": 1.0}
        ranked = merge_selected_words(tokens, [True] * len(tokens), scores, top_fraction=0.5)
        self.assertEqual(
            {candidate.phrase for candidate, _ in ranked},
            {"zeta alpha", "zeta"},
        )

    def test_phrase_score_sums_constituent_word_scores(self) -> None:
        ranked = merge_selected_words(
            ["linear", "equations"],
            [True, True],
            {"linear": 2.0, "equations": 3.0},
            top_fraction=1.0,
        )
        self.assertEqual(ranked[0][0].phrase, "linear equations")
        self.assertEqual(ranked[0][1], 5.0)

    def test_inspec_treebank_markers_do_not_become_word_tokens(self) -> None:
        self.assertEqual(document_text(["A", "-LRB-", "test", "-RRB-"]), "A ( test )")

    def test_metrics_use_stemming_and_fixed_precision_denominator(self) -> None:
        self.assertEqual(normalize_phrase("Systems"), normalize_phrase("system"))
        values = document_metrics(["systems"], ["system", "linear equations"], cutoff=5)
        self.assertAlmostEqual(values["precision"], 0.2)
        self.assertAlmostEqual(values["recall"], 0.5)
        self.assertAlmostEqual(values["f1"], 2 / 7)

    def test_candidate_coverage(self) -> None:
        coverage, matched, total = candidate_coverage(
            ["linear equations"], ["linear equations", "missing phrase"]
        )
        self.assertEqual((coverage, matched, total), (0.5, 1, 2))

    def test_smoke_run_writes_auditable_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_path = root / "data.jsonl"
            config_path = root / "config.json"
            output_path = root / "output"
            samples = [
                {"id": f"doc-{index}", "text": "linear and diophantine equations", "keywords": ["linear equations"]}
                for index in range(3)
            ]
            data_path.write_text("".join(json.dumps(item) + "\n" for item in samples), encoding="utf-8")
            config = {
                "experiment_name": "unit_test",
                "dataset": {"expected_min_documents": 1, "expected_max_documents": 5},
                "preprocessing": {"pos_tags": ["JJ", "NN"]},
                "textrank": {"windows": [2, 4, 8], "top_word_fraction": 1 / 3, "damping": 0.85, "tolerance": 0.0001, "max_iterations": 100},
                "ablations": {
                    "no_pos_filter": {"enabled": True, "comparison_window": 2},
                    "exclude_single_character_tokens": {
                        "enabled": True,
                        "comparison_window": 2,
                    },
                },
                "evaluation": {"cutoffs": [5, 10], "case_count": 2, "case_seed": 7, "include_text_in_cases": False},
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")

            def fake_tag_text(_text: str) -> list[tuple[str, str]]:
                return [("linear", "JJ"), ("and", "CC"), ("diophantine", "JJ"), ("equations", "NN")]

            with patch("textrank_repro.experiment.tag_text", side_effect=fake_tag_text):
                run_experiment(config_path, data_path, output_path)

            for filename in ("run.log", "run_metadata.json", "metrics.csv", "predictions.jsonl", "cases_10.jsonl"):
                self.assertTrue((output_path / filename).is_file(), filename)
            metadata = json.loads((output_path / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(len(metadata["config_sha256"]), 64)
            self.assertIn("graph.py", metadata["code_files_sha256"])
            prediction_lines = (output_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(prediction_lines), 3)
            self.assertEqual(len((output_path / "cases_10.jsonl").read_text(encoding="utf-8").splitlines()), 2)
            predictions = [json.loads(line) for line in prediction_lines]
            self.assertIn("textrank_w2_no_pos_filter", predictions[0]["predictions"])
            self.assertIn("textrank_w2_no_single_char", predictions[0]["predictions"])
            self.assertGreater(
                predictions[0]["candidate_count_by_method"]["textrank_w2_no_pos_filter"],
                predictions[0]["candidate_count_by_method"]["textrank_w2"],
            )
            metric_text = (output_path / "metrics.csv").read_text(encoding="utf-8-sig")
            self.assertIn("textrank_w2_no_pos_filter", metric_text)
            self.assertIn("textrank_w2_no_single_char", metric_text)

            variant_config_path = root / "variant_config.json"
            variant_output_path = root / "variant_output"
            config["preprocessing"]["pos_tagger"] = "spacy"
            config["textrank"]["edge_weighting"] = "distance_decay"
            variant_config_path.write_text(json.dumps(config), encoding="utf-8")
            with (
                patch(
                    "textrank_repro.experiment.tag_text_spacy",
                    side_effect=lambda text, _model: fake_tag_text(text),
                ),
                patch(
                    "textrank_repro.experiment.spacy_model_info",
                    return_value={
                        "spacy_version": "test",
                        "model_name": "test-model",
                        "model_version": "test",
                    },
                ),
            ):
                run_experiment(variant_config_path, data_path, variant_output_path)

            variant_metadata = json.loads(
                (variant_output_path / "run_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(variant_metadata["configuration"]["preprocessing"]["pos_tagger"], "spacy")
            self.assertEqual(
                variant_metadata["configuration"]["textrank"]["edge_weighting"],
                "distance_decay",
            )
            self.assertEqual(
                variant_metadata["spacy_model_info"]["model_name"],
                "test-model",
            )


if __name__ == "__main__":
    unittest.main()
