"""Portable tests: python -m unittest discover -s tests -v."""
import ast
import itertools
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "quantitative_metrics"))
import metrics as m
from metrics import (
    alpha_ndcg_at_k,
    evaluate_ranked_results,
    ndcg_at_k,
    normalized_entropy_at_k,
)


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.array(["A", "A", "A", "B"])
        self.batches = np.array(["x", "y", "x", "y"])

    def alpha(self, labels, batches, **kwargs):
        return m.compute_alpha_ndcg_batch(
            ["A"], [labels], [batches], reference_labels=self.labels,
            reference_batches=self.batches, **kwargs)[0][0]

    def test_ndcg_perfect(self):
        self.assertEqual(m.ndcg(["A"], [["A", "A"]], 2, reference_labels=self.labels)[1], 1)

    def test_ndcg_reference_denominator_not_observed_hits(self):
        result = m.ndcg(["A"], [["A", "B"]], 2, reference_labels=self.labels)[1]
        self.assertAlmostEqual(result, 1 / (1 + 1 / math.log2(3)))

    def test_ndcg_rank_discount(self):
        result = m.ndcg(["A"], [["B", "A"]], 2, reference_labels=self.labels)[1]
        self.assertAlmostEqual(result, (1 / math.log2(3)) / (1 + 1 / math.log2(3)))

    def test_ndcg_rare_type(self):
        self.assertEqual(m.ndcg(["B"], [["B", "A"]], 2, reference_labels=self.labels)[1], 1)

    def test_no_correct_hits_zero(self):
        self.assertEqual(m.ndcg(["A"], [["B"]], 1, reference_labels=self.labels)[1], 0)
        self.assertEqual(self.alpha(["B"], ["y"], k=1), 0)

    def test_absent_reference_type_nan_excluded(self):
        values, mean = m.ndcg(["C", "A"], [["A"], ["A"]], 1, reference_labels=self.labels)
        self.assertTrue(np.isnan(values[0]))
        self.assertEqual(mean, 1)
        result = m.compute_batch_diversity_metrics(
            ["C", "A"], [["A"], ["A"]], [["x"], ["x"]],
            reference_labels=self.labels, reference_batches=self.batches, k=1)
        self.assertEqual(result["nDCG_valid_fraction"], 0.5)
        self.assertEqual(result["α-NDCG_valid_fraction"], 0.5)
        self.assertEqual(result["α-NDCG_gini"], 0)

    def test_all_absent_gini_zero_with_undefined_ndcg(self):
        result = m.compute_balanced_metrics(["C"], [["A"]], k=1, reference_labels=self.labels)
        self.assertEqual(result["nDCG_gini"], 0)
        self.assertEqual(result["nDCG_celltype_gini"], 0)

    def test_entropy_balanced_one(self):
        self.assertEqual(m.compute_entropy_batch([["x", "y"]], reference_batches=self.batches, k=2)[1], 1)

    def test_entropy_single_batch_zero(self):
        self.assertEqual(m.compute_entropy_batch([["x", "x"]], reference_batches=["x", "x"], k=2)[1], 0)

    def test_entropy_uses_all_reference_batches(self):
        self.assertAlmostEqual(m.compute_entropy_batch([["x", "y"]], reference_batches=["x", "y", "z", "w"], k=2)[1], 0.5)

    def test_entropy_not_conditioned_on_correctness(self):
        result = m.compute_batch_diversity_metrics(
            ["C"], [["A", "B"]], [["x", "y"]],
            reference_labels=self.labels, reference_batches=self.batches, k=2)
        self.assertEqual(result["Average Entropy"], 1)
        self.assertTrue(np.isnan(result["Average α-NDCG"]))

    def test_alpha_perfect_diverse(self):
        self.assertEqual(self.alpha(["A", "A", "A"], ["x", "y", "x"], k=3), 1)

    def test_alpha_penalizes_repeated_correct_batch(self):
        diverse = self.alpha(["A", "A"], ["x", "y"], k=2)
        repeated = self.alpha(["A", "A"], ["x", "x"], k=2)
        self.assertLess(repeated, diverse)

    def test_wrong_cell_does_not_consume_novelty(self):
        result = self.alpha(["B", "A"], ["y", "y"], k=2)
        self.assertAlmostEqual(result, (1 / math.log2(3)) / (1 + 1 / math.log2(3)))

    def test_alpha_zero_equals_ndcg(self):
        a = self.alpha(["B", "A", "A"], ["y", "x", "y"], k=3, alpha=0)
        n = m.ndcg(["A"], [["B", "A", "A"]], 3, reference_labels=self.labels)[1]
        self.assertEqual(a, n)

    def test_alpha_finite_inventory_ideal_exhaustive(self):
        # Exhaustively verify the ideal against all permutations in a small pool.
        labels = np.array(["A"] * 5 + ["B"])
        batches = np.array(["x", "x", "x", "y", "z", "z"])
        for alpha in (0, 0.2, 0.5, 0.9):
            scores = [
                alpha_ndcg_at_k("A", labels[list(p)], batches[list(p)], labels, batches, k=4, alpha=alpha)
                for p in itertools.permutations(range(6), 4)
            ]
            self.assertAlmostEqual(max(scores), 1)
            self.assertGreaterEqual(min(scores), 0)

    def test_cutoff_ignores_later_ranks(self):
        self.assertEqual(self.alpha(["A", "B"], ["x", "y"], k=1), 1)
        self.assertEqual(m.ndcg(["A"], [["A", "B"]], 1, reference_labels=self.labels)[1], 1)
        self.assertEqual(m.compute_entropy_batch([["x", "y"]], reference_batches=self.batches, k=1)[1], 0)

    def test_fewer_candidates_than_k(self):
        self.assertEqual(self.alpha(self.labels, self.batches, k=50),
                         alpha_ndcg_at_k("A", self.labels, self.batches, self.labels, self.batches, k=50))
        self.assertEqual(m.ndcg(["A"], [["A"]], 50, reference_labels=["A"])[1], 1)

    def test_invalid_alpha_including_old_positional_k(self):
        for alpha in (-0.1, 1, 50, np.nan, np.inf):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                self.alpha(["A"], ["x"], k=1, alpha=alpha)

    def test_invalid_cutoff(self):
        for k in (0, -1, 1.5, True):
            with self.subTest(k=k), self.assertRaises(ValueError):
                m.ndcg(["A"], [["A"]], k, reference_labels=self.labels)

    def test_missing_reference_is_error(self):
        with self.assertRaisesRegex(ValueError, "complete eligible reference"):
            m.ndcg(["A"], [["A"]])
        with self.assertRaises(ValueError):
            m.compute_entropy_batch([["x"]])
        with self.assertRaises(ValueError):
            m.compute_alpha_ndcg_batch(["A"], [["A"]], [["x"]])

    def test_invalid_reference_pool_is_not_silently_augmented(self):
        with self.assertRaisesRegex(ValueError, "reference pool"):
            self.alpha(["B", "B"], ["y", "y"], k=2)
        with self.assertRaises(ValueError):
            m.ndcg(["B"], [["B", "B"]], 2, reference_labels=self.labels)
        with self.assertRaises(ValueError):
            m.compute_entropy_batch([["unknown"]], reference_batches=self.batches)

    def test_invalid_shapes_and_missing_values(self):
        with self.assertRaises(ValueError):
            self.alpha(["A", "A"], ["x"], k=2)
        with self.assertRaises(ValueError):
            m.ndcg(["A", "A"], [["A"]], 1, reference_labels=self.labels)
        with self.assertRaises(ValueError):
            m.ndcg(["A"], [[None]], 1, reference_labels=self.labels)

    def test_reference_loader_keeps_duplicate_cells(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reference_obs.csv"
            pd.DataFrame({"cell_type": self.labels, "batch": self.batches}).to_csv(path, index=False)
            labels, batches = m.load_reference_metadata(path, "cell_type", "batch")
            np.testing.assert_array_equal(labels, self.labels)
            np.testing.assert_array_equal(batches, self.batches)

    def test_drivers_explicit_alpha_and_k_and_reference(self):
        for path in (ROOT / "quantitative_metrics").glob("calulate_metrics_*.py"):
            tree = ast.parse(path.read_text())
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and
                     isinstance(n.func, ast.Name) and n.func.id == "compute_batch_diversity_metrics"]
            self.assertTrue(calls, path.name)
            for call in calls:
                self.assertEqual(len(call.args), 3)
                keywords = {kw.arg: kw.value for kw in call.keywords}
                self.assertEqual(keywords["alpha"].value, 0.5)
                self.assertEqual(keywords["k"].id, "k")
                self.assertIn("reference_labels", keywords)
                self.assertIn("reference_batches", keywords)

    def test_reference_export_uses_complete_pool(self):
        # Execute only the new export statements; no Scanpy/FAISS/model needed.
        tree = ast.parse((ROOT / "retrieval/main.py").read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        pos = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "reference_obs" for t in n.targets))
        import os
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            env = dict(pd=pd, os=os, args=SimpleNamespace(output_dir=folder),
                       obs_list=["cell_type", "donor_id"],
                       all_obs={"cell_type": np.array(["Q", "A", "A", "B"]),
                                "donor_id": np.array(["query", "x", "x", "y"])},
                       ref_global_idx=np.array([1, 2, 3]))
            code = compile(ast.Module(body=main.body[pos:pos + 2], type_ignores=[]), "<export>", "exec")
            exec(code, env)
            ref = pd.read_csv(Path(folder) / "reference_obs.csv")
            self.assertEqual(list(ref.columns), ["cell_type", "batch"])
            self.assertEqual(ref["cell_type"].tolist(), ["A", "A", "B"])
            self.assertEqual(ref["batch"].tolist(), ["x", "x", "y"])


if __name__ == "__main__":
    unittest.main()
