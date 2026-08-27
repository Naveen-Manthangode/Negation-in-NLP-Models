"""Paired confidence intervals and model-comparison significance tests.

The bootstrap resamples complete pair IDs, never individual sentences. This
preserves the dependence between each original and its edited counterpart.
McNemar's exact test compares whether two models differ in item-level errors.
"""
from __future__ import annotations
import argparse
from itertools import combinations
from math import comb
from pathlib import Path
import numpy as np
import pandas as pd

METRICS = ("accuracy_original", "accuracy_counterpart", "expected_flip_accuracy", "paired_both_correct")

def pair_table(df: pd.DataFrame) -> pd.DataFrame:
    original = df[df.variant == "original"].set_index("pair_id")
    counterpart = df[df.variant != "original"].set_index("pair_id")
    table = pd.DataFrame(index=original.index)
    table["accuracy_original"] = (original.prediction == original.gold_label).astype(float)
    table["accuracy_counterpart"] = (counterpart.prediction == counterpart.gold_label).astype(float)
    gold_flip = original.gold_label != counterpart.gold_label
    predicted_flip = original.prediction != counterpart.prediction
    table["expected_flip_accuracy"] = (gold_flip == predicted_flip).astype(float)
    table["paired_both_correct"] = table.accuracy_original * table.accuracy_counterpart
    return table

def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float]:
    n = len(values)
    estimates = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        estimates[i] = values[rng.integers(0, n, n)].mean()
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)

def exact_mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Return A-only correct, B-only correct, and two-sided exact p-value."""
    a_only = int(np.sum((a == 1) & (b == 0)))
    b_only = int(np.sum((a == 0) & (b == 1)))
    discordant = a_only + b_only
    if discordant == 0:
        p = 1.0
    else:
        tail = sum(comb(discordant, k) for k in range(min(a_only, b_only) + 1)) / (2 ** discordant)
        p = min(1.0, 2.0 * tail)
    return a_only, b_only, p

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-name", default="imdb_contrast_negation")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--predictions-dir", type=Path, default=Path("results/predictions"))
    p.add_argument("--n-bootstrap", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    tables = {}
    ci_rows = []
    for model in a.models:
        path = a.predictions_dir / f"{a.dataset_name}__{model}.csv"
        tables[model] = pair_table(pd.read_csv(path))
        for metric in METRICS:
            values = tables[model][metric].to_numpy()
            low, high = bootstrap_ci(values, rng, a.n_bootstrap)
            ci_rows.append({"dataset": a.dataset_name, "model": model, "metric": metric,
                            "estimate": values.mean(), "ci_95_low": low, "ci_95_high": high,
                            "n_pairs": len(values), "bootstrap_samples": a.n_bootstrap, "seed": a.seed})
    comparison_rows = []
    for model_a, model_b in combinations(a.models, 2):
        common = tables[model_a].index.intersection(tables[model_b].index)
        for metric in METRICS:
            va = tables[model_a].loc[common, metric].to_numpy()
            vb = tables[model_b].loc[common, metric].to_numpy()
            a_only, b_only, p_value = exact_mcnemar(va, vb)
            comparison_rows.append({"dataset": a.dataset_name, "metric": metric,
                                    "model_a": model_a, "model_b": model_b,
                                    "estimate_a": va.mean(), "estimate_b": vb.mean(),
                                    "difference_a_minus_b": va.mean() - vb.mean(),
                                    "a_only_correct": a_only, "b_only_correct": b_only,
                                    "mcnemar_exact_p": p_value, "n_pairs": len(common)})
    out = Path("results/significance"); out.mkdir(parents=True, exist_ok=True)
    ci = pd.DataFrame(ci_rows); comparisons = pd.DataFrame(comparison_rows)
    ci.to_csv(out / "bootstrap_confidence_intervals.csv", index=False)
    comparisons.to_csv(out / "mcnemar_model_comparisons.csv", index=False)
    print("\n95% paired bootstrap confidence intervals")
    print(ci.to_string(index=False))
    print("\nExact McNemar comparisons (p < 0.05 is conventionally significant; interpret with multiple-testing caution)")
    print(comparisons.to_string(index=False))

if __name__ == "__main__":
    main()
