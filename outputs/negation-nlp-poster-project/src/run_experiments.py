"""Prepare data, evaluate selected models, and aggregate poster metrics."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import yaml
from .prepare_data import prepare
from .evaluate import evaluate

def upsert_csv(path: Path, new: pd.DataFrame, keys: list[str]) -> None:
    """Preserve unrelated runs while replacing rows with the same logical keys."""
    if path.exists():
        old = pd.read_csv(path)
        if set(keys) <= set(old.columns):
            old_keys = old[keys].astype(str).agg("\x1f".join, axis=1)
            new_keys = set(new[keys].astype(str).agg("\x1f".join, axis=1))
            old = old[~old_keys.isin(new_keys)]
            new = pd.concat([old, new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--models", nargs="+", default=["rule_based"]); p.add_argument("--device", type=int, default=-1); p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--data", type=Path, default=Path("data/processed/pairs.csv")); p.add_argument("--dataset-name", default="controlled_negation")
    p.add_argument("--config", type=Path, default=Path("config/models.yaml")); a = p.parse_args()
    processed = a.data
    if not processed.exists() and processed == Path("data/processed/pairs.csv"): prepare(Path("data/raw/seed_pairs.csv"), processed)
    if not processed.exists(): raise FileNotFoundError(f"Dataset not found: {processed}. Run its preparation script first.")
    configs = yaml.safe_load(a.config.read_text())["models"]; rows=[]; breakdown=[]
    for name in a.models:
        if name not in configs: raise ValueError(f"Unknown model {name}; choose from {list(configs)}")
        pred, metrics = evaluate(processed, Path(f"results/predictions/{a.dataset_name}__{name}.csv"), name, configs[name], a.device, a.batch_size)
        rows.append({"dataset": a.dataset_name, "model": name, "model_id": configs[name].get("model_id", "local_rule_based_baseline"), "training_domain": configs[name].get("training_domain", "not_applicable"), "run_utc": datetime.now(timezone.utc).isoformat(), **metrics, "n_truncated": int(pred.was_truncated.sum())})
        for kind, group in pred.groupby("perturbation_type"):
            counterpart = group[group.variant != "original"]
            breakdown.append({"dataset": a.dataset_name, "model": name, "perturbation_type": kind, "n_pairs": len(counterpart), "counterpart_accuracy": counterpart.correct.mean(), "mean_counterpart_confidence": counterpart.confidence.mean()})
    Path("results").mkdir(exist_ok=True)
    summary = pd.DataFrame(rows); detail = pd.DataFrame(breakdown)
    upsert_csv(Path("results/summary_metrics.csv"), summary, ["dataset", "model"])
    upsert_csv(Path("results/by_negation_type.csv"), detail, ["dataset", "model", "perturbation_type"])
    summary_dir = Path("results/summaries"); summary_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_dir / f"{a.dataset_name}.csv", index=False)
    print(summary[["dataset", "model", "accuracy_original", "accuracy_counterpart", "prediction_change_matches_gold", "paired_both_correct", "n_truncated"]].to_string(index=False))

if __name__ == "__main__": main()
