"""Evaluate one classifier and save auditable item-level predictions."""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

REQUIRED_COLUMNS = {"pair_id", "variant", "text", "gold_label", "perturbation_type"}
VALID_LABELS = {"positive", "negative"}

def validate_evaluation_data(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df[list(REQUIRED_COLUMNS)].isnull().any().any():
        raise ValueError("Required dataset columns may not contain missing values")
    if not set(df.gold_label) <= VALID_LABELS:
        raise ValueError(f"Gold labels must be {sorted(VALID_LABELS)}")
    if not set(df.variant) <= {"original", "counterpart"}:
        raise ValueError("Variants must be 'original' or 'counterpart'")
    counts = df.groupby(["pair_id", "variant"]).size()
    if (counts != 1).any():
        raise ValueError("Each pair_id must occur exactly once per variant")
    variant_counts = df.groupby("pair_id").variant.nunique()
    if len(variant_counts) == 0 or (variant_counts != 2).any():
        raise ValueError("Every pair_id must contain one original and one counterpart")

POS = {"good", "excellent", "enjoy", "wonderful", "helpful", "recommend", "improve", "well", "like", "love", "reliable", "clear", "pleasant", "useful"}
NEG = {"bad", "difficult", "confusing", "dislike", "disappointing", "unhelpful", "crashes", "predictable"}
NEGATORS = {"not", "never", "hardly", "isn't", "aren't", "didn't", "doesn't"}

def rule_predict(texts: list[str]) -> list[dict]:
    out = []
    for text in texts:
        words = re.findall(r"[a-z]+(?:n't)?", text.lower())
        score = sum(w in POS for w in words) - sum(w in NEG for w in words)
        if any(w in NEGATORS for w in words) or "far from" in text.lower(): score *= -1
        label = "positive" if score >= 0 else "negative"
        out.append({"label": label, "score": min(0.99, 0.55 + 0.1 * abs(score))})
    return out

def hf_predict(texts: list[str], model_id: str, label_map: dict, device: int, batch_size: int) -> tuple[list[dict], dict]:
    from transformers import pipeline
    classifier = pipeline("text-classification", model=model_id, tokenizer=model_id, device=device)
    max_length = min(classifier.tokenizer.model_max_length, 100_000)
    token_lengths = [len(x) for x in classifier.tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]]
    raw = classifier(texts, batch_size=batch_size, truncation=True)
    predictions = [{"label": label_map.get(x["label"], x["label"].lower()), "score": x["score"]} for x in raw]
    unknown = sorted({x["label"] for x in predictions} - VALID_LABELS)
    if unknown:
        raise ValueError(f"Model produced unmapped labels: {unknown}. Update config/models.yaml")
    metadata = {"token_lengths": token_lengths, "max_length": max_length,
                "resolved_revision": getattr(classifier.model.config, "_commit_hash", None) or "unknown"}
    return predictions, metadata

def calculate_metrics(df: pd.DataFrame) -> dict:
    validate_evaluation_data(df)
    orig = df[df.variant == "original"].set_index("pair_id")
    counterpart = df[df.variant != "original"].set_index("pair_id")
    if len(orig) != len(counterpart):
        raise ValueError("Every pair needs exactly one original and one counterpart")
    joined = orig.join(counterpart, lsuffix="_orig", rsuffix="_counterpart")
    gold_should_flip = joined.gold_label_orig != joined.gold_label_counterpart
    pred_flipped = joined.prediction_orig != joined.prediction_counterpart
    return {
        "n_pairs": len(joined),
        "accuracy_all": accuracy_score(df.gold_label, df.prediction),
        "macro_f1_all": f1_score(df.gold_label, df.prediction, average="macro"),
        "accuracy_original": accuracy_score(orig.gold_label, orig.prediction),
        "accuracy_counterpart": accuracy_score(counterpart.gold_label, counterpart.prediction),
        "accuracy_drop": accuracy_score(orig.gold_label, orig.prediction) - accuracy_score(counterpart.gold_label, counterpart.prediction),
        "prediction_change_matches_gold": float((pred_flipped == gold_should_flip).mean()),
        "expected_flip_accuracy": float((pred_flipped == gold_should_flip).mean()),
        "paired_both_correct": float(((joined.gold_label_orig == joined.prediction_orig) & (joined.gold_label_counterpart == joined.prediction_counterpart)).mean()),
        "mean_confidence_original": orig.confidence.mean(),
        "mean_confidence_counterpart": counterpart.confidence.mean(),
    }

def evaluate(data: Path, output: Path, name: str, cfg: dict, device=-1, batch_size=16) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(data)
    if "perturbation_type" not in df and "negation_type" in df:
        df = df.rename(columns={"negation_type": "perturbation_type"})
    validate_evaluation_data(df)
    texts = df.text.astype(str).tolist()
    if cfg["backend"] == "rule_based":
        preds = rule_predict(texts)
        metadata = {"token_lengths": [None] * len(df), "max_length": None, "resolved_revision": "local"}
    else:
        preds, metadata = hf_predict(texts, cfg["model_id"], cfg.get("label_map", {}), device, batch_size)
    df["prediction"] = [x["label"] for x in preds]
    df["confidence"] = [x["score"] for x in preds]
    df["correct"] = df.prediction == df.gold_label
    df["model_name"] = name
    df["model_id"] = cfg.get("model_id", "local_rule_based_baseline")
    df["model_revision"] = metadata["resolved_revision"]
    df["token_count"] = metadata["token_lengths"]
    df["was_truncated"] = False if metadata["max_length"] is None else df.token_count > metadata["max_length"]
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df, calculate_metrics(df)

def main() -> None:
    import yaml
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/processed/pairs.csv")); p.add_argument("--model", required=True)
    p.add_argument("--config", type=Path, default=Path("config/models.yaml")); p.add_argument("--device", type=int, default=-1); p.add_argument("--batch-size", type=int, default=16)
    a = p.parse_args(); cfg = yaml.safe_load(a.config.read_text())["models"][a.model]
    _, metrics = evaluate(a.data, Path(f"results/predictions/{a.model}.csv"), a.model, cfg, a.device, a.batch_size)
    print(pd.Series(metrics).to_string())

if __name__ == "__main__": main()
