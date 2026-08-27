"""Evaluate one classifier and save auditable item-level predictions."""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

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

def hf_predict(texts: list[str], model_id: str, label_map: dict, device: int, batch_size: int) -> list[dict]:
    from transformers import pipeline
    classifier = pipeline("text-classification", model=model_id, tokenizer=model_id, device=device)
    raw = classifier(texts, batch_size=batch_size, truncation=True)
    return [{"label": label_map.get(x["label"], x["label"].lower()), "score": x["score"]} for x in raw]

def calculate_metrics(df: pd.DataFrame) -> dict:
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
        "expected_flip_accuracy": float((pred_flipped == gold_should_flip).mean()),
        "paired_both_correct": float(((joined.gold_label_orig == joined.prediction_orig) & (joined.gold_label_counterpart == joined.prediction_counterpart)).mean()),
        "mean_confidence_original": orig.confidence.mean(),
        "mean_confidence_counterpart": counterpart.confidence.mean(),
    }

def evaluate(data: Path, output: Path, name: str, cfg: dict, device=-1, batch_size=16) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(data)
    if "perturbation_type" not in df and "negation_type" in df:
        df = df.rename(columns={"negation_type": "perturbation_type"})
    texts = df.text.astype(str).tolist()
    if cfg["backend"] == "rule_based": preds = rule_predict(texts)
    else: preds = hf_predict(texts, cfg["model_id"], cfg.get("label_map", {}), device, batch_size)
    df["prediction"] = [x["label"] for x in preds]
    df["confidence"] = [x["score"] for x in preds]
    df["correct"] = df.prediction == df.gold_label
    df["model_name"] = name
    df["model_id"] = cfg.get("model_id", "local_rule_based_baseline")
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
