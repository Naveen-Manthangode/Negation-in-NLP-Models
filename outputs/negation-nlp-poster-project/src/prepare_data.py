"""Validate paired examples and reshape them to one row per sentence."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

PAIR_COLUMNS = ["pair_id", "original_text", "negated_text", "original_label", "negated_label", "negation_type"]
LABELS = {"positive", "negative"}

def prepare(input_path: Path, output_path: Path) -> pd.DataFrame:
    pairs = pd.read_csv(input_path)
    missing = set(PAIR_COLUMNS) - set(pairs.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if pairs.pair_id.duplicated().any():
        raise ValueError("pair_id values must be unique")
    observed = set(pairs.original_label) | set(pairs.negated_label)
    if not observed <= LABELS:
        raise ValueError(f"Labels must be {sorted(LABELS)}; found {sorted(observed)}")
    if (pairs.original_text.str.strip() == pairs.negated_text.str.strip()).any():
        raise ValueError("Each pair must contain different texts")
    frames = []
    for variant, text_col, label_col in [
        ("original", "original_text", "original_label"),
        ("counterpart", "negated_text", "negated_label"),
    ]:
        frame = pairs[["pair_id", "negation_type", text_col, label_col]].copy()
        frame.columns = ["pair_id", "perturbation_type", "text", "gold_label"]
        frame["variant"] = variant
        frames.append(frame)
    long = pd.concat(frames, ignore_index=True).sort_values(["pair_id", "variant"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(output_path, index=False)
    print(f"Wrote {len(long)} sentences ({len(pairs)} pairs) to {output_path}")
    return long

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("data/raw/seed_pairs.csv"))
    p.add_argument("--output", type=Path, default=Path("data/processed/pairs.csv"))
    a = p.parse_args()
    prepare(a.input, a.output)

if __name__ == "__main__": main()
