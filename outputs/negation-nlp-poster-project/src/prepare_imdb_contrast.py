"""Download and convert the official AllenAI IMDb Contrast Set.

Rows in each original/contrast TSV are aligned by position in the official
release. By default, this script keeps only pairs whose changed tokens contain
a negation cue. Use --all-contrasts for the complete behavioral contrast set.
"""
from __future__ import annotations
import argparse
import re
import urllib.request
from pathlib import Path
import pandas as pd

BASE = "https://raw.githubusercontent.com/allenai/contrast-sets/main/IMDb/data"
CUES = {
    "not", "no", "never", "neither", "nor", "nobody", "nothing", "nowhere",
    "hardly", "scarcely", "barely", "without", "cannot", "can't", "couldn't",
    "didn't", "doesn't", "don't", "hadn't", "hasn't", "haven't", "isn't",
    "wasn't", "weren't", "won't", "wouldn't", "shouldn't", "ain't",
}

def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, destination)

def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:n't)?", str(text).lower())

def changed_negation_cues(original: str, contrast: str) -> list[str]:
    """Return negation cues whose token counts changed across the edit."""
    a, b = tokens(original), tokens(contrast)
    return sorted(cue for cue in CUES if a.count(cue) != b.count(cue))

def convert(split: str, raw_dir: Path, output: Path, negation_only: bool = True) -> pd.DataFrame:
    paths = {}
    for variant in ("original", "contrast"):
        path = raw_dir / f"{split}_{variant}.tsv"
        if not path.exists():
            download(f"{BASE}/{split}_{variant}.tsv", path)
        paths[variant] = path
    original = pd.read_csv(paths["original"], sep="\t")
    contrast = pd.read_csv(paths["contrast"], sep="\t")
    if len(original) != len(contrast):
        raise ValueError("Official original and contrast files are not aligned")
    rows = []
    for i, (a, b) in enumerate(zip(original.itertuples(), contrast.itertuples()), 1):
        cues = changed_negation_cues(a.Text, b.Text)
        if negation_only and not cues:
            continue
        perturbation = "+".join(cues) if cues else "other_human_contrast"
        for variant, item in (("original", a), ("counterpart", b)):
            rows.append({
                "pair_id": f"imdb_{split}_{i}",
                "perturbation_type": perturbation,
                "text": str(item.Text).replace("<br />", " ").strip(),
                "gold_label": str(item.Sentiment).lower(),
                "variant": variant,
                "source": "allenai_imdb_contrast_set",
            })
    result = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    mode = "negation-related" if negation_only else "all"
    print(f"Wrote {len(result)//2} {mode} IMDb contrast pairs to {output}")
    return result

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "test"], default="test")
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/imdb_contrast"))
    p.add_argument("--output", type=Path, default=Path("data/processed/imdb_contrast_negation.csv"))
    p.add_argument("--all-contrasts", action="store_true", help="Keep edits unrelated to negation too")
    a = p.parse_args()
    convert(a.split, a.raw_dir, a.output, negation_only=not a.all_contrasts)

if __name__ == "__main__":
    main()
