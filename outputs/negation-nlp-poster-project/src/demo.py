"""Print paired predictions for quick classroom/poster demonstrations."""
import argparse
import pandas as pd
def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",default="rule_based"); p.add_argument("--dataset-name",default="controlled_negation"); p.add_argument("--errors-only",action="store_true"); a=p.parse_args()
    df=pd.read_csv(f"results/predictions/{a.dataset_name}__{a.model}.csv")
    if a.errors_only: df=df[~df.correct]
    for pair_id,g in df.groupby("pair_id"):
        print(f"\nPair {pair_id} ({g.perturbation_type.iloc[0]})")
        for r in g.itertuples(): print(f"  {r.variant:8} gold={r.gold_label:8} pred={r.prediction:8} conf={r.confidence:.3f} | {r.text}")
if __name__ == "__main__": main()
