from pathlib import Path
import pandas as pd

def main():
    df=pd.read_csv("results/summary_metrics.csv")
    if "prediction_change_matches_gold" not in df and "expected_flip_accuracy" in df:
        df["prediction_change_matches_gold"] = df["expected_flip_accuracy"]
    cols=["dataset","model","n_pairs","accuracy_original","accuracy_counterpart","accuracy_drop","prediction_change_matches_gold","paired_both_correct"]
    pretty=df[cols].copy(); pretty.columns=["Dataset","Model","Pairs","Acc. original","Acc. counterpart","Acc. drop","Prediction-change agreement","Both correct"]
    for c in pretty.columns[3:]: pretty[c]=pretty[c].map(lambda x:f"{x:.3f}")
    Path("results/poster_table.md").write_text(pretty.to_markdown(index=False)+"\n")
    Path("results/poster_table.tex").write_text(pretty.to_latex(index=False,escape=True)+"\n")
    print("Poster tables written to results/")
if __name__ == "__main__": main()
