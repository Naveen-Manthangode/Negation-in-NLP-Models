from pathlib import Path
import pandas as pd

def main():
    df=pd.read_csv("results/summary_metrics.csv")
    cols=["dataset","model","n_pairs","accuracy_original","accuracy_counterpart","accuracy_drop","expected_flip_accuracy","paired_both_correct"]
    pretty=df[cols].copy(); pretty.columns=["Dataset","Model","Pairs","Acc. original","Acc. counterpart","Acc. drop","Flip accuracy","Both correct"]
    for c in pretty.columns[3:]: pretty[c]=pretty[c].map(lambda x:f"{x:.3f}")
    Path("results/poster_table.md").write_text(pretty.to_markdown(index=False)+"\n")
    Path("results/poster_table.tex").write_text(pretty.to_latex(index=False,escape=True)+"\n")
    print("Poster tables written to results/")
if __name__ == "__main__": main()
