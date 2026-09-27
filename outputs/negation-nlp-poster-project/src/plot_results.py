"""Generate poster-ready plots only from recorded experiment results."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def save(fig, name):
    out=Path("results/figures"); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out/f"{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(out/f"{name}.pdf", bbox_inches="tight"); plt.close(fig)

def main():
    sns.set_theme(style="whitegrid", context="talk")
    summary=pd.read_csv("results/summary_metrics.csv")
    if "prediction_change_matches_gold" not in summary and "expected_flip_accuracy" in summary:
        summary["prediction_change_matches_gold"] = summary["expected_flip_accuracy"]
    long=summary.melt(id_vars=["dataset","model"], value_vars=["accuracy_original","accuracy_counterpart","prediction_change_matches_gold","paired_both_correct"], var_name="metric", value_name="score")
    long["system"] = long["dataset"] + " / " + long["model"]
    fig,ax=plt.subplots(figsize=(11,6)); sns.barplot(long,x="system",y="score",hue="metric",ax=ax); ax.set_ylim(0,1); ax.set_ylabel("Score"); ax.set_xlabel(""); ax.tick_params(axis="x",rotation=20); ax.legend(title="Metric",bbox_to_anchor=(1.02,1),loc="upper left"); save(fig,"model_comparison")
    by=pd.read_csv("results/by_negation_type.csv"); by["system"] = by["dataset"] + " / " + by["model"]; pivot=by.pivot(index="system",columns="perturbation_type",values="counterpart_accuracy")
    fig,ax=plt.subplots(figsize=(9,max(2.5,1.2*len(pivot)))); sns.heatmap(pivot,annot=True,vmin=0,vmax=1,cmap="RdYlGn",ax=ax); ax.set_xlabel("Perturbation construction"); ax.set_ylabel("Dataset / model"); save(fig,"negation_type_heatmap")
    print("Figures written to results/figures/")

if __name__ == "__main__": main()
