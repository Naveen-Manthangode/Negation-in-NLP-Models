from pathlib import Path
import pandas as pd
from src.prepare_data import prepare
from src.evaluate import evaluate, calculate_metrics

def test_prepare_and_evaluate(tmp_path):
    out=tmp_path/"pairs.csv"; df=prepare(Path("data/raw/seed_pairs.csv"),out)
    assert len(df)==48 and set(df.variant)=={"original","counterpart"}
    pred,metrics=evaluate(out,tmp_path/"pred.csv","rule_based",{"backend":"rule_based"})
    assert len(pred)==48 and 0<=metrics["paired_both_correct"]<=1

def test_metrics_perfect():
    df=pd.DataFrame({"pair_id":[1,1],"variant":["original","counterpart"],"gold_label":["positive","negative"],"prediction":["positive","negative"],"confidence":[.9,.8]})
    m=calculate_metrics(df); assert m["expected_flip_accuracy"]==1 and m["paired_both_correct"]==1
