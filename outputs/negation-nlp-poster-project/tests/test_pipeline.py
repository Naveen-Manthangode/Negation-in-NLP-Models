from pathlib import Path
import pandas as pd
from src.prepare_data import prepare
from src.evaluate import evaluate, calculate_metrics, validate_evaluation_data
from src.prepare_imdb_contrast import changed_token_count

def test_prepare_and_evaluate(tmp_path):
    out=tmp_path/"pairs.csv"; df=prepare(Path("data/raw/seed_pairs.csv"),out)
    assert len(df)==48 and set(df.variant)=={"original","counterpart"}
    pred,metrics=evaluate(out,tmp_path/"pred.csv","rule_based",{"backend":"rule_based"})
    assert len(pred)==48 and 0<=metrics["paired_both_correct"]<=1

def test_metrics_perfect():
    df=pd.DataFrame({"pair_id":[1,1],"variant":["original","counterpart"],"text":["good","not good"],"perturbation_type":["not","not"],"gold_label":["positive","negative"],"prediction":["positive","negative"],"confidence":[.9,.8]})
    m=calculate_metrics(df); assert m["expected_flip_accuracy"]==1 and m["paired_both_correct"]==1

def test_duplicate_pair_variant_is_rejected():
    df=pd.DataFrame({"pair_id":[1,1,1],"variant":["original","original","counterpart"],"text":["a","a","b"],"perturbation_type":["not"]*3,"gold_label":["positive","positive","negative"]})
    try:
        validate_evaluation_data(df)
        assert False, "duplicate pair variant should fail"
    except ValueError as exc:
        assert "exactly once" in str(exc)

def test_changed_token_count():
    assert changed_token_count("This is good", "This is not good") == 1
