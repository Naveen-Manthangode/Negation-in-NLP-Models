# Negation Capturing in NLP Models

A reproducible 5-ECTS poster project that measures how sentiment classifiers behave when an affirmative sentence is changed by negation. Every item is a **minimal pair**: an original sentence and a controlled negated counterpart.

## Research questions

1. How much does classification performance change on negated text?
2. Does a model flip its prediction when the gold sentiment flips?
3. Which negation constructions (`not`, `never`, contractions, implicit negation) are hardest?
4. Does confidence fall after negation?

## Project layout

```text
config/models.yaml              model registry
data/raw/seed_pairs.csv         editable hand-written seed examples
data/processed/                 generated datasets
results/                        metrics, predictions, tables, figures
src/prepare_data.py             validates/expands paired data
src/prepare_imdb_contrast.py    downloads/converts IMDb Contrast Sets
src/evaluate.py                 evaluates one model
src/run_experiments.py          runs all configured models
src/plot_results.py             produces poster-ready PNG/PDF figures
src/make_poster_tables.py       produces Markdown/LaTeX tables
src/demo.py                     interactive pair inspection
tests/test_pipeline.py          offline tests
```

## Quick start

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python -m src.prepare_data
python -m src.run_experiments --models rule_based
python -m src.plot_results
python -m src.make_poster_tables
```

For the real transformer experiment (internet is needed on first run):

```bash
python -m src.run_experiments --models distilbert_sst2 roberta_sentiment
python -m src.plot_results
python -m src.make_poster_tables
```

Models are cached locally by Hugging Face. For a CPU laptop, start with DistilBERT. Use `--device 0` for the first GPU, or `--device -1` for CPU.

## Use a public dataset (optional)

The included controlled pairs make the project immediately runnable. To enlarge it with real sentences, prepare a CSV with the same columns as `data/raw/seed_pairs.csv`, manually verify each transformation, and run:

```bash
python -m src.prepare_data --input path/to/pairs.csv --output data/processed/pairs.csv
```

Automatic insertion of “not” often creates ungrammatical or label-ambiguous text, so this project deliberately requires paired examples to be reviewed. Public sources such as SST-2 may supply original sentences, but their modified counterparts need annotation and the dataset license must be respected.

## Outputs for the poster

- `results/summary_metrics.csv`: model-level metrics
- `results/by_negation_type.csv`: construction-level breakdown
- `results/predictions/*.csv`: item-level audit trail
- `results/figures/model_comparison.png/.pdf`
- `results/figures/negation_type_heatmap.png/.pdf`
- `results/poster_table.md` and `poster_table.tex`

Do not present the included rule-based output as a transformer result. Run the named models and report the generated files, model revisions, date, hardware, and dataset size. Inspect errors before writing conclusions.

## IMDb Contrast Set experiment

The official AllenAI release contains aligned human-edited original/contrast reviews. The default command retains only pairs where the edit changes a negation cue:

```bash
python -m src.prepare_imdb_contrast
python -m src.run_experiments --data data/processed/imdb_contrast_negation.csv --dataset-name imdb_contrast_negation --models distilbert_sst2 bert_sst2 roberta_sentiment --device -1
python -m src.plot_results
python -m src.make_poster_tables
python -m src.demo --dataset-name imdb_contrast_negation --model distilbert_sst2 --errors-only
```

Calculate paired 95% bootstrap confidence intervals and exact McNemar tests:

```bash
python -m src.significance --dataset-name imdb_contrast_negation --models distilbert_sst2 bert_sst2 roberta_sentiment --n-bootstrap 10000 --seed 42
```

This creates `results/significance/bootstrap_confidence_intervals.csv` and
`results/significance/mcnemar_model_comparisons.csv`. Bootstrap sampling is by
complete pair ID. Treat `p < 0.05` as a conventional threshold, report exact
p-values, and note that testing many model/metric combinations increases the
chance of false positives.

To evaluate all IMDb contrast edits, including changes unrelated to negation:

```bash
python -m src.prepare_imdb_contrast --all-contrasts --output data/processed/imdb_contrast_all.csv
python -m src.run_experiments --data data/processed/imdb_contrast_all.csv --dataset-name imdb_contrast_all --models distilbert_sst2 bert_sst2 roberta_sentiment --device -1
```

The full set measures general contrast robustness. Only the filtered set directly supports claims about changed negation cues. The downloader retrieves data from the official `allenai/contrast-sets` repository and therefore needs internet access.

## Suggested poster structure

**Motivation → Paired design → Models and metrics → Main comparison figure → Negation-type heatmap → Error examples → Limitations.** Report confidence intervals or multiple seeds if fine-tuning is added. The current project evaluates pretrained models and makes no causal claim about model internals.

## Reproducibility notes

- Seeds are fixed where applicable.
- Exact model IDs are recorded in prediction and summary files.
- No results are bundled as scientific findings; generated results reflect what was actually executed.
- Add your name, course, institution, supervisor, and execution environment before submission.
