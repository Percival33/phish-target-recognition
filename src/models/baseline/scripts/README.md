# Baseline Model Scripts

This directory contains utility scripts for the baseline phishing detection model.

## evaluate_baseline.py

Evaluation script for baseline model results. This script takes a CSV file (output from `query.py`) and calculates classification and target identification metrics using the `tools.metrics` module.

### Usage

```bash
# Basic evaluation
uv run python scripts/evaluate_baseline.py results.csv

# Evaluation with ROC curve plot
uv run python scripts/evaluate_baseline.py results.csv --plot
```

### Arguments

- `csv_path`: Path to CSV file containing query results (output from `query.py`)
- `--plot`: Optional flag to generate ROC curve plot

### Input CSV Format

The script expects a CSV file with the following columns:
- `file`: Query image filename
- `baseline_class`: Predicted class (0 or 1)
- `baseline_distance`: Distance to closest match
- `baseline_target`: Predicted target (from closest match)
- `true_class`: True class label
- `true_target`: True target label

### Output

The script outputs two sets of metrics:

**Class Classification Metrics:**
- `f1_weighted`: Weighted F1 score
- `roc_auc`: ROC AUC score
- `mcc`: Matthews Correlation Coefficient
- `precision`: Macro-averaged precision
- `recall`: Macro-averaged recall

**Target Identification Metrics:**
- `target_f1_micro`: Micro-averaged F1 score for targets
- `target_f1_macro`: Macro-averaged F1 score for targets
- `target_f1_weighted`: Weighted F1 score for targets
- `target_mcc`: Matthews Correlation Coefficient for targets
- `precision`: Macro-averaged precision for targets
- `recall`: Macro-averaged recall for targets
- `identification_rate`: Rate of correctly identified phishing targets (Idp/Repp_TP)

### Example

```bash
# Run evaluation on query results
uv run python scripts/evaluate_baseline.py output/query_results.csv --plot
```

This will display the metrics in the terminal and show an ROC curve plot.
