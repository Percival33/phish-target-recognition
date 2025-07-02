# Evaluation Package - User Guide

A comprehensive evaluation system for phishing target recognition models with cross-validation and statistical analysis.

## Quick Start - What Do You Want To Do?

Choose your path based on what you want to accomplish:

### 🎯 Path A: Test Models on Provided Datasets (Recommended for First-Time Users)
- Use datasets from Phishpedia and VisualPhish papers
- Run cross-validation evaluation
- Compare model performance with statistical tests
- **[Go to Section: Testing with Provided Datasets](#testing-with-provided-datasets)**

### 🎯 Path B: Test Models on Your Own Dataset
- Use your own phishing/benign images
- Prepare data in required format
- Run evaluation pipeline
- **[Go to Section: Using Your Own Dataset](#using-your-own-dataset)**

### 🎯 Path C: Reproduce Paper Results
- Replicate exact results from research papers
- Use original training/test splits
- **[Go to Section: Reproducing Paper Results](#reproducing-paper-results)**

---

## Prerequisites

Before starting, ensure you have completed the main project setup:

```bash
# From project root directory
just tools
export PROJECT_ROOT_DIR=$(pwd)
```

---

## Testing with Provided Datasets

This is the **recommended approach** for first-time users. It uses the datasets from the original papers and provides robust cross-validation evaluation.

### Step 1: Download and Prepare Datasets

```bash
# Download Phishpedia dataset
uv run --with gdown gdown 12ypEMPRQ43zGRqHGut0Esq2z5en0DH4g -O download_phish.zip
mkdir -p $PROJECT_ROOT_DIR/data/raw/phishpedia/phish_sample_30k
unzip -q download_phish.zip -d $PROJECT_ROOT_DIR/data/raw/phishpedia/phish_sample_30k
rm download_phish.zip

uv run --with gdown gdown 1yORUeSrF5vGcgxYrsCoqXcpOUHt-iHq_ -O download_benign.zip
mkdir -p $PROJECT_ROOT_DIR/data/raw/phishpedia/benign_sample_30k
unzip -q download_benign.zip -d $PROJECT_ROOT_DIR/data/raw/phishpedia/benign_sample_30k
rm download_benign.zip

# Download VisualPhish dataset
uv run --with gdown gdown 1l-aQk54F0tAZ-RPfOyGo1jtz-Dsxo1Ao -O download_vp.zip
mkdir -p $PROJECT_ROOT_DIR/data/raw/visualphish
unzip -q download_vp.zip -d $PROJECT_ROOT_DIR/data/raw/visualphish
rm download_vp.zip
```

### Step 2: Create Cross-Validation Splits

This step creates balanced train/validation splits for robust evaluation:

```bash
cd $PROJECT_ROOT_DIR/src/cross_validation
just splits-links
```

**What this does:**
- Creates 3-fold cross-validation splits (configurable in `config.json`)
- Generates `data_splits/` directory with `split_0/`, `split_1/`, `split_2/`
- Each split contains `train.csv`, `val.csv`, and organized image symlinks
- Ensures balanced class distribution across splits

### Step 3: Run Models on Each Split

Now run each model on every cross-validation split:

#### 3a. Run Phishpedia

```bash
cd $PROJECT_ROOT_DIR/src/models/phishpedia
just setup
just extract-targetlist
just prepare

# Login to Weights & Biases (replace YOUR_API_KEY)
uv run wandb login YOUR_API_KEY

# Run on each split
for split in 0 1 2; do
    echo "Running Phishpedia on split_${split}..."
    uv run phishpedia.py \
        --folder $PROJECT_ROOT_DIR/data_splits/split_${split}/Phishpedia/images/val \
        --output_txt split_${split}.txt \
        --log
done
```

#### 3b. Run VisualPhish

```bash
cd $PROJECT_ROOT_DIR

# Setup (one time)
uv sync --frozen
uv run wandb login YOUR_API_KEY

# Train and evaluate on each split
for split in 0 1 2; do
    echo "Training VisualPhish on split_${split}..."

    # Train model
    uv run trainer.py \
        --dataset-path $PROJECT_ROOT_DIR/data_splits/split_${split}/VisualPhish \
        --logdir $PROJECT_ROOT_DIR/logdir/split_${split} \
        --output-dir $PROJECT_ROOT_DIR/data/processed/VisualPhish/split_${split}

    # Evaluate model
    uv run src/models/visualphishnet/eval_new.py \
        --emb-dir $PROJECT_ROOT_DIR/data/processed/VisualPhish/split_${split} \
        --threshold 8.0 \
        --result-path $PROJECT_ROOT_DIR/logs/VisualPhish/split_${split}
done
```

#### 3c. Run Baseline

```bash
cd $PROJECT_ROOT_DIR/src/models/baseline

for split in 0 1 2; do
    echo "Running Baseline on split_${split}..."
    uv run python evaluate_baseline.py \
        --split-dir $PROJECT_ROOT_DIR/data_splits/split_${split}/Baseline
done
```

### Step 4: Run Statistical Evaluation

After all models have finished, run the evaluation pipeline:

```bash
cd $PROJECT_ROOT_DIR/src/eval
just evaluate
```

**What this produces:**
- `critical_difference_analysis_results/identification_rate_rankings.png` - Model performance rankings
- `critical_difference_analysis_results/identification_rate_cd_diagram.png` - Statistical significance diagram
- `critical_difference_analysis_results/identification_rate_summary.csv` - Detailed results

---

## Using Your Own Dataset

If you have your own phishing/benign images, follow these steps:

### Step 1: Prepare Your Data

Your data should be organized as:
```
my_dataset/
├── images/
│   ├── 00000.png
│   ├── 00001.jpg
│   └── ...
└── labels.txt  # One URL per line, matching image order
```

### Step 2: Convert to Required Format

```bash
# Generate CSV file for organizer scripts
uv run scripts/prepare_data_for_organizer.py \
    --image-folder my_dataset/images/ \
    --labels-file my_dataset/labels.txt \
    --output-csv my_dataset/prepared_data.csv \
    --is-phishing  # or --no-is-phishing for benign data
```

### Step 3: Organize for Each Model

```bash
# For Phishpedia
uv run src/organize_by_sample.py \
    --csv my_dataset/prepared_data.csv \
    --screenshots my_dataset/images/ \
    --output my_dataset/phishpedia_format

# For VisualPhish
uv run src/organize_by_target.py \
    --csv my_dataset/prepared_data.csv \
    --screenshots my_dataset/images/ \
    --output my_dataset/visualphish_format
```

### Step 4: Update Configuration

Edit `config.json` to point to your dataset:

```json
{
  "cross_validation_config": {
    "dataset_image_paths": {
      "my_dataset": {
        "path": "my_dataset/visualphish_format",
        "label_strategy": "directory",
        "target_mapping": {
          "phishing": "phishing",
          "benign": "trusted_list"
        }
      }
    }
  }
}
```

### Step 5: Follow Steps 2-4 from "Testing with Provided Datasets"

The process is identical once your data is properly formatted.

---

## Reproducing Paper Results

To reproduce exact results from the original papers:

### Option 1: Use Original Splits
If you have the original train/test splits from the papers, place them in:
- `data/original_splits/phishpedia/`
- `data/original_splits/visualphish/`

### Option 2: Disable Cross-Validation
Edit `config.json`:
```json
{
  "cross_validation_config": {
    "enabled": false
  }
}
```

Then run models on the full datasets without splitting.

---

## Configuration Guide

The `config.json` file controls all evaluation parameters:

### Key Configuration Sections

#### Cross-Validation Settings
```json
"cross_validation_config": {
  "enabled": true,           // Enable/disable cross-validation
  "n_splits": 3,            // Number of folds (3 is recommended)
  "shuffle": true,          // Shuffle data before splitting
  "random_state": 42,       // For reproducible splits
  "output_splits_directory": "data_splits"
}
```

#### Dataset Paths
```json
"dataset_image_paths": {
  "visualphish": {
    "path": "data/raw/visualphish",
    "label_strategy": "labels_file",  // or "directory"
    "target_mapping": {
      "phishing": "newly_crawled_phishing",
      "benign": "benign_test"
    }
  }
}
```

#### Model Result Paths
```json
"paths_to_csv_files": {
  "VisualPhish": {
    "Phishpedia": "logs/phishpedia/results.csv",
    "VisualPhish": "logs/visualphish/results.csv",
    "Baseline": "logs/baseline/results.csv"
  }
}
```

**Important:** Update these paths to match where your models save their results.

---

## Troubleshooting

### Common Issues

#### "No such file or directory" errors
- Ensure `PROJECT_ROOT_DIR` is set correctly
- Check that datasets are downloaded to the right locations
- Verify model result files exist at configured paths

#### Cross-validation splits look wrong
- Check `target_mapping` in config matches your dataset structure
- Ensure label files have correct format (one label per line)
- Verify image files are readable and in supported formats

#### Models fail to run
- Ensure all dependencies are installed: `uv sync --frozen`
- Check that Weights & Biases is configured: `uv run wandb login`
- Verify model-specific requirements (see model README files)

#### Statistical evaluation fails
- Ensure all model result files exist and have correct format
- Check that CSV column names match configuration
- Verify at least 2 models have results for comparison

### Getting Help

1. Check model-specific README files in `src/models/*/`
2. Verify your configuration against `config.json` examples
3. Ensure all prerequisite steps are completed
4. Check that file paths in configuration are correct

---

## Understanding the Results

### Output Files

- **Rankings Plot** (`identification_rate_rankings.png`): Shows average performance of each model across all splits
- **Critical Difference Diagram** (`identification_rate_cd_diagram.png`): Statistical significance test results
- **Summary CSV** (`identification_rate_summary.csv`): Detailed numerical results

### Interpreting Statistical Tests

- **Friedman Test**: Tests if there are significant differences between models
- **Nemenyi Post-hoc Test**: Determines which specific models differ significantly
- **Critical Difference**: Models within this distance are not significantly different

Models connected by a horizontal line in the CD diagram do **not** have statistically significant differences in performance.
