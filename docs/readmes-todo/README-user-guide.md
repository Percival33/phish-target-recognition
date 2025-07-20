# Phishing Target Recognition - User Guide

A comprehensive system for evaluating phishing detection models with cross-validation and statistical analysis.

## 🚀 Quick Start - What Do You Want To Do?

Choose your path based on what you want to accomplish:

### 🎯 **Path A: Complete Model Evaluation (Recommended)**
- Use provided datasets from research papers
- Run all three models (Phishpedia, VisualPhish, Baseline)
- Get statistical comparison with cross-validation
- **[Go to: Complete Evaluation Workflow](#complete-evaluation-workflow)**

### 🎯 **Path B: Test Single Model**
- Run just one specific model (Phishpedia, VisualPhish, or Baseline)
- Quick testing or debugging
- **[Go to: Single Model Testing](#single-model-testing)**

### 🎯 **Path C: Use Your Own Dataset**
- Evaluate models on your own phishing/benign images
- Custom data preparation and evaluation
- **[Go to: Custom Dataset Evaluation](#custom-dataset-evaluation)**

### 🎯 **Path D: Web Interface Demo**
- Interactive web demo for testing individual images
- **[Go to: Web Interface Setup](#web-interface-setup)**

---

## Prerequisites

Before starting, install these tools:

```bash
# Install Just (command runner)
# See: https://github.com/casey/just?tab=readme-ov-file#packages

# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/0.5.18/install.sh | sh
source $HOME/.local/bin/env

# Install unzip (for datasets)
# Usually pre-installed on most systems
```

**Initial Setup (Required for all paths):**
```bash
# Clone and enter project directory
cd /path/to/your/project

# Set up environment variable (IMPORTANT!)
export PROJECT_ROOT_DIR=$(pwd)
echo "export PROJECT_ROOT_DIR=$(pwd)" >> ~/.bashrc  # or ~/.zshrc
source ~/.bashrc  # or source ~/.zshrc

# Install project tools
just tools
```

---

## Complete Evaluation Workflow

This is the **recommended approach** for comprehensive model evaluation and comparison.

### Step 1: Download Datasets

```bash
# Download Phishpedia dataset (30k samples each)
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

# Download preprocessed VisualPhish data (for training)
uv run --with gdown gdown 1ewejN6qo3Bkb8IYSKeklU4GIlRHqPlUC -O - --quiet | tar zxvf - -C "$PROJECT_ROOT_DIR/data/interim"
```

### Step 2: Create Cross-Validation Splits

**This step is crucial** - it creates balanced train/test splits for robust evaluation:

```bash
cd $PROJECT_ROOT_DIR/src/cross_validation
just splits-links
```

**What this creates:**
- `$PROJECT_ROOT_DIR/data_splits/split_0/`, `split_1/`, `split_2/` directories
- Each split contains separate folders for each model format
- `train.csv` and `val.csv` files for each split
- Balanced class distribution across all splits

### Step 3: Set Up Weights & Biases (Required)

```bash
# Get your API key from https://wandb.ai/settings
uv run wandb login YOUR_API_KEY_HERE
```

### Step 4: Run All Models on All Splits

#### 4a. Run Phishpedia

```bash
cd $PROJECT_ROOT_DIR/src/models/phishpedia

# One-time setup
just setup
just extract-targetlist
just prepare

# Run on each cross-validation split
for split in 0 1 2; do
    echo "=== Running Phishpedia on split_${split} ==="
    uv run phishpedia.py \
        --folder $PROJECT_ROOT_DIR/data_splits/split_${split}/Phishpedia/images/val \
        --output_txt $PROJECT_ROOT_DIR/logs/phishpedia/split_${split}_results.txt \
        --log
done
```

#### 4b. Run VisualPhish

```bash
cd $PROJECT_ROOT_DIR

# Setup dependencies
uv sync --frozen
# For macOS users with TensorFlow issues:
# uv sync --extra macos

# Train and evaluate on each split
for split in 0 1 2; do
    echo "=== Training VisualPhish on split_${split} ==="

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

#### 4c. Run Baseline

```bash
cd $PROJECT_ROOT_DIR/src/models/baseline

for split in 0 1 2; do
    echo "=== Running Baseline on split_${split} ==="

    # Create index from training data
    uv run load.py \
        --images $PROJECT_ROOT_DIR/data_splits/split_${split}/Baseline/images/train \
        --index $PROJECT_ROOT_DIR/data/processed/baseline/split_${split}_index.faiss \
        --batch-size 256 \
        --overwrite

    # Query validation data
    uv run query.py \
        --images $PROJECT_ROOT_DIR/data_splits/split_${split}/Baseline/images/val \
        --index $PROJECT_ROOT_DIR/data/processed/baseline/split_${split}_index.faiss \
        --output $PROJECT_ROOT_DIR/logs/baseline/split_${split}_results.csv \
        --threshold 0.5 \
        --overwrite
done
```

### Step 5: Run Statistical Evaluation

After all models complete, run the evaluation:

```bash
cd $PROJECT_ROOT_DIR/src/eval
just evaluate
```

**Results will be in:** `$PROJECT_ROOT_DIR/src/eval/critical_difference_analysis_results/`
- `identification_rate_rankings.png` - Performance comparison
- `identification_rate_cd_diagram.png` - Statistical significance
- `identification_rate_summary.csv` - Detailed metrics

---

## Single Model Testing

If you want to test just one model quickly:

### Phishpedia Only

```bash
# Setup
cd $PROJECT_ROOT_DIR/src/models/phishpedia
just setup
just extract-targetlist
just prepare

# Run on provided dataset (no cross-validation)
uv run phishpedia.py \
    --folder $PROJECT_ROOT_DIR/data/raw/phishpedia/phish_sample_30k \
    --output_txt phishpedia_results.txt \
    --log
```

### VisualPhish Only

```bash
# Setup
cd $PROJECT_ROOT_DIR
uv sync --frozen
uv run wandb login YOUR_API_KEY

# Train on provided dataset
uv run trainer.py \
    --dataset-path $PROJECT_ROOT_DIR/data/interim/VisualPhish \
    --logdir $PROJECT_ROOT_DIR/logdir \
    --output-dir $PROJECT_ROOT_DIR/data/processed/VisualPhish

# Evaluate
uv run src/models/visualphishnet/eval_new.py \
    --emb-dir $PROJECT_ROOT_DIR/data/processed/VisualPhish \
    --threshold 8.0
```

### Baseline Only

```bash
cd $PROJECT_ROOT_DIR/src/models/baseline

# Create index from benign images
uv run load.py \
    --images $PROJECT_ROOT_DIR/data/interim/VisualPhish/trusted_list \
    --index $PROJECT_ROOT_DIR/data/processed/baseline/trusted_index.faiss \
    --batch-size 256

# Test against phishing images
uv run query.py \
    --images $PROJECT_ROOT_DIR/data/interim/VisualPhish/phishing \
    --index $PROJECT_ROOT_DIR/data/processed/baseline/trusted_index.faiss \
    --output $PROJECT_ROOT_DIR/logs/baseline/results.csv \
    --threshold 0.5 \
    --is-phish
```

---

## Custom Dataset Evaluation

To use your own phishing/benign images:

### Step 1: Prepare Your Data

Organize your data as:
```
my_dataset/
├── images/
│   ├── 00000.png
│   ├── 00001.jpg
│   └── ... (numerically ordered)
└── labels.txt  # One URL per line, matching image order
```

### Step 2: Convert to Model Formats

```bash
# Generate CSV for organization scripts
uv run scripts/prepare_data_for_organizer.py \
    --image-folder my_dataset/images/ \
    --labels-file my_dataset/labels.txt \
    --output-csv my_dataset/prepared_data.csv \
    --is-phishing  # Use --no-is-phishing for benign data

# Organize for Phishpedia
uv run src/organize_by_sample.py \
    --csv my_dataset/prepared_data.csv \
    --screenshots my_dataset/images/ \
    --output my_dataset/phishpedia_format

# Organize for VisualPhish
uv run src/organize_by_target.py \
    --csv my_dataset/prepared_data.csv \
    --screenshots my_dataset/images/ \
    --output my_dataset/visualphish_format
```

### Step 3: Update Configuration

Edit `config.json`:
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

### Step 4: Run Complete Evaluation

Follow **Steps 2-5** from [Complete Evaluation Workflow](#complete-evaluation-workflow).

---

## Web Interface Setup

Interactive demo for testing individual images:

### Step 1: Prepare Model Files

Ensure you have completed model training and have these files:
- `data/processed/VisualPhish/model2.h5`
- `data/processed/VisualPhish/whitelist_emb.npy`
- `data/processed/VisualPhish/whitelist_file_names.npy`
- `data/processed/VisualPhish/whitelist_labels.npy`
- `src/models/phishpedia/models/` (directory)
- `src/models/phishpedia/LOGO_FEATS.npy`
- `src/models/phishpedia/LOGO_FILES.npy`

### Step 2: Start Services

```bash
# Start backend services
docker-compose up -d

# Start web interface (in another terminal)
uv run streamlit run src/website.py
```

**Access:** Open http://localhost:8501 in your browser.

---

## Configuration Guide

### Cross-Validation Settings (`config.json`)

```json
{
  "cross_validation_config": {
    "enabled": true,          // Enable cross-validation
    "n_splits": 3,           // Number of folds
    "shuffle": true,         // Shuffle before splitting
    "random_state": 42,      // For reproducible results
    "output_splits_directory": "data_splits"
  }
}
```

### Dataset Configuration

```json
{
  "dataset_image_paths": {
    "visualphish": {
      "path": "data/raw/visualphish",
      "label_strategy": "labels_file",  // or "directory" or "subfolders"
      "target_mapping": {
        "phishing": "newly_crawled_phishing",
        "benign": "benign_test"
      }
    }
  }
}
```

**Label Strategies:**
- `"labels_file"`: Images + `labels.txt` file
- `"directory"`: Images organized in brand subdirectories
- `"subfolders"`: Each image in separate folder with `shot.png`

---

## Troubleshooting

### Common Issues

**"Command not found" errors:**
```bash
# Ensure PROJECT_ROOT_DIR is set
echo $PROJECT_ROOT_DIR  # Should show your project path
export PROJECT_ROOT_DIR=$(pwd)  # If empty

# Ensure tools are installed
just tools
```

**Dataset download failures:**
```bash
# Check internet connection and retry
# Ensure sufficient disk space (>10GB needed)
# Try downloading individual files manually
```

**Cross-validation splits look wrong:**
```bash
# Check dataset structure matches config.json
ls -la $PROJECT_ROOT_DIR/data/raw/visualphish/
ls -la $PROJECT_ROOT_DIR/data/raw/phishpedia/

# Verify target_mapping in config.json
```

**Models fail to run:**
```bash
# Ensure Weights & Biases is set up
uv run wandb login YOUR_API_KEY

# Check dependencies
uv sync --frozen

# For macOS TensorFlow issues:
uv sync --extra macos
```

**Evaluation fails:**
```bash
# Ensure all model results exist
ls -la $PROJECT_ROOT_DIR/logs/*/

# Check config.json paths match actual result locations
# Verify CSV files have correct column names
```

### Getting Help

1. **Check model-specific README files:** `src/models/*/README.md`
2. **Verify configuration:** Compare your `config.json` with provided examples
3. **Check file paths:** Ensure all paths in config match actual file locations
4. **Review logs:** Check console output for specific error messages

---

## Understanding Results

### Cross-Validation Output

After running the complete workflow, you'll get:

**Performance Rankings** (`identification_rate_rankings.png`):
- Shows average performance across all CV folds
- Higher is better for identification rate

**Statistical Significance** (`identification_rate_cd_diagram.png`):
- Models connected by horizontal lines are NOT significantly different
- Models far apart ARE significantly different

**Detailed Results** (`identification_rate_summary.csv`):
- Per-fold results for each model
- Statistical test results
- Confidence intervals

### Model-Specific Outputs

- **Phishpedia:** Text files with detected targets and confidence scores
- **VisualPhish:** CSV files with distance scores and classifications
- **Baseline:** CSV files with similarity scores and matches

---

## External Dataset Preparation (Advanced)

For detailed instructions on preparing external datasets, see:
- [External Dataset Guide](../external-datasets.md)
- [Cross-Validation README](../../src/cross_validation/README.md)

**When to use:** Only if you have raw image datasets that need conversion to the required formats.
