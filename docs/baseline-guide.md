# Baseline Phishing Detection - User Guide

A fast baseline phishing detection service using perceptual hashing and similarity search for phishing target recognition.

## Quick start

- Cross-validation: see [Cross-Validation Usage](#cross-validation-usage)
- Quick testing: see [Quick Testing](#quick-testing)
- Custom dataset: see [Custom Dataset Usage](#custom-dataset-usage)

---

## How It Works

The baseline model uses:
- perceptual hashing to fingerprint images
- FAISS for nearest-neighbor search
- a distance threshold to classify matches

Phishing pages often copy legitimate pages; similar perceptual hashes indicate potential phishing.

---

## Quick Testing

Test the baseline quickly on provided datasets:

### Step 1: Setup

```bash
cd $PROJECT_ROOT_DIR/src/models/baseline

# Ensure dependencies are installed
uv sync --frozen

# Download VisualPhish dataset (if not done already)
uv run --with gdown gdown 1ewejN6qo3Bkb8IYSKeklU4GIlRHqPlUC -O - --quiet | tar zxvf - -C "$PROJECT_ROOT_DIR/data/interim"
```

### Step 2: Create Index from Benign Images

```bash
# Create index from trusted/benign images
uv run load.py \
    --images "$PROJECT_ROOT_DIR/data/interim/VisualPhish/trusted_list" \
    --index "$PROJECT_ROOT_DIR/data/processed/baseline/trusted_index.faiss" \
    --labels "$PROJECT_ROOT_DIR/data/interim/VisualPhish/trusted_list/targets.txt" \
    --batch-size 256 \
    --overwrite \
    --log
```


### Step 3: Test Against Phishing Images

```bash
# Query phishing images against benign index
uv run query.py \
    --images "$PROJECT_ROOT_DIR/data/interim/VisualPhish/phishing" \
    --index "$PROJECT_ROOT_DIR/data/processed/baseline/trusted_index.faiss" \
    --output "$PROJECT_ROOT_DIR/logs/baseline/phishing_vs_benign_results.csv" \
    --labels "$PROJECT_ROOT_DIR/data/interim/VisualPhish/phishing/targets2.txt" \
    --threshold 0.5 \
    --is-phish \
    --overwrite \
    --log
```


### Step 4: Check Results

```bash
# View results
head -20 "$PROJECT_ROOT_DIR/logs/baseline/phishing_vs_benign_results.csv"
```

---

## Custom Dataset Usage

To use your own images:

### Step 1: Prepare Your Data

Organize as:
```
my_dataset/
├── benign_images/
│   ├── image1.png
│   ├── image2.jpg
│   └── ...
├── phishing_images/
│   ├── phish1.png
│   ├── phish2.jpg
│   └── ...
└── labels/ (optional)
    ├── benign_labels.txt
    └── phishing_labels.txt
```

### Step 2: Create Index from Benign Images

```bash
cd $PROJECT_ROOT_DIR/src/models/baseline

uv run load.py \
    --images my_dataset/benign_images/ \
    --index my_dataset/benign_index.faiss \
    --labels my_dataset/labels/benign_labels.txt \
    --batch-size 256 \
    --overwrite
```

### Step 3: Test Phishing Images

```bash
uv run query.py \
    --images my_dataset/phishing_images/ \
    --index my_dataset/benign_index.faiss \
    --output my_dataset/results.csv \
    --labels my_dataset/labels/phishing_labels.txt \
    --threshold 0.5 \
    --is-phish \
    --overwrite
```

---

## Understanding the Scripts

### `load.py` - Index Creation

**Purpose**: Create searchable index from a collection of images.

**Key Parameters**:
- `--images`: Directory containing images to index
- `--index`: Where to save the FAISS index file
- `--labels`: Optional file with target labels (one per line)
- `--is-phish`: Mark all images as phishing (default: benign)
- `--batch-size`: Process images in batches (default: 256)

**Output Files**:
- `*.faiss`: FAISS index with perceptual hash embeddings
- `*.csv`: Metadata with image paths and target labels

### `query.py` - Image Search

**Purpose**: Search for similar images in an existing index.

**Key Parameters**:
- `--images`: Directory containing query images
- `--index`: Path to existing FAISS index
- `--output`: CSV file for results
- `--threshold`: Distance threshold for classification
- `--is-phish`: Mark query images as phishing for evaluation
- `--top-k`: Number of similar images to return (default: 1)

**Output**: CSV with query image, best match, distance, and classifications.

---

## Configuration and Tuning

### Distance Threshold

The `--threshold` parameter is crucial for performance:

- **Lower threshold (e.g., 0.3)**: Stricter matching, fewer false positives
- **Higher threshold (e.g., 0.8)**: More lenient matching, fewer false negatives
- **Default (0.5)**: Balanced approach

**Tuning approach**:
1. Start with default (0.5)
2. If too many false positives → decrease threshold
3. If too many false negatives → increase threshold

### Batch Size

- **Larger batches**: Faster processing, more memory usage
- **Smaller batches**: Slower processing, less memory usage
- **Default (256)**: Good balance for most systems

---

## Expected Data Structure

### VisualPhish Format

```
VisualPhish/
├── phishing/
│   ├── Target_A/
│   │   ├── T0_0.png
│   │   └── T0_1.png
│   ├── Target_B/
│   │   └── T1_0.png
│   └── targets2.txt
└── trusted_list/
    ├── Target_A/
    │   ├── image1.png
    │   └── image2.png
    ├── Target_B/
    │   └── image3.png
    └── targets.txt
```

### Labels Files Format

```
# targets.txt or targets2.txt
Target_A
Target_B
Target_C
```

One target name per line, corresponding to subdirectory names.

---

## Troubleshooting

### Common Issues

**"FAISS index not found"**:
```bash
# Ensure index was created successfully
ls -la path/to/your/index.faiss

# Check load.py completed without errors
# Re-run load.py if needed
```

**"No images found"**:
```bash
# Check image directory exists and contains images
ls -la path/to/images/

# Supported formats: .jpg, .jpeg, .png
find path/to/images/ -name "*.jpg" -o -name "*.png"
```

## Integration with Main Project

The baseline automatically integrates with:

- **Evaluation pipeline**: Outputs results in expected format
- **Statistical analysis**: Results included in comparative analysis
- **Configuration**: Reads settings from main `config.json`

**No additional setup needed** when using the main project workflow.
