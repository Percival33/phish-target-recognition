# Baseline Phishing Detection - User Guide

A fast baseline phishing detection service using perceptual hashing and similarity search for phishing target recognition.

## 🚀 Quick Start - What Do You Want To Do?

### 🎯 **Path A: Cross-Validation Evaluation (Recommended)**
- Use with the main project's cross-validation workflow
- Automatically handled by main evaluation pipeline
- **[Go to: Cross-Validation Usage](#cross-validation-usage)**

### 🎯 **Path B: Quick Testing**
- Test baseline on provided datasets quickly
- Compare benign vs phishing images
- **[Go to: Quick Testing](#quick-testing)**

### 🎯 **Path C: Custom Dataset**
- Use your own images for baseline evaluation
- **[Go to: Custom Dataset Usage](#custom-dataset-usage)**

---

## How It Works

The baseline model uses:
- **Perceptual hashing**: Creates fingerprints of images that are similar for visually similar images
- **FAISS similarity search**: Fast nearest-neighbor search to find matching images
- **Distance threshold**: Images below threshold distance are considered matches

**Key insight**: Phishing pages often visually copy legitimate pages, so similar perceptual hashes indicate potential phishing.

---

## Cross-Validation Usage

**This is automatically handled** when you run the main project's complete evaluation workflow. The baseline will:

1. **Create index** from training images in each CV split
2. **Query validation images** against the training index
3. **Output results** in the format expected by the evaluation system

**No manual intervention needed** - just follow the main README's [Complete Evaluation Workflow](README-user-guide.md#complete-evaluation-workflow).

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

**What this does:**
- Processes all benign images in `trusted_list/`
- Creates perceptual hash embeddings
- Saves FAISS index for fast similarity search
- Marks all images as benign (`true_class = 0`)

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

**What this does:**
- Tests phishing images against the benign index
- Images with distance < 0.5 are flagged as "similar to benign" (potential false negatives)
- Images with distance ≥ 0.5 are correctly identified as "different from benign"
- Results saved to CSV with performance metrics

### Step 4: Check Results

```bash
# View results
head -20 "$PROJECT_ROOT_DIR/logs/baseline/phishing_vs_benign_results.csv"

# Check performance summary (if available)
tail -10 "$PROJECT_ROOT_DIR/logs/baseline/phishing_vs_benign_results.csv"
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

# Supported formats: .jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp
find path/to/images/ -name "*.jpg" -o -name "*.png"
```

**Poor performance**:
```bash
# Try different threshold values
uv run query.py ... --threshold 0.3  # Stricter
uv run query.py ... --threshold 0.7  # More lenient

# Check if images are actually similar visually
# Baseline works best when phishing pages copy legitimate designs
```

**Memory issues**:
```bash
# Reduce batch size
uv run load.py ... --batch-size 128
uv run query.py ... --batch-size 128
```

### Performance Expectations

- **Speed**: Very fast (hundreds of images per second)
- **Memory**: Low memory usage compared to deep learning models
- **Accuracy**: Good for visually similar phishing pages, limited for sophisticated attacks

### When Baseline Works Well

✅ **Good for**:
- Phishing pages that copy legitimate page layouts
- Large-scale screening of similar-looking pages
- Fast initial filtering before more sophisticated analysis

❌ **Limited for**:
- Sophisticated phishing with original designs
- Pages with different layouts but same target brand
- Text-based phishing detection

---

## Integration with Main Project

The baseline automatically integrates with:

- **Cross-validation**: Handles CV splits automatically
- **Evaluation pipeline**: Outputs results in expected format
- **Statistical analysis**: Results included in comparative analysis
- **Configuration**: Reads settings from main `config.json`

**No additional setup needed** when using the main project workflow.
