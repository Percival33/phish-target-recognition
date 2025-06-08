# Baseline Phishing Detection Service

A baseline phishing detection service that uses perceptual hashing (via the perception library) and FAISS vector similarity search to detect potential phishing attempts.

## Features

- Fast perceptual hash computation using the perception library
- Efficient similarity search using FAISS vector database
- Configurable distance threshold for phishing detection
- Target mapping support for brand name normalization
- FastAPI-based REST API interface
- Command-line tools for index creation and querying

## Prerequisites and Setup

### 1. Install Required Tools
Ensure you have the following installed:
- **Just**: Command runner for project tasks
- **uv**: Python package and environment manager

### 2. Environment Configuration
```bash
# Set PROJECT_ROOT_DIR environment variable
export PROJECT_ROOT_DIR=$(pwd)

# Add to shell config for persistence
echo "export PROJECT_ROOT_DIR=$(pwd)" >> ~/.zshrc
source ~/.zshrc
```

### 3. Install Development Tools and Dependencies
```bash
# Install development tools
just tools

# Synchronize Python dependencies
uv sync --frozen
```

### 4. Dataset Preparation (VisualPhish)
```bash
# Download VisualPhish dataset
uv run --with gdown gdown 1ewejN6qo3Bkb8IYSKeklU4GIlRHqPlUC -O - --quiet | tar zxvf - -C "$PROJECT_ROOT_DIR/data/interim"
```

Expected data structure at `$PROJECT_ROOT_DIR/data/interim/VisualPhish`:
```
VisualPhish/
├── phishing/
│   ├── Target_A/
│   │   ├── T0_0.png
│   │   └── T0_1.png
│   ├── Target_B/
│   │   └── T1_0.png
│   └── targets2.txt  # Phishing target names
└── trusted_list/
    ├── Target_A/
    │   ├── image1.png
    │   └── image2.png
    ├── Target_B/
    │   └── image3.png
    └── targets.txt  # Benign target names
```

## Scripts

### `load.py` - Index Creation Script

The `load.py` script processes images and creates a FAISS index with perceptual hash embeddings.

#### Usage

```bash
uv run src/models/baseline/load.py --images <images_dir> --index <index_path> [OPTIONS]
```

#### Command-line Arguments

- `--images` (required): Path to directory containing images to index
- `--index` (required): Path where the FAISS index will be saved
- `--labels`: Path to labels.txt file containing target labels (optional)
- `--is-phish`: Flag to mark all processed images as phishing (optional, default: False)
- `--batch-size`: Batch size for processing (optional, default: 256)
- `--overwrite`: Overwrite existing index if it exists (optional)
- `--log`: Enable wandb logging (optional)

#### The `--is-phish` Flag for `load.py`

This flag determines the `true_class` value stored in the metadata for **all processed images**:

- **If present** (`--is-phish`): All images are marked as phishing (`true_class = 1`)
- **If absent** (default): All images are marked as benign (`true_class = 0`)

This class information is stored alongside each image's embedding in the metadata and is used during querying for evaluation purposes.

#### Complete Examples

**Create benign images index from VisualPhish trusted_list:**
```bash
# Create output directory
mkdir -p "$PROJECT_ROOT_DIR/data/processed/baseline/"

# Index all benign images from trusted_list
uv run src/models/baseline/load.py \
    --images "$PROJECT_ROOT_DIR/data/interim/VisualPhish/trusted_list" \
    --index "$PROJECT_ROOT_DIR/data/processed/baseline/trusted_index.faiss" \
    --labels "$PROJECT_ROOT_DIR/data/interim/VisualPhish/trusted_list/targets.txt" \
    --batch-size 256 \
    --log \
    --overwrite
```

**Create phishing images index from VisualPhish phishing directory:**
```bash
# Index all phishing images from phishing directory
uv run src/models/baseline/load.py \
    --images "$PROJECT_ROOT_DIR/data/interim/VisualPhish/phishing" \
    --index "$PROJECT_ROOT_DIR/data/processed/baseline/phishing_index.faiss" \
    --labels "$PROJECT_ROOT_DIR/data/interim/VisualPhish/phishing/targets2.txt" \
    --is-phish \
    --batch-size 256 \
    --log \
    --overwrite
```

#### Expected Output Files
After successful execution, verify these files exist:
- `*.faiss` - FAISS index file containing perceptual hash embeddings
- `*.csv` - Metadata file with image paths, target labels, and embedding information

### `query.py` - Image Query Script

The `query.py` script searches for similar images in an existing FAISS index and outputs results to CSV.

#### Usage

```bash
uv run src/models/baseline/query.py --images <query_images_dir> --index <index_path> --output <results.csv> [OPTIONS]
```

#### Command-line Arguments

- `--images` (required): Path to directory containing query images
- `--index` (required): Path to existing FAISS index
- `--output` (required): Path to output CSV file for results
- `--labels`: Path to labels.txt file containing target labels for query images (optional)
- `--threshold`: Distance threshold for classification (optional)
- `--top-k`: Number of similar images to return per query (optional, default: 1)
- `--batch-size`: Batch size for processing (optional, default: 256)
- `--overwrite`: Overwrite existing output file (optional)
- `--is-phish`: Flag to mark all query images as phishing for evaluation (optional, default: False)
- `--log`: Enable wandb logging (optional)

#### The `--is-phish` Flag for `query.py`

This flag sets the **ground truth class** for **all query images** used in evaluation:

- **If present** (`--is-phish`): All query images are treated as phishing (`true_class = 1`)
- **If absent** (default): All query images are treated as benign (`true_class = 0`)

This `true_class` for the query image is included in the output CSV, allowing for evaluation by comparing it against the `baseline_class` (the `true_class` of the matched image from the index).

#### Complete Examples and Use Cases

**1. Phishing Detection Evaluation (Query benign images against phishing index):**
```bash
# Test if benign images are incorrectly flagged as phishing
uv run src/models/baseline/query.py \
    --images "$PROJECT_ROOT_DIR/data/interim/VisualPhish/trusted_list" \
    --index "$PROJECT_ROOT_DIR/data/processed/baseline/phishing_index.faiss" \
    --output "$PROJECT_ROOT_DIR/data/processed/baseline/benign_vs_phishing_results.csv" \
    --labels "$PROJECT_ROOT_DIR/data/interim/VisualPhish/trusted_list/targets.txt" \
    --threshold 0.5 \
    --top-k 1 \
    --batch-size 256 \
    --overwrite
```

**2. Phishing Classification (Query unknown images against phishing index):**
```bash
# Classify unknown images as phishing/benign based on distance threshold
uv run src/models/baseline/query.py \
    --images "/path/to/unknown/images" \
    --index "$PROJECT_ROOT_DIR/data/processed/baseline/phishing_index.faiss" \
    --output "$PROJECT_ROOT_DIR/data/processed/baseline/classification_results.csv" \
    --threshold 0.5 \
    --top-k 1 \
    --batch-size 256 \
    --overwrite
```

**3. Similarity Search (Find similar phishing samples):**
```bash
# Query phishing images against phishing index for similarity analysis
uv run src/models/baseline/query.py \
    --images "$PROJECT_ROOT_DIR/data/interim/VisualPhish/phishing" \
    --index "$PROJECT_ROOT_DIR/data/processed/baseline/phishing_index.faiss" \
    --output "$PROJECT_ROOT_DIR/data/processed/baseline/phishing_similarity_results.csv" \
    --labels "$PROJECT_ROOT_DIR/data/interim/VisualPhish/phishing/targets2.txt" \
    --is-phish \
    --top-k 5 \
    --batch-size 256 \
    --overwrite
```

#### Distance Interpretation and Threshold Guidance

- **Lower distances = Higher similarity**
- **Classification Logic**: Images with distance below threshold are classified as phishing
- **Threshold Tuning**: Test different threshold values (e.g., 0.3, 0.5, 1.0) for optimal classification performance
- **Typical Range**: Distance values usually range from 0.0 (identical) to 4.0+ (very different)

#### Output Format

The query script outputs a CSV file with the following columns:

- `file`: Query image filename
- `baseline_class`: The `true_class` of the matched image from the index (0 or 1)
- `baseline_distance`: Distance score to the matched image
- `baseline_target`: The target/brand of the matched image
- `true_class`: The ground truth class of the query image (0 or 1, based on `--is-phish`)
- `true_target`: The target/brand of the query image (from labels or directory structure)

### `scripts/evaluate_baseline.py` - Evaluation Script

The evaluation script processes CSV output from `query.py` and calculates comprehensive classification and target identification metrics.

#### Usage

```bash
# Basic evaluation
uv run python scripts/evaluate_baseline.py results.csv

# Evaluation with ROC curve plot
uv run python scripts/evaluate_baseline.py results.csv --plot
```

#### Command-line Arguments

- `csv_path` (required): Path to CSV file containing query results (output from `query.py`)
- `--plot`: Optional flag to generate ROC curve plot

#### Output Metrics

The script calculates two sets of metrics:

**Class Classification Metrics:**
- F1 weighted score, ROC AUC, Matthews Correlation Coefficient
- Macro-averaged precision and recall

**Target Identification Metrics:**
- Target F1 scores (micro, macro, weighted)
- Target Matthews Correlation Coefficient
- Target precision and recall
- Identification rate (correctly identified phishing targets)

#### Examples

```bash
# Evaluate query results
uv run python scripts/evaluate_baseline.py output/query_results.csv

# Evaluate with ROC curve visualization
uv run python scripts/evaluate_baseline.py output/query_results.csv --plot
```

## Complete Workflow Examples

### Workflow 1: Build Benign Index and Evaluate Against Phishing

```bash
# 1. Create benign images index
uv run src/models/baseline/load.py \
    --images "$PROJECT_ROOT_DIR/data/interim/VisualPhish/trusted_list" \
    --index "$PROJECT_ROOT_DIR/data/processed/baseline/trusted_index.faiss" \
    --labels "$PROJECT_ROOT_DIR/data/interim/VisualPhish/trusted_list/targets.txt" \
    --overwrite

# 2. Query phishing images against benign index
uv run src/models/baseline/query.py \
    --images "$PROJECT_ROOT_DIR/data/interim/VisualPhish/phishing" \
    --index "$PROJECT_ROOT_DIR/data/processed/baseline/trusted_index.faiss" \
    --output "$PROJECT_ROOT_DIR/data/processed/baseline/phishing_vs_benign_results.csv" \
    --labels "$PROJECT_ROOT_DIR/data/interim/VisualPhish/phishing/targets2.txt" \
    --is-phish \
    --threshold 1.0 \
    --overwrite

# 3. Evaluate results
uv run python scripts/evaluate_baseline.py "$PROJECT_ROOT_DIR/data/processed/baseline/phishing_vs_benign_results.csv" --plot
```

### Workflow 2: Build Phishing Index for Classification

```bash
# 1. Create phishing images index
uv run src/models/baseline/load.py \
    --images "$PROJECT_ROOT_DIR/data/interim/VisualPhish/phishing" \
    --index "$PROJECT_ROOT_DIR/data/processed/baseline/phishing_index.faiss" \
    --labels "$PROJECT_ROOT_DIR/data/interim/VisualPhish/phishing/targets2.txt" \
    --is-phish \
    --overwrite

# 2. Test classification with benign images
uv run src/models/baseline/query.py \
    --images "$PROJECT_ROOT_DIR/data/interim/VisualPhish/trusted_list" \
    --index "$PROJECT_ROOT_DIR/data/processed/baseline/phishing_index.faiss" \
    --output "$PROJECT_ROOT_DIR/data/processed/baseline/classification_results.csv" \
    --labels "$PROJECT_ROOT_DIR/data/interim/VisualPhish/trusted_list/targets.txt" \
    --threshold 0.5 \
    --overwrite

# 3. Evaluate classification performance
uv run python scripts/evaluate_baseline.py "$PROJECT_ROOT_DIR/data/processed/baseline/classification_results.csv" --plot
```

## True Class Values

For both stored metadata (via `load.py`) and query image evaluation (via `query.py`):

- **`true_class = 0`**: Signifies a **benign** image
- **`true_class = 1`**: Signifies a **phishing** image

These integer values are used consistently throughout the system for classification and evaluation purposes.

## File Structure

- `BaselineServing.py` - Main service implementation
- `BaselineEmbedder.py` - Core embedding and search functionality
- `load.py` - Command-line tool for creating FAISS indices
- `query.py` - Command-line tool for querying indices
- `config/target_mappings.json` - Brand name normalization mappings
- `index/faiss_index.idx` - Pre-built FAISS index (mounted at runtime)

## Configuration

The service can be configured using environment variables or command line arguments:

- `PORT` - Service port (default: 8888)
- `DISTANCE_THRESHOLD` - Similarity threshold for phishing detection (default: 1.0)

## API

### POST /predict

Endpoint for phishing detection predictions.

Request body:
```json
{
    "url": "string",
    "image": "base64 encoded image"
}
```

Response:
```json
{
    "url": "string",
    "class": "int (0 or 1)",
    "target": "string (brand name)",
    "confidence": "float",
    "baseline_distance": "float"
}
```

## Docker

The service is containerized and can be run using Docker:

```bash
docker build -t baseline-service .
docker run -p 8888:8888 \
    -v ./index:/code/index \
    -v ./config:/code/config \
    -e DISTANCE_THRESHOLD=1.0 \
    baseline-service
```
