# Baseline Phishing Detection Service

A baseline phishing detection service that uses perceptual hashing (via the perception library) and FAISS vector similarity search to detect potential phishing attempts.

## Features

- Fast perceptual hash computation using the perception library
- Efficient similarity search using FAISS vector database
- Configurable distance threshold for phishing detection
- Target mapping support for brand name normalization
- FastAPI-based REST API interface
- Command-line tools for index creation and querying

## Scripts

### `load.py` - Index Creation Script

The `load.py` script processes images and creates a FAISS index with perceptual hash embeddings.

#### Usage

```bash
python load.py --images <images_dir> --index <index_path> [OPTIONS]
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

#### Examples

```bash
# Index benign images
python load.py --images /path/to/benign_images --index benign_index.faiss

# Index phishing images
python load.py --images /path/to/phishing_images --index phishing_index.faiss --is-phish

# Index with custom batch size and overwrite existing
python load.py --images /path/to/images --index my_index.faiss --batch-size 128 --overwrite
```

### `query.py` - Image Query Script

The `query.py` script searches for similar images in an existing FAISS index and outputs results to CSV.

#### Usage

```bash
python query.py --images <query_images_dir> --index <index_path> --output <results.csv> [OPTIONS]
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

#### Output Format

The query script outputs a CSV file with the following columns:

- `file`: Query image filename
- `baseline_class`: The `true_class` of the matched image from the index (0 or 1)
- `baseline_distance`: Distance score to the matched image
- `baseline_target`: The target/brand of the matched image
- `true_class`: The ground truth class of the query image (0 or 1, based on `--is-phish`)
- `true_target`: The target/brand of the query image (from labels or directory structure)

#### Examples

```bash
# Query benign images against an index
python query.py --images /path/to/benign_queries --index my_index.faiss --output results.csv

# Query phishing images for evaluation
python query.py --images /path/to/phishing_queries --index my_index.faiss --output results.csv --is-phish

# Query with custom parameters
python query.py --images /path/to/queries --index my_index.faiss --output results.csv --top-k 5 --threshold 2.0
```

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
