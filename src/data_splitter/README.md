# Data Splitter

Dataset splitting tool for creating stratified 60:20:20 train/validation/test splits.

## Installation

```bash
# Navigate to data_splitter directory
cd src/data_splitter

# Install dependencies
uv sync
```

## Usage

```bash
# Run with uv (recommended)
uv run split_data.py config.json

# Or activate virtual environment and run directly
uv run python split_data.py config.json

# Alternative syntax with --config flag
uv run split_data.py --config /path/to/config.json
```

## Configuration Format

Configuration should be placed under a `data_split` key in your JSON file:

```json
{
  "data_split": {
    "random_state": 42,
    "output_directory": "data_splits",
    "datasets": {
      "dataset_name": {
        "path": "path/to/data",
        "label_strategy": "subfolders|labels_file|directory",
        "target_mapping": {
          "phishing": "phish_subdirectory",
          "benign": "benign_subdirectory"
        },
        "create_symlinks": true|false
      }
    }
  }
}
```

### Label Strategies

- **subfolders**: Each sample in its own subdirectory with `shot.png`
- **labels_file**: Flat structure with `labels.txt` file
- **directory**: Images organized in class subdirectories

### Output Directory Path Resolution

- **Relative paths**: Prefixed with `PROJECT_ROOT_DIR` (e.g., `"data_splits"` → `$PROJECT_ROOT/data_splits`)
- **Absolute paths**: Used as-is (e.g., `"/tmp/splits"` → `/tmp/splits`)

## Output

Creates `train.csv`, `val.csv`, `test.csv` files with consistent format:

```csv
file,true_target,true_class
/path/to/image1.jpg,amazon,1
/path/to/image2.jpg,benign,0
```

### Split Proportions
- **Train**: 60%
- **Validation**: 20%
- **Test**: 20%

All splits maintain class distribution through stratified sampling.

## Features

- **Stratified sampling**: Maintains class balance across all splits
- **Path resolution**: Smart handling of relative vs absolute output paths
- **Symlink support**: Optional creation of organized image directories
- **Comprehensive logging**: Uses project-wide logging configuration
- **Error handling**: Validates data requirements and provides clear error messages
- **CLI interface**: Flexible command-line argument parsing

## Requirements

- Python 3.9+
- Dependencies managed via `uv` (see `pyproject.toml`)

## Development

```bash
# Install in development mode
uv sync --dev

# Run tests (when available)
uv run pytest

# Format code
uv run black .
```
