# Cross-Validation Data Splits

Robust scripts for creating and validating cross-validation data splits.

## Architecture

- **Protocol-based design**: Uses Python protocols for dependency inversion
- **Single Responsibility**: Each class has one clear purpose
- **Composition over inheritance**: Components are composed together
- **Extensible validation**: Easy to add new validators

## Core Components

### Data Processing (`cv_splits.py`)
- **`SimpleDataProcessor`**: Processes datasets and discovers image files
- **`StratifiedSplitGenerator`**: Creates stratified K-fold splits
- **`CSVFileWriter`**: Writes training and validation CSV files
- **`PerSampleSymlinkManager`**: Creates symlinks for image organization
- **`CrossValidationSplitter`**: Main orchestrator class

### Validation (`cv_validator.py`)
- **`DirectoryStructureValidator`**: Validates split directory structure
- **`CSVFilesValidator`**: Validates CSV file format and content
- **`ClassBalanceValidator`**: Checks class distribution across splits
- **`SymlinkValidator`**: Validates symlink integrity
- **`CrossValidationValidator`**: Main validation orchestrator

### Common Utilities (`common.py`)
- **`ConfigLoader`**: Loads and parses JSON configuration
- **`DirectoryIterator`**: Utilities for directory traversal
- **`PathUtils`**: Common path operations
- **`CVArgumentParser`**: Shared argument parsing
- **`CVConstants`**: System constants and patterns

## Scripts

### `cv_splits.py` - Create Data Splits
```bash
uv run python cv_splits.py [options]
just splits              # Basic splits
just splits-links        # With symlinks
just splits-config file  # Custom config
```

**Options:**
- `--config`: Config file path (default: `config.json`)
- `--splits-dir`: Output directory (overrides config)
- `--create-symlinks`: Create symlinks to original images

### `cv_validator.py` - Validate Splits
```bash
uv run python cv_validator.py [options]
just validate                # Basic validation
just validate-config file    # Custom config
```

**Options:**
- `--config`: Config file path (default: `config.json`)
- `--splits-dir`: Splits directory (overrides config)

## Project Setup

### Dependencies
- **Python**: >=3.9
- **numpy**: >=2.0.2
- **pandas**: >=2.3.0
- **scikit-learn**: >=1.6.1
- **tools**: Custom utilities package (included as wheel)

### Installation
```bash
just setup-cv    # Setup environment and dependencies
uv sync          # Or manually with UV
```

### Development Commands
```bash
just test        # Setup and run basic test
just test-links  # Full test with symlinks
just clean       # Clean generated splits and environment
just clean-data  # Clean only generated splits
just clean-env   # Clean only environment artifacts
```

## Output Structure

```
data_splits/
├── split_0/
│   ├── VisualPhish/
│   │   ├── train.csv        # file, true_target, true_class
│   │   ├── val.csv          # + prediction columns (configurable prefixes)
│   │   └── images/          # symlinks (optional)
│   │       ├── train/       # training images
│   │       └── val/         # validation images
│   └── Phishpedia/
│       ├── train.csv
│       ├── val.csv
│       └── images/
├── split_1/
└── split_2/
```

## Adding Dataset Folders

To add new datasets to the cross-validation process, update the configuration:

### Configuration Format

```json
{
  "data_input_config": {
    "csv_column_prefixes": {
      "Phishpedia": "pp",
      "VisualPhish": "vp",
      "Baseline": "baseline"
    }
  },
  "cross_validation_config": {
    "enabled": true,
    "n_splits": 3,
    "shuffle": true,
    "random_state": 42,
    "output_splits_directory": "data_splits",
    "dataset_image_paths": {
      "DatasetName": {
        "path": "/path/to/dataset",
        "label_strategy": "directory|labels_file|subfolders",
        "target_mapping": {
          "phishing": "phishing_dir",
          "benign": "benign_dir"
        }
      }
    }
  }
}
```

**Note**: The `csv_column_prefixes` in `data_input_config` determine the algorithm column prefixes used in validation CSV files (e.g., `pp_target`, `vp_class`).

### Target Mapping

`target_mapping` provides a canonical class abstraction layer that maps standardized class names (`"phishing"`, `"benign"`) to dataset-specific directory structures. This decouples the cross-validation logic from heterogeneous dataset naming conventions.

**Purpose**: Enables uniform binary classification across datasets with different organizational schemas without dataset-specific code branches.

**Example mappings**:
```json
"VisualPhish": {
  "target_mapping": {
    "phishing": "newly_crawled_phishing",  // maps to actual dir name
    "benign": "benign_test"               // maps to actual dir name
  }
}
```

### Label Discovery Strategies

**`directory`**: Images organized in brand/target subdirectories
```
dataset/phishing/brand1/image1.jpg
dataset/phishing/brand2/image2.jpg
dataset/benign/benign_site/image3.jpg
```

**`labels_file`**: Flat structure with `labels.txt`
```
dataset/phishing/image1.jpg
dataset/phishing/labels.txt  # one label per line matching sorted image order
dataset/benign/image2.jpg
```

**`subfolders`**: Nested subdirs each containing `shot.png`
```
dataset/phishing/brand1+timestamp/shot.png
dataset/phishing/brand2+timestamp/shot.png
dataset/benign/benign_folder/shot.png
```

### Example: Adding New Dataset

```json
{
  "NewDataset": {
    "path": "/data/my_dataset",
    "label_strategy": "directory",
    "target_mapping": {
      "phishing": "malicious",
      "benign": "legitimate"
    }
  }
}
```

The system will automatically:
- Discover images using the specified strategy
- Apply target mapping for consistent labeling
- Include the dataset in cross-validation splits
- Generate corresponding CSV files and symlinks
