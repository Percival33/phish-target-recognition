# Cross-Validation Data Splits

Robust scripts for creating and validating cross-validation data splits.

## Architecture

- **Protocol-based design**: Uses Python protocols for dependency inversion
- **Single Responsibility**: Each class has one clear purpose
- **Composition over inheritance**: Components are composed together
- **Extensible validation**: Easy to add new validators

## Core Components

### Data Processing (`cv_splits.py`)
- **`ImageFileDiscoverer`**: Discovers image files using configurable extensions
- **`LabelExtractor`**: Extracts labels from `labels.txt` files
- **`SimpleDataProcessor`**: Processes datasets using multiple label strategies
- **`StratifiedSplitGenerator`**: Creates stratified K-fold splits
- **`CSVFileWriter`**: Writes training and validation CSV files with algorithm columns
- **`PerSampleSymlinkManager`**: Creates organized symlinks for image organization
- **`CrossValidationSplitter`**: Main orchestrator class using dependency injection

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

### Image File Discovery

The system discovers images using these file extensions:
- `*.jpg`, `*.jpeg`, `*.png`, `*.gif`
- `*.bmp`, `*.tiff`, `*.webp`
- Case-insensitive matching (both uppercase and lowercase)

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

### CSV Files
- **Training CSV** (`train.csv`): Contains `file`, `true_target`, `true_class` columns
- **Validation CSV** (`val.csv`): Contains training columns plus empty prediction columns for each algorithm (e.g., `pp_target`, `pp_class`, `vp_target`, `vp_class`)

### Directory Structure
```
data_splits/
├── split_0/
│   ├── VisualPhish/
│   │   ├── train.csv        # file, true_target, true_class
│   │   ├── val.csv          # + prediction columns (configurable prefixes)
│   │   └── images/          # symlinks (optional)
│   │       ├── train/       # training images organized by class then target
│   │       │   ├── benign/
│   │       │   │   ├── target1/  # e.g., specific benign sites
│   │       │   │   └── target2/
│   │       │   └── phish/
│   │       │       ├── target1/  # e.g., specific brands/targets
│   │       │       └── target2/
│   │       └── val/         # validation images organized by target only
│   │           ├── target1/ # mixed benign/phish samples by target
│   │           ├── target2/
│   │           └── target3/
│   └── Phishpedia/
│       ├── train.csv
│       ├── val.csv
│       └── images/
│           ├── train/
│           │   ├── benign/
│           │   │   └── [target_subdirs]/
│           │   └── phish/
│           │       └── [target_subdirs]/
│           └── val/
│               └── [target_subdirs]/
├── split_1/
└── split_2/
```

**Note**:
- Training symlinks are organized by class first (`benign/`/`phish/`), then by target
- Validation symlinks are organized by target only (no class segregation)
- Benign training folders will contain multiple target subdirectories when specific targets can be determined
- If benign targets cannot be determined, samples are grouped under a single "benign" folder

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

**Note**: The `csv_column_prefixes` in `data_input_config` determine the algorithm column prefixes used in validation CSV files (e.g., `pp_target`, `vp_class`). These columns are automatically added as empty strings to validation CSV files.

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

### Target Label Handling

**Benign Target Discovery**: The system attempts to determine specific target names for benign samples:
- Uses actual target labels from `labels.txt` files when available
- Falls back to subdirectory names when meaningful
- Defaults to generic "benign" label only when specific targets cannot be determined

**Duplicate Target Validation**: The system validates that no target name appears in both phishing and benign classes within the same dataset. If duplicates are found, an error is raised: `"Duplicate target found in dataset {name}: {targets}"`

### Label Discovery Strategies

**`directory`**: Images organized in brand/target subdirectories
```
dataset/phishing/brand1/image1.jpg
dataset/phishing/brand2/image2.jpg
dataset/benign/benign_site/image3.jpg
```
- Uses subdirectory names as `true_target` values
- Automatically assigns `true_class` based on phishing/benign mapping

**`labels_file`**: Flat structure with `labels.txt`
```
dataset/phishing/image1.jpg
dataset/phishing/labels.txt  # one label per line matching sorted image order
dataset/benign/image2.jpg
dataset/benign/labels.txt    # optional for benign
```
- Matches labels to images by sorted filename order
- For benign samples: uses actual label from `labels.txt` if available, otherwise defaults to "benign"

**`subfolders`**: Nested subdirs each containing `shot.png`
```
dataset/phishing/brand1+timestamp/shot.png
dataset/phishing/brand2+timestamp/shot.png
dataset/benign/benign_folder/shot.png
```
- Each subfolder must contain exactly one `shot.png` file
- Uses subfolder name as `true_target` or reads from individual `labels.txt` files
- For benign: uses subfolder name, or defaults to "benign"

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
- Discover images using the specified strategy and supported extensions
- Apply target mapping for consistent labeling
- Include the dataset in stratified cross-validation splits
- Generate corresponding CSV files with appropriate column structure
- Create organized symlinks (if requested) with proper directory hierarchy
