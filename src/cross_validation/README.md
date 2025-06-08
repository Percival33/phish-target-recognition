# Cross-Validation Data Splits

This package provides robust scripts for creating and validating cross-validation data splits.

## Architecture

The system is built with clean architecture patterns:

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
Creates stratified train/val splits with optional symlink creation.

```bash
# Using UV (recommended)
uv run python cv_splits.py [options]

# Using just commands
just splits              # Basic splits
just splits-links        # With symlinks
just splits-config file  # Custom config
```

**Options:**
- `--config`: Config file path (default: `config.json`)
- `--splits-dir`: Output directory (overrides config)
- `--create-symlinks`: Create symlinks to original images

**Features:**
- Supports multiple dataset structures (direct images, Phishpedia subdirs)
- Extracts labels from `labels.txt` files
- Creates train.csv (simple format) and val.csv (with prediction columns)
- Optional symlink creation for organized image access

### `cv_validator.py` - Validate Splits
Comprehensive validation of generated splits.

```bash
# Using UV (recommended)
uv run python cv_validator.py [options]

# Using just commands
just validate                # Basic validation
just validate-config file    # Custom config
```

**Options:**
- `--config`: Config file path (default: `config.json`)
- `--splits-dir`: Splits directory (overrides config)

**Validations:**
- Directory structure completeness
- CSV file format and required columns
- Class balance across splits
- Symlink integrity (if present)
- File existence verification

## Project Setup

### Dependencies
- **Python**: >=3.9
- **numpy**: >=2.0.2
- **pandas**: >=2.3.0
- **scikit-learn**: >=1.6.1
- **tools**: Custom utilities package (included as wheel)

### Installation
```bash
# Setup environment and dependencies
just setup-cv

# Or manually with UV
uv sync
```

### Development Commands
```bash
# Setup and run basic test
just test

# Full test with symlinks
just test-full

# Clean generated splits
just clean

# Clean entire environment
just clean-cv
```

## Output Structure

```
data_splits/
├── split_0/
│   ├── VisualPhish/
│   │   ├── train.csv        # file, true_target, true_class
│   │   ├── val.csv          # + prediction columns (pp_, vp_, baseline_)
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

## Configuration Format

The system expects a JSON configuration with:

```json
{
  "cross_validation_config": {
    "enabled": true,
    "n_splits": 3,
    "shuffle": true,
    "random_state": 42,
    "output_splits_directory": "data_splits",
    "dataset_image_paths": {
      "DatasetName": {
        "path": "/path/to/dataset",
        "label_strategy": "directory",
        "target_mapping": {
          "phishing": "phishing_dir",
          "benign": "benign_dir"
        }
      }
    }
  }
}
```

## SOLID Principles Applied

### Single Responsibility Principle
- Each validator handles one type of validation
- Data processor focuses only on data loading
- File writer handles only CSV generation

### Open/Closed Principle
- New validators can be added without modifying existing code
- Protocol-based interfaces allow easy extensions

### Liskov Substitution Principle
- All validators implement the same `Validator` protocol
- Components can be swapped without breaking functionality

### Interface Segregation Principle
- Small, focused protocols (DataProcessor, SplitGenerator, etc.)
- Clients depend only on methods they use

### Dependency Inversion Principle
- High-level modules depend on abstractions (protocols)
- Concrete implementations are injected at runtime

## Integration

**Training Scripts**:
```python
# Read training data
train_df = pd.read_csv("data_splits/split_0/dataset/train.csv")
# Columns: file, true_target, true_class
```

**Evaluation Scripts**:
```python
# Read evaluation data
val_df = pd.read_csv("data_splits/split_0/dataset/val.csv")
# Add predictions to pp_target, vp_target, baseline_target columns
```

## Architecture Benefits

- **Maintainable**: Clear separation of concerns
- **Testable**: Each component can be tested in isolation
- **Extensible**: Easy to add new validators or processors
- **Robust**: Comprehensive validation and error handling
- **Flexible**: Protocol-based design allows easy substitution

**Total LOC:** ~760 lines across 3 main files
**Test Coverage**: Comprehensive validation suite
**Extensibility**: High - new features can be added without breaking changes
