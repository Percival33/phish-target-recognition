# External datasets

<!-- TOC -->
* [External datasets](#external-datasets)
  * [Prerequisites](#prerequisites)
  * [How to prepare dataset to use Phishpedia model?](#how-to-prepare-dataset-to-use-phishpedia-model)
    * [Evaluation for Phishpedia](#evaluation-for-phishpedia)
      * [Preparing Evaluation Data with `organize.py`](#preparing-evaluation-data-with-organizepy)
    * [Data Preparation for Phishpedia (`organize_by_sample.py`)](#data-preparation-for-phishpedia-organize_by_samplepy)
      * [CSV File Requirements](#csv-file-requirements)
  * [How to prepare data for VisualPhish model?](#how-to-prepare-data-for-visualphish-model)
    * [Data Preparation for VisualPhish (`organize_by_target.py`)](#data-preparation-for-visualphish-organize_by_targetpy)
<!-- TOC -->

## Prerequisites

Before using the data preparation scripts, ensure you have the following:

- **Python 3.9+** with required dependencies
- **uv** package manager installed
- Project dependencies installed: `uv sync --frozen`

All scripts use inline dependency declarations and can be run with `uv run` without additional setup.

## How to prepare dataset to use Phishpedia model?

This section covers the different methods for preparing external datasets to work with the Phishpedia model. The choice
of method depends on your data format and intended use case.

### Evaluation for Phishpedia

To prepare data for cross-model or cross-dataset performance evaluation, particularly when using Phishpedia's
organization scripts, you can use scripts to organize your datasets into a consistent format. For example, the
`src/models/phishpedia/scripts/organize.py` script can be used to categorize images based on labels.

#### Preparing Evaluation Data with `organize.py`

The `organize.py` script takes a directory of images, a file containing labels for these images, and an output
directory. It then creates subdirectories in the output path, named according to the labels (or a mapping defined in the
script, like using a domain name for a brand), and copies the corresponding images into these subdirectories.

**Example Command:**

Suppose you have a dataset of phishing images located in `datasets/VisualPhish/newly_crawled_phishing/` and a
corresponding `labels.txt` file in the same directory. To process this data and save the organized output to a directory
named `evaluation_dataset_prepared`, you would run the following command from the project root:

```bash
uv run src/models/phishpedia/scripts/organize.py datasets/VisualPhish/newly_crawled_phishing/ datasets/VisualPhish/newly_crawled_phishing/labels.txt ./evaluation_dataset_prepared
```

**Explanation of Arguments:**

- `src/models/phishpedia/scripts/organize.py`: The path to the organization script.
- `datasets/VisualPhish/newly_crawled_phishing/`: The input directory containing the image files (e.g., `000.png`,
  `001.png`, ...).
- `datasets/VisualPhish/newly_crawled_phishing/labels.txt`: The path to the text file where each line is a label
  corresponding to an image.
- `./evaluation_dataset_prepared`: The output directory where the script will create folders for each label and copy the
  images into them.

This structured dataset can then be used for evaluating the performance of different models. Ensure your `labels.txt`
file is correctly formatted, with each label on a new line, corresponding to the numerically ordered image files in your
input directory.

### Data Preparation for Phishpedia (`organize_by_sample.py`)

> [!IMPORTANT]
> This step is required only for external datasets. Data from the _Phishpedia_ dataset do not require modification.

The `organize_by_sample.py` script creates a directory structure where each sample (image + metadata) is placed in a
separate folder within `phishing` or `trusted_list` directories. This format is required for Phishpedia model training
and evaluation.

#### CSV File Requirements

Before using `organize_by_sample.py`, you need to prepare a CSV file containing your dataset metadata. The CSV file must
include the following columns:

- **`url`** (required): Full URL of the website
- **`fqdn`** (required): Fully Qualified Domain Name extracted from the URL
- **`screenshot_object`** (required): Filename or path to the screenshot file
- **`affected_entity`** (required): Target brand/entity name (e.g., "apple", "paypal")
- **`screenshot_hash`** (optional): Hash of the screenshot for deduplication
- **`is_phishing`** (optional): Boolean (1/0 or True/False) indicating if the sample is phishing. Defaults to 1 (
  phishing) if not provided

**Example CSV format:**

```csv
url,fqdn,screenshot_object,affected_entity,screenshot_hash,is_phishing
https://fake-apple.com/login,fake-apple.com,001.png,apple,abc123,1
https://legitimate-apple.com,apple.com,002.png,apple,def456,0
```

**Usage:**

Run the script from the main project directory:

```bash
uv run src/organize_by_sample.py --csv PATH_TO_CSV_FILE --screenshots PATH_TO_SCREENSHOTS_FOLDER --output PATH_TO_OUTPUT_DIRECTORY
```

**Arguments:**

- `--csv`: Path to the CSV file containing your dataset metadata
- `--screenshots`: Path to the directory containing screenshot files
- `--output`: Path where the organized dataset structure will be created

**Example:**

```bash
uv run src/organize_by_sample.py \
  --csv datasets/my_dataset/metadata.csv \
  --screenshots datasets/my_dataset/images/ \
  --output ./organized_phishpedia_dataset
```

**Generated Data Structure:**

The script creates a directory structure where each sample is placed in a folder named by combining the target and
sample ID:

```
OUTPUT_DIRECTORY/
├── phishing/
│   ├── apple+sample1/
│   │   ├── info.txt      # Contains the full URL
│   │   └── shot.png      # Screenshot
│   ├── apple+sample2/
│   │   ├── info.txt
│   │   └── shot.png
│   └── paypal+sample3/
│       ├── info.txt
│       └── shot.png
└── trusted_list/
    ├── apple+sample4/
    │   ├── info.txt
    │   └── shot.png
    └── google+sample5/
        ├── info.txt
        └── shot.png
```

Each `info.txt` file contains only the URL from the CSV. The script automatically determines whether samples are
phishing or trusted based on the `is_phishing` column in the CSV.

## How to prepare data for VisualPhish model?

### Data Preparation for VisualPhish (`organize_by_target.py`)

The `organize_by_target.py` script organizes screenshots into target-based directories, which is the format required by
the VisualPhish model. Images are grouped by brand/target rather than by individual samples.

**CSV File Requirements:**

The CSV file must contain the same columns as described for Phishpedia:

- **`url`** (required): Full URL of the website
- **`fqdn`** (required): Fully Qualified Domain Name
- **`screenshot_object`** (required): Filename or path to the screenshot file
- **`affected_entity`** (required): Target brand/entity name
- **`screenshot_hash`** (optional): Hash of the screenshot
- **`is_phishing`** (optional): Boolean indicating if the sample is phishing (1) or trusted (0)

**Usage:**

Run the script from the main project directory:

```bash
uv run src/organize_by_target.py --csv PATH_TO_CSV_FILE --screenshots PATH_TO_SCREENSHOTS_FOLDER --output PATH_TO_OUTPUT_DIRECTORY
```

**Arguments:**

- `--csv`: Path to the CSV file containing your dataset metadata
- `--screenshots`: Path to the directory containing screenshot files
- `--output`: Path where the organized dataset structure will be created

**Example:**

```bash
uv run src/organize_by_target.py \
  --csv datasets/my_dataset/metadata.csv \
  --screenshots datasets/my_dataset/images/ \
  --output ./organized_visualphish_dataset
```

**Script Behavior:**

- **Target naming**: Target directory names are created from the `affected_entity` column, converted to lowercase. Empty
  values become "unknown".
- **File naming**: Images are renamed using a pattern `T{target_id}_{image_count}.png` where `target_id` is the target's
  index and `image_count` is incremented for each image of that target.
- **Automatic separation**: Samples are automatically separated into `phishing` and `trusted_list` directories based on
  the `is_phishing` column value (defaults to phishing if column is missing).
- **Target lists**: The script generates `targets2.txt` (phishing targets) and `targets.txt` (trusted targets) files
  containing alphabetically sorted target names.

**Generated Data Structure:**

The dataset is organized with images grouped by target brand:

```
OUTPUT_DIRECTORY/
├── phishing/
│   ├── apple/
│   │   ├── T0_0.png
│   │   ├── T0_1.png
│   │   └── T0_2.png
│   ├── paypal/
│   │   ├── T1_0.png
│   │   └── T1_1.png
│   └── targets2.txt      # List of phishing targets: apple, paypal, etc.
└── trusted_list/
    ├── apple/
    │   ├── T0_0.png
    │   └── T0_1.png
    ├── google/
    │   └── T1_0.png
    └── targets.txt       # List of trusted targets: apple, google, etc.
```

This structure allows the VisualPhish model to learn target-specific features by having all images of the same brand
grouped together.
