# Phishing Target Recognition

## Prerequisites

Before starting work on the project, make sure you have the following tools installed:

- **Just**: A task management system. Installation instructions can be found [here](https://github.com/casey/just?tab=readme-ov-file#packages).
- **uv**: An advanced Python package and environment manager. Install using the following commands:
  ```bash
  curl -LsSf https://astral.sh/uv/0.5.18/install.sh | sh
  source $HOME/.local/bin/env
  ```
- **unzip**: A tool for decompressing ZIP files.

## Project Configuration

In the main project folder, execute the following commands:

1.  Install development tools:
    ```bash
    just tools
    ```
2.  Set the `PROJECT_ROOT_DIR` environment variable to point to the main project directory:
    ```bash
    export PROJECT_ROOT_DIR=$(pwd)
    ```
    You can also add this command to your shell configuration file (e.g., `~/.zshrc` or `~/.bashrc`) to make it available in every new terminal session:
    ```bash
    echo "export PROJECT_ROOT_DIR=$(pwd)" >> ~/.zshrc # For Zsh
    # or
    echo "export PROJECT_ROOT_DIR=$(pwd)" >> ~/.bashrc # For Bash
    source ~/.zshrc # or source ~/.bashrc
    ```

## Phishpedia

Instructions for running and preparing data for the Phishpedia model.

**Location:** Execute all commands for Phishpedia in the `src/models/phishpedia/` folder.

### Execution Steps:

1.  **Setup and Target List Extraction:**
    ```bash
    just setup
    just extract-targetlist
    ```
2.  **Data Preparation:** Before running the model, update domain mappings and prepare datasets:
    ```bash
    just prepare
    ```
3.  **Login to Wandb:**
    ```bash
    uv run wandb login YOUR_API_KEY
    ```
    Replace `YOUR_API_KEY` with your Weights & Biases API key.
4.  **Run Phishpedia Model:**
    ```bash
    uv run phishpedia.py --folder PATH_TO_DATA --log
    ```
    Replace `PATH_TO_DATA` with the path to the folder containing the prepared dataset.

### Data Preparation for Phishpedia (`organize_by_sample.py`):

To prepare data in the required format, use the `organize_by_sample.py` script. **Note:** This script should be run from the main project directory.

```bash
uv run src/organize_by_sample.py --csv PATH_TO_CSV_FILE --screenshots PATH_TO_SCREENSHOTS_FOLDER --output PATH_TO_OUTPUT_DIRECTORY
```

-   `--csv`: Path to the CSV file with data.
-   `--screenshots`: Path to the parent folder containing screenshots.
-   `--output`: Path to the directory where the processed data will be saved.

**Data Structure for Phishpedia:**

Each sample (image + information) is placed in a separate folder (`sample1`, `sample2`, etc.) inside the `phishing` or `trusted_list` directories.

```
PATH_TO_DATA/
├── phishing/
│   ├── sample1/
│   │   ├── info.txt      # Contains the full URL and target information
│   │   └── shot.png      # Screenshot
│   ├── sample2/
│   │   ├── info.txt
│   │   └── shot.png
│   └── targets2.txt    # List of targets for the phishing dataset
└── trusted_list/
    ├── sample1/
    │   ├── info.txt
    │   └── shot.png
    ├── sample2/
    │   ├── info.txt
    │   └── shot.png
    └── targets.txt     # List of targets for the trusted dataset
```

## VisualPhish

Instructions for running and preparing data for the VisualPhish model.

### Execution Steps:

1.  **Synchronize Dependencies:** In the main project folder, execute:
    ```bash
    uv sync --frozen
    ```
2.  **Download VisualPhish Data:** In the main project folder, execute (by default, data will be downloaded to `PROJECT_ROOT_DIR/data/interim`):
    ```bash
    # Make sure you have gdown installed: uv pip install gdown
    uv run --with gdown gdown 1ewejN6qo3Bkb8IYSKeklU4GIlRHqPlUC -O - --quiet | tar zxvf - -C "$PROJECT_ROOT_DIR/data/interim"
    ```
    If you want to download data to a different folder, change `"$PROJECT_ROOT_DIR/data/interim"` to your chosen path.
3.  **Login to Wandb:** In the main project folder, execute:
    ```bash
    uv run wandb login YOUR_API_KEY
    ```
    Replace `YOUR_API_KEY` with your Weights & Biases API key.
4.  **Run VisualPhish Training:** In the main project folder, execute:
    ```bash
    uv run trainer.py --dataset-path PATH_TO_DATASET --logdir LOG_DIRECTORY --output-dir OUTPUT_DIRECTORY
    ```
    Default values (if no arguments are provided):
    -   `--dataset-path`: `$PROJECT_ROOT_DIR/data/interim/VisualPhish`
    -   `--logdir`: `$PROJECT_ROOT_DIR/logdir`
    -   `--output-dir`: `$PROJECT_ROOT_DIR/data/processed/VisualPhish`

### Data Preparation for VisualPhish (`organize_by_target.py`):

To prepare data in the required format, use the `organize_by_target.py` script. **Note:** This script should be run from the main project directory.

```bash
uv run src/organize_by_target.py --csv PATH_TO_CSV_FILE --screenshots PATH_TO_SCREENSHOTS_FOLDER --output PATH_TO_OUTPUT_DIRECTORY
```

-   `--csv`: Path to the CSV file. The CSV file must contain the columns: `url`, `fqdn`, `screenshot_object`, `screenshot_hash`, `affected_entity`.
-   `--screenshots`: Path to the parent folder containing screenshots. Screenshot file names are taken from the `screenshot_object` column.
-   `--output`: Path to the directory where the processed data will be saved.

**Additional Information about the `organize_by_target.py` Script:**

-   The `is_phishing` column in the CSV file is optional. If it exists, a value of `False` means benign samples. If the column does not exist, all samples are treated as phishing.
-   Target directory names are created based on the `affected_entity` column:
    -   The value is converted to lowercase.
    -   If the value is empty (NaN), the name 'unknown' is used.
-   Targets are sorted alphabetically.
-   The script creates `targets2.txt` (in the `phishing` folder) and `targets.txt` (in the `trusted_list` folder) files, containing sorted target names.

**Data Structure for VisualPhish:**

The dataset is organized by targets.

```
PATH_TO_OUTPUT_DIRECTORY/
├── phishing/
│   ├── Target_A/
│   │   ├── Sample1.png
│   │   └── Sample2.png
│   ├── Target_B/
│   │   └── Sample3.png
│   └── targets2.txt      # List of targets for the phishing dataset
└── trusted_list/         # (Analogous structure if trusted data is present)
    ├── Target_X/
    │   ├── Sample4.png
    │   └── Sample5.png
    └── targets.txt       # List of targets for the trusted dataset
```

## Other Useful Commands

https://stackoverflow.com/questions/77250743/mac-xcode-g-cannot-compile-even-a-basic-c-program-issues-with-standard-libr

-   Synchronization with Overleaf repository (example):
    ```bash
    git fetch overleaf
    git checkout thesis-t
    git merge overleaf/master  # Or: git rebase overleaf/master
    git push overleaf thesis-t:master
    ```