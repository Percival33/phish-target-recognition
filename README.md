# Phishing Target Recognition

## Prerequisites

Before starting work on the project, make sure you have the following tools installed:

- **Just**: A command runner. Installation instructions can be found [here](https://github.com/casey/just?tab=readme-ov-file#packages).
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

-   `--csv`: Path to the CSV file with data. The CSV file must contain the following columns:
    - `url`: Full URL of the website
    - `fqdn`: Fully Qualified Domain Name
    - `screenshot_object`: Path to the screenshot file (relative or absolute)
    - `screenshot_hash`: Hash of the screenshot
    - `affected_entity`: Target brand/entity name
    - `is_phishing` (optional): Boolean indicating if the sample is phishing (True) or legitimate (False)
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

### Evaluation for Phishpedia

To prepare data for cross-model or cross-dataset performance evaluation, particularly when using Phishpedia's organization scripts, you can use scripts to organize your datasets into a consistent format. For example, the `src/models/phishpedia/scripts/organize.py` script can be used to categorize images based on labels.

#### Preparing Evaluation Data with `organize.py`

The `organize.py` script takes a directory of images, a file containing labels for these images, and an output directory. It then creates subdirectories in the output path, named according to the labels (or a mapping defined in the script, like using a domain name for a brand), and copies the corresponding images into these subdirectories.

**Example Command:**

Suppose you have a dataset of newly crawled phishing images located in `datasets/VisualPhish/newly_crawled_phishing/` and a corresponding `labels.txt` file in the same directory. To process this data and save the organized output to a directory named `evaluation_dataset_prepared`, you would run the following command from the project root:

```bash
uv run src/models/phishpedia/scripts/organize.py datasets/VisualPhish/newly_crawled_phishing/ datasets/VisualPhish/newly_crawled_phishing/labels.txt ./evaluation_dataset_prepared
```

**Explanation of Arguments:**

-   `src/models/phishpedia/scripts/organize.py`: The path to the organization script.
-   `datasets/VisualPhish/newly_crawled_phishing/`: The input directory containing the image files (e.g., `000.png`, `001.png`, ...).
-   `datasets/VisualPhish/newly_crawled_phishing/labels.txt`: The path to the text file where each line is a label corresponding to an image.
-   `./evaluation_dataset_prepared`: The output directory where the script will create folders for each label and copy the images into them.

This structured dataset can then be used for evaluating the performance of different models. Ensure your `labels.txt` file is correctly formatted, with each label on a new line, corresponding to the numerically ordered image files in your input directory.

#### Preparing CSV Data for `organize_by_sample.py` with `prepare_data_for_organizer.py`

If you have a folder of images (e.g., `00000.png`, `00001.jpg`, ...) and a corresponding text file where each line is a URL for an image, you can use the `scripts/prepare_data_for_organizer.py` script to generate a CSV file. This CSV file will be in the format required by the `src/organize_by_sample.py` script, which is useful for preparing datasets for Phishpedia.

The `prepare_data_for_organizer.py` script assigns numerical filenames (e.g., `001.png`, `image_002.jpg`) to the URLs based on their order in the labels file (line 1 of labels corresponds to the first numerically sorted image, line 2 to the second, and so on).

**Example Command:**

Suppose you have images in `datasets/my_raw_data/images/` and a text file with URLs at `datasets/my_raw_data/urls.txt`. To generate a CSV file named `prepared_for_organizer.csv` in the `datasets/my_raw_data/` directory, you would run the following command from the project root:

```bash
uv run scripts/prepare_data_for_organizer.py \
  --image-folder datasets/my_raw_data/images/ \
  --labels-file datasets/my_raw_data/urls.txt \
  --output-csv datasets/my_raw_data/prepared_for_organizer.csv \
  --is-phishing
```

**Explanation of Arguments:**

-   `scripts/prepare_data_for_organizer.py`: The path to the CSV preparation script.
-   `--image-folder`: Path to the directory containing the image files. The script will attempt to sort images numerically based on their filenames (e.g., `000.png`, `001.png`, ...).
-   `--labels-file`: Path to the text file where each line contains a URL corresponding to an image, sorted in the same order as the images.
-   `--output-csv`: The path where the generated CSV file will be saved.
-   `--is-phishing` (optional): Marks all entries in the CSV as phishing. Use `--no-is-phishing` to mark them as legitimate. Defaults to phishing. The `screenshot_object` column in the CSV will always contain just the image filename (e.g., `001.png`); this means the `--image-folder` provided to this script should typically be the same as the `--screenshots` folder provided to `organize_by_sample.py`.

The generated CSV will contain the columns: `url`, `fqdn`, `screenshot_object`, `screenshot_hash`, `affected_entity` (set to null/empty), and `is_phishing`. This CSV can then be used with the `organize_by_sample.py` script:

```bash
uv run src/organize_by_sample.py \
  --csv datasets/my_raw_data/prepared_for_organizer.csv \
  --screenshots datasets/my_raw_data/images/ \
  --output ./evaluation_dataset_by_sample
```

This allows for a streamlined process from raw images and URLs to the structured format required by Phishpedia.

## VisualPhish

Instructions for running and preparing data for the VisualPhish model.

### Execution Steps:

1.  **Synchronize Dependencies:** In the main project folder, execute:
    ```bash
    uv sync --frozen
    ```
2.  **Download VisualPhish Data:** In the main project folder, execute (by default, data will be downloaded to `PROJECT_ROOT_DIR/data/interim`):
    ```bash
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

### Evaluation for VisualPhish

To evaluate the VisualPhish model using pre-computed embeddings and a specific threshold, you can run the `eval_new.py` script. This script will load the target list embeddings from a specified directory and the test data embeddings from a default or specified save folder.

**Note for macOS users:** If you encounter issues with TensorFlow, you might need to install it specifically for macOS. You can do this by running:
```bash
uv sync --extra macos
```
This command should be executed after the general `uv sync --frozen` if TensorFlow issues persist.

**Example Evaluation Command:**

The following command runs the evaluation script using target list embeddings from a specified directory. Replace `EMB_FOLDER` with the actual path to the directory containing the target list embedding files (e.g., `whitelist_emb.npy`, `whitelist_labels.npy`, `whitelist_file_names.npy`). The script also assumes that test embeddings and other necessary files (like `all_labels.npy`, `all_file_names.npy`, `pairwise_distances.npy`) are located in the directory specified by `--save-folder` (defaults to `logs/VisualPhish-Results/` if not overridden in the script or command line) or are generated if the script is modified to re-process data.

```bash
uv run src/models/visualphishnet/eval_new.py --emb-dir EMB_FOLDER --threshold 8.0
```

This will output metrics to the console and save detailed results to CSV and text files in the directory specified by `--result-path` (defaults to `logs/VisualPhish/`).

## Website

This project includes a Streamlit-based web interface, defined in `src/website.py`, for analyzing images. The website relies on a backend API and associated model services, which can be managed using Docker Compose as defined in `docker-compose.yml`.

### Prerequisites

1.  **Docker and Docker Compose**: Ensure Docker and Docker Compose are installed on your system.
2.  **Backend Services & Model Files**: The API backend (`api` service in `docker-compose.yml`) communicates with model-specific services (`visualphish` and `phishpedia`). These services require specific files to be present in your project directory at the following locations (relative to the project root):
    *   **For the `visualphish` service:**
        *   `data/processed/VisualPhish/model2.h5`
        *   `data/processed/VisualPhish/whitelist_emb.npy`
        *   `data/processed/VisualPhish/whitelist_file_names.npy`
        *   `data/processed/VisualPhish/whitelist_labels.npy`
    *   **For the `phishpedia` service:**
        *   `src/models/phishpedia/models/` (This directory and its contents)
        *   `src/models/phishpedia/LOGO_FEATS.npy`
        *   `src/models/phishpedia/LOGO_FILES.npy`
    Ensure these files and directories are correctly placed before attempting to start the services. These files are typically generated or downloaded during the model preparation steps outlined in the "Phishpedia" and "VisualPhish" sections of this README.

3.  **Website Dependencies**: All Python dependencies for the Streamlit website, including Streamlit itself, must be installed. If you have not done so already, synchronize your environment using:
    ```bash
    uv sync --frozen
    ```
    This command should install all necessary packages listed in your project's dependency file.

### Running the Backend Services and Website

1.  **Start Backend Services**:
    Navigate to the root directory of the project in your terminal and run the Docker Compose services in detached mode:
    ```bash
    docker-compose up -d
    ```
    This command will build (if not already built) and start the `api`, `visualphish`, and `phishpedia` services defined in `docker-compose.yml`. The API service will then be accessible at `http://localhost:8000` (as configured in `docker-compose.yml` and `src/website.py`).

2.  **Run the Streamlit Website**:
    Once the backend services are running (verify their status with `docker-compose ps`), start the Streamlit web application. In the project root directory, execute:
    ```bash
    uv run streamlit run src/website.py
    ```
    The website should then be accessible in your web browser, typically at `http://localhost:8501`. It will connect to the API service running via Docker Compose.

## Other Useful Commands

-   Synchronization with Overleaf repository (example):
    ```bash
    git fetch overleaf
    git checkout thesis-t
    git merge overleaf/master  # Or: git rebase overleaf/master
    git push overleaf thesis-t:master
    ```
