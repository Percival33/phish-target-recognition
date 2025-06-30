# Phishing Target Recognition

<!-- TOC -->
* [Phishing Target Recognition](#phishing-target-recognition)
  * [Prerequisites](#prerequisites)
  * [Project Configuration](#project-configuration)
  * [Phishpedia](#phishpedia)
    * [Execution Steps:](#execution-steps)
    * [External datasets](#external-datasets)
  * [VisualPhish](#visualphish)
    * [Execution Steps:](#execution-steps-1)
    * [Evaluation for VisualPhish](#evaluation-for-visualphish)
  * [Cross validation](#cross-validation)
  * [Datasets](#datasets)
  * [Website](#website)
    * [Prerequisites](#prerequisites-1)
    * [Running the Backend Services and Website](#running-the-backend-services-and-website)
  * [Other Useful Commands](#other-useful-commands)
<!-- TOC -->

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
    In case of failure make sure that dataset is in `$PROJECT_ROOT_DIR/data/raw/phishpedia/` folder
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

### External datasets
For external dataset preparation see [external dataset preparation guide](./docs/external-datasets.md)

## VisualPhish

Instructions for running and preparing data for the VisualPhish model.

### Execution Steps:

1.  **Synchronize Dependencies:** In the main project folder, execute:
    ```bash
    uv sync --frozen
    ```
    >![Note]
    > **Note for macOS users:** If you encounter issues with TensorFlow, you might need to install it specifically for macOS. You can do this by running:
    > ```bash
    > uv sync --extra macos
    > ```
    > This command should be executed after the general `uv sync --frozen` if TensorFlow issues persist.
2.  **Login to Wandb:** In the main project folder, execute:
    ```bash
    uv run wandb login YOUR_API_KEY
    ```
    Replace `YOUR_API_KEY` with your Weights & Biases API key.
3.  **Run VisualPhish Training:** In the main project folder, execute:
    ```bash
    uv run trainer.py --dataset-path PATH_TO_DATASET --logdir LOG_DIRECTORY --output-dir OUTPUT_DIRECTORY
    ```
    Default values (if no arguments are provided):
    -   `--dataset-path`: `$PROJECT_ROOT_DIR/data/interim/VisualPhish`
    -   `--logdir`: `$PROJECT_ROOT_DIR/logdir`
    -   `--output-dir`: `$PROJECT_ROOT_DIR/data/processed/VisualPhish`

### Evaluation for VisualPhish

To evaluate the VisualPhish model using pre-computed embeddings and a specific threshold, you can run the `eval_new.py` script. This script will load the target list embeddings from a specified directory and the test data embeddings from a default or specified save folder.

**Example Evaluation Command:**

The following command runs the evaluation script using target list embeddings from a specified directory. Replace `EMB_FOLDER` with the actual path to the directory containing the target list embedding files (e.g., `whitelist_emb.npy`, `whitelist_labels.npy`, `whitelist_file_names.npy`). The script also assumes that test embeddings and other necessary files (like `all_labels.npy`, `all_file_names.npy`, `pairwise_distances.npy`) are located in the directory specified by `--save-folder` (defaults to `logs/VisualPhish-Results/` if not overridden in the script or command line) or are generated if the script is modified to re-process data.

```bash
uv run src/models/visualphishnet/eval_new.py --emb-dir EMB_FOLDER --threshold 8.0 # default value from original paper
```

This will output metrics to the console and save detailed results to CSV and text files in the directory specified by `--result-path` (defaults to `logs/VisualPhish/`).

## Cross validation


## Datasets
This project uses two main datasets for phishing target recognition:
- **Phishpedia**: A dataset of phishing websites with associated screenshots and metadata.
```shell
uv run --with gdown gdown 12ypEMPRQ43zGRqHGut0Esq2z5en0DH4g -O download_phish.zip && mkdir -p $PROJECT_ROOT_DIR/data/raw/phishpedia/phish_sample_30k && unzip -q download_phish.zip -d $PROJECT_ROOT_DIR/data/raw/phishpedia/phish_sample_30k && rm download_phish.zip
uv run --with gdown gdown 1yORUeSrF5vGcgxYrsCoqXcpOUHt-iHq_ -O download_benign.zip && mkdir -p $PROJECT_ROOT_DIR/data/raw/phishpedia/benign_sample_30k && unzip -q download_benign.zip -d $PROJECT_ROOT_DIR/data/raw/phishpedia/benign_sample_30k && rm download_benign.zip
```
- **VisualPhish**: A dataset of phishing images with associated metadata, including a whitelist of legitimate brands.
```shell
uv run --with gdown gdown 1l-aQk54F0tAZ-RPfOyGo1jtz-Dsxo1Ao -O download_vp.zip && mkdir -p $PROJECT_ROOT_DIR/data/raw/visualphish && unzip -q download_vp.zip -d $PROJECT_ROOT_DIR/data/raw/visualphish && rm download_vp.zip
```


File available below is preprocessed dataset of already cropped images.
```bash
uv run --with gdown gdown 1ewejN6qo3Bkb8IYSKeklU4GIlRHqPlUC -O - --quiet | tar zxvf - -C "$PROJECT_ROOT_DIR/data/interim"
```

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
