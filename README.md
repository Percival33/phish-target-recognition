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
  * [Baseline](#baseline)
  * [Cross validation](#cross-validation)
  * [Evaluation](#evaluation)
  * [Datasets](#datasets)
  * [Website](#website)
    * [Prerequisites](#prerequisites-1)
    * [Running the Backend Services and Website](#running-the-backend-services-and-website)
  * [Other Useful Commands](#other-useful-commands)
<!-- TOC -->

## Prerequisites

Before starting, install these tools:

- **Just**: A command runner. Installation instructions can be found [here](https://github.com/casey/just?tab=readme-ov-file#packages).
- **uv**: An advanced Python package and environment manager. Install using the following commands:
  ```bash
  curl -LsSf https://astral.sh/uv/0.5.18/install.sh | sh
  source $HOME/.local/bin/env
  ```
- **unzip**: A tool for decompressing ZIP files.

**Initial Setup (Required for all paths):**
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

## What do you want to do?

### Complete Model Evaluation
TODO
### Test Single Model
TODO

### Web Interface Demo
TODO

### Use own dataset for evaluation
Prepare csv file with such columns:
```shell
    "url",
    "fqdn", # fully qualified domain
    "screenshot_object", # path to image
    "affected_entity",
    "is_phishing" # column with 1 and 0 depending on sample being phishing
```

then run `src/organize_by_sample.csv`
#### Organize for Phishpedia
```shell
uv run src/organize_by_sample.py \
    --csv my_dataset/prepared_data.csv \
    --screenshots my_dataset/images/ \ # this is folder which with screenshot_object will form a valid path to image
    --output my_dataset/phishpedia_format
```
#### Step 3: Update Configuration

Edit `config.json`:
```json
{
  "cross_validation_config": {
    "dataset_image_paths": {
      "my_dataset": {
        "path": "my_dataset/phishpedia_format",
        "label_strategy": "subfolders",
        "target_mapping": {
          "phishing": "phishing",
          "benign": "trusted_list"
        }
      }
    }
  }
}
```

path in dataset automaticaly prefixed with `PROJECT_ROOT_DIR`.

#### Step 4: Setup cross validation
To do this step you need to have `PROJECT_ROOT_DIR` set and dataset registered in `config.json`.

Go to `src/cross_validation` and run `just setup` and then `just splits-links`.

#### Step 5: Run models

Run model on every split.
For `Phishpedia` see [preparation steps](./docs/docs-to-process.md#phishpedia) which includes preparation of domain mappings.
If you have specific domains and examples update them before running models.

**Run on each cross-validation split**
```bash
for split in 0 1 2; do
    echo "=== Running Phishpedia on split_${split} ==="
    uv run phishpedia.py \
        --folder $PROJECT_ROOT_DIR/data_splits/split_${split}/Phishpedia/images/val \
        --output_txt $PROJECT_ROOT_DIR/logs/phishpedia/split_${split}_results.txt \
        --log
done
```

For `VisualPhish`
TODO

For `Baseline`
TODO
