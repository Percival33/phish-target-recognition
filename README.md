# Phishing Target Recognition

<!-- TOC -->
* [Phishing Target Recognition](#phishing-target-recognition)
  * [Prerequisites](#prerequisites)
  * [What do you want to do?](#what-do-you-want-to-do)
    * [Complete Model Evaluation](#complete-model-evaluation)
    * [Test Single Model](#test-single-model)
    * [Web Interface Demo](#web-interface-demo)
    * [Use own dataset for evaluation](#use-own-dataset-for-evaluation)
      * [Update configuration](#update-configuration)
      * [Organize for `Phishpedia`](#organize-for-phishpedia)
        * [Step 1: Update target mapping](#step-1-update-target-mapping)
        * [Step 2: Run models](#step-2-run-models)
      * [Organize for `VisualPhish`](#organize-for-visualphish)
        * [Step 1: Setup environment](#step-1-setup-environment)
        * [Step 2: Run models](#step-2-run-models-1)
        * [Step 3: Run evaluation](#step-3-run-evaluation)
      * [Organize for `Baseline`](#organize-for-baseline)
        * [Step 1: Setup cross validation](#step-1-setup-cross-validation)
        * [Step 2: Run models](#step-2-run-models-2)
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

#### Update configuration

Edit `config.json` to configure your dataset. Here are the key configuration options:

**Dataset Structure Configuration (`dataset_image_paths`):**

Choose your `label_strategy` based on how your dataset is organized:

**Option 1: `"subfolders"` strategy** - Use when images are organized in separate phishing/benign folders:
```
your_dataset/
├── phishing_images/     # folder containing phishing screenshots
│   ├── image1.png
│   └── image2.png
└── benign_images/       # folder containing benign screenshots
    ├── image3.png
    └── image4.png
```

**Option 2: `"labels_file"` strategy** - Use when all images are in one folder with a labels.txt file:
```
your_dataset/
├── all_images/          # folder containing all screenshots
│   ├── 000.png
│   ├── 001.png
│   └── 002.png
└── labels.txt           # file with one label per line (matching image order)
```

**Configuration Examples:**

For **subfolder-organized dataset** (like Phishpedia format):
```json
{
  "cross_validation_config": {
    "dataset_image_paths": {
      "my_phishpedia_dataset": {
        "path": "data/raw/my_dataset",
        "label_strategy": "subfolders",
        "target_mapping": {
          "phishing": "phish_sample_30k",
          "benign": "benign_sample_30k"
        }
      }
    }
  }
}
```

For **labels-file dataset** (like VisualPhish format):
```json
{
  "cross_validation_config": {
    "enabled": true,
    "n_splits": 3,
    "shuffle": true,
    "random_state": 42,
    "output_splits_directory": "data_splits",
    "dataset_image_paths": {
      "my_visualphish_dataset": {
        "path": "data/raw/my_dataset",
        "label_strategy": "labels_file",
        "target_mapping": {
          "phishing": "newly_crawled_phishing",
          "benign": "benign_test"
        }
      }
    }
  }
}
```

**Result Files Configuration (`data_input_config`):**
```json
{
  "data_input_config": {
    "csv_column_prefixes": {
      "Phishpedia": "pp",
      "VisualPhish": "vp",
      "Baseline": "baseline"
    }
  }
}
```

**Configuration Parameters:**
- `path`: Relative to `PROJECT_ROOT_DIR`, points to your dataset folder
- `label_strategy`: `"subfolders"` or `"labels_file"`
- `target_mapping`: Maps `"phishing"`/`"benign"` to actual folder names in your dataset
- `n_splits`: Number of cross-validation folds (default: 3)
- `output_splits_directory`: Where to save split data (default: "data_splits")

#### Organize for `Phishpedia`
Warning: `screenshots` path is folder which with screenshot_object will form a valid path to image.

```shell
uv run src/organize_by_sample.py \
    --csv path_to_prepared_data.csv \
    --screenshots my_dataset/images/ \
    --output my_dataset/phishpedia_format
```

##### Step 1: Update target mapping

Run `just prepare-mapping` to update target mapping. This will update `src/models/phishpedia/models/domain_map.pkl` file.
If you have any other targets to add, you need to:can add them to the `domain_map.pkl` file.
- add their names to `domain_map.pkl` as new entries (target_name, target_domain) example:
```python
domain_map['amazon'] = ['amazon.com']
```
- add logos into `src/models/phishpedia/models/expand_targetlist`.

##### Step 2: Run models

Run model on every split.
For `Phishpedia` see [preparation steps](./docs/docs-to-process.md#phishpedia) which includes preparation of domain mappings.
If you have specific domains and examples update them before running models.

**Run on each cross-validation split**
```bash
mkdir -p $PROJECT_ROOT_DIR/logs/phishpedia # create logs directory

for split in 0 1 2; do
    echo "=== Running Phishpedia on split_${split} ==="
    uv run phishpedia.py \
        --folder $PROJECT_ROOT_DIR/data_splits/split_${split}/phishpedia/images/val \
        --output_txt $PROJECT_ROOT_DIR/logs/phishpedia/split_${split}_results.txt \
        --log
done
```

#### Organize for `VisualPhish`
Warning: `screenshots` path is folder which with screenshot_object will form a valid path to image.

For CSV preparation, see the [CSV file requirements](#use-own-dataset-for-evaluation) section above.

```shell
uv run src/organize_by_target.py \
    --csv path_to_prepared_data.csv \
    --screenshots my_dataset/images/ \
    --output my_dataset/visualphish_format
```

##### Step 1: Setup environment
To do this step you need to have `PROJECT_ROOT_DIR` set and dataset registered in `config.json`.

Synchronize dependencies and login to Wandb:
```bash
uv sync --frozen
uv run wandb login YOUR_API_KEY
```

##### Step 2: Run models

**Run on each cross-validation split**
```bash
cd $PROJECT_ROOT_DIR/src/models/visualphishnet/
mkdir -p $PROJECT_ROOT_DIR/logs/visualphish # create logs directory


for split in 0 1 2; do
    echo "=== Running VisualPhish on split_${split} ==="
    uv run trainer.py \
        --dataset-path $PROJECT_ROOT_DIR/data_splits/split_${split}/{dataset} \
        --logdir $PROJECT_ROOT_DIR/logs/visualphish/split_${split} \
        --output-dir $PROJECT_ROOT_DIR/data/processed/VisualPhish/split_${split}

done
```

##### Step 3: Run evaluation

**Evaluate on each cross-validation split**
```bash
cd $PROJECT_ROOT_DIR/src/models/visualphishnet/
mkdir -p $PROJECT_ROOT_DIR/logs/visualphish # create logs directory


for split in 0 1 2; do
    echo "=== Evaluating VisualPhish on split_${split} ==="
    uv run eval_new.py \
        --emb-dir $PROJECT_ROOT_DIR/data/processed/VisualPhish/split_${split} \
        --data-dir $PROJECT_ROOT_DIR/data_splits/split_${split}/{dataset} \
        --threshold 8.0 \
        --result-path $PROJECT_ROOT_DIR/logs/visualphish/split_${split}_eval \
        --save-folder $PROJECT_ROOT_DIR/logs/visualphish/split_${split}_results
done
```

#### Organize for `Baseline`
Warning: `screenshots` path is folder which with screenshot_object will form a valid path to image.

```shell
uv run src/organize_by_target.py \
    --csv path_to_prepared_data.csv \
    --screenshots my_dataset/images/ \
    --output my_dataset/baseline_format
```

##### Step 1: Setup cross validation
To do this step you need to have `PROJECT_ROOT_DIR` set and dataset registered in `config.json`.

Go to `src/cross_validation` and run `just setup` and then `just splits-links`.

##### Step 2: Run models
First run `load.py` to create FAISS index.
Phishing and benign samples must be run in separate steps. Below samples are separated in different folders.
Later query it, using `query.py`.

**Run on each cross-validation split**
```bash
mkdir -p $PROJECT_ROOT_DIR/logs/baseline # create logs directory

cd $PROJECT_ROOT_DIR/src/models/baseline/

for split in 0 1 2; do
    echo "=== Running Baseline on split_${split} ==="
    uv run load.py \
    --images $PROJECT_ROOT_DIR/data_splits/split_${split}/{dataset}/images/train/phish \
    --index $PROJECT_ROOT_DIR/data/processed/baseline/index_${split}.faiss \
    --is-phish \
    --batch-size 256 \
    --log

    uv run load.py \
    --images $PROJECT_ROOT_DIR/data_splits/split_${split}/{dataset}/images/train/benign \
    --index $PROJECT_ROOT_DIR/data/processed/baseline/index_${split}.faiss \
    --batch-size 256 \
    --log \
    --append

    uv run query.py \
    --images $PROJECT_ROOT_DIR/data_splits/split_${split}/{dataset}/images/val \
    --index $PROJECT_ROOT_DIR/data/processed/baseline/index_${split}.faiss \
    --output $PROJECT_ROOT_DIR/logs/baseline/results_${split}.csv \
    --unknown \
    --threshold 0.5 \
    --batch-size 256 \
    --log
done
```
