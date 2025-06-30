# External datasets

<!-- TOC -->
* [External datasets](#external-datasets)
  * [How to prepare dataset to use Phishpedia model?](#how-to-prepare-dataset-to-use-phishpedia-model)
    * [Evaluation for Phishpedia](#evaluation-for-phishpedia)
      * [Preparing Evaluation Data with `organize.py`](#preparing-evaluation-data-with-organizepy)
    * [Data Preparation for Phishpedia (`organize_by_sample.py`):](#data-preparation-for-phishpedia-organize_by_samplepy)
      * [Preparing CSV Data for `organize_by_sample.py` with `prepare_data_for_organizer.py`](#preparing-csv-data-for-organize_by_samplepy-with-prepare_data_for_organizerpy)
  * [How to prepare data for VisualPhish model?](#how-to-prepare-data-for-visualphish-model)
    * [Data Preparation for VisualPhish (`organize_by_target.py`):](#data-preparation-for-visualphish-organize_by_targetpy)
<!-- TOC -->

## How to prepare dataset to use Phishpedia model?
TODO: check methods
### Evaluation for Phishpedia

To prepare data for cross-model or cross-dataset performance evaluation, particularly when using Phishpedia's organization scripts, you can use scripts to organize your datasets into a consistent format. For example, the `src/models/phishpedia/scripts/organize.py` script can be used to categorize images based on labels.

#### Preparing Evaluation Data with `organize.py`

The `organize.py` script takes a directory of images, a file containing labels for these images, and an output directory. It then creates subdirectories in the output path, named according to the labels (or a mapping defined in the script, like using a domain name for a brand), and copies the corresponding images into these subdirectories.

**Example Command:**

Suppose you have a dataset of phishing images located in `datasets/VisualPhish/newly_crawled_phishing/` and a corresponding `labels.txt` file in the same directory. To process this data and save the organized output to a directory named `evaluation_dataset_prepared`, you would run the following command from the project root:

```bash
uv run src/models/phishpedia/scripts/organize.py datasets/VisualPhish/newly_crawled_phishing/ datasets/VisualPhish/newly_crawled_phishing/labels.txt ./evaluation_dataset_prepared
```

**Explanation of Arguments:**

-   `src/models/phishpedia/scripts/organize.py`: The path to the organization script.
-   `datasets/VisualPhish/newly_crawled_phishing/`: The input directory containing the image files (e.g., `000.png`, `001.png`, ...).
-   `datasets/VisualPhish/newly_crawled_phishing/labels.txt`: The path to the text file where each line is a label corresponding to an image.
-   `./evaluation_dataset_prepared`: The output directory where the script will create folders for each label and copy the images into them.

This structured dataset can then be used for evaluating the performance of different models. Ensure your `labels.txt` file is correctly formatted, with each label on a new line, corresponding to the numerically ordered image files in your input directory.


### Data Preparation for Phishpedia (`organize_by_sample.py`):

> [!IMPORTANT]
> This step is required only for external datasets. Data from the _Phishpedia_ dataset do not require modification.

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

## How to prepare data for VisualPhish model?
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
