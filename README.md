# Phish Target Recognition


## Instrukcja uruchomienia (Szczegółowa)

Wymagane:
- [Just](https://github.com/casey/just?tab=readme-ov-file#packages)
- uv
```
curl -LsSf https://astral.sh/uv/0.5.18/install.sh | sh
source $HOME/.local/bin/env
```
- unzip
- W głównym folderze należy wywołać `just tools` i `export PROJECT_ROOT_DIR=$(pwd)`

### Phishpedia:
1. Wywołaj `just setup` a następnie `just extract-targetlist`
2. Wywołaj `uv run wandb login <LOGIN>`
3. Uruchom poleceniem
```sh
uv run phishpedia.py --folder <FOLDER> --log
```
gdzie `FOLDER` jest ścieżką do zbioru danych.
4. Folder z logo i poprawnymi domenami 

Zbiór danych powinien być w postaci:


### VisualPhish:
1. Wywołaj `uv sync --frozen`
2. `uv run --with gdown gdown 1ewejN6qo3Bkb8IYSKeklU4GIlRHqPlUC -O - --quiet | tar zxvf - -C <FOLDER>`
domyślnie folder to `/data/interim`
3. Wywołaj `uv run wandb login <LOGIN>`
4. Uruchom poleceniem
```sh
uv run trainer.py --dataset-path <DATASET_PATH> --logdir <LOGDIR> --output-dir <OUTPUT_DIR>
```
gdzie domyślnie:
- `DATASET_PATH` to `PROJECT_ROOT_DIR/data/interim/VisualPhish`
- `LOGDIR` to `PROJECT_ROOT_DIR/logdir`
- `OUTPUT_DIR` to `PROJECT_ROOT_DIR/data/processed/VisualPhish`
- profile_batch ??

Zbiór danych powinien być w postaci:
- Phishing
    - Cel
        * Próbka1
        * Próbka2
    - Cel2
        * Próbka3
    - targets2.txt
- trusted_list
    - Cel1
        * Próbka1
        * Próbka2
    - Cel2
        * Próbka3
    - targets.txt

## Inne
- `sudo xcode-select --switch /Applications/Xcode.app` from [here](https://github.com/PX4/PX4-SITL_gazebo-classic/issues/1021)

https://stackoverflow.com/questions/77250743/mac-xcode-g-cannot-compile-even-a-basic-c-program-issues-with-standard-libr

git fetch overleaf
git checkout thesis-t
git merge overleaf/master  # Or use `git rebase overleaf/master`
git push overleaf thesis-t:master

echo "export PROJECT_ROOT_DIR=?" >> ~/.zshrc