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
2. Przed uruchomieniem modelu, wywołaj `just prepare` aby zaktualizować mapowanie domen i przygotować zbiory danych
3. Wywołaj `uv run wandb login <LOGIN>`
4. Uruchom poleceniem
```sh
uv run phishpedia.py --folder <FOLDER> --log
```
gdzie `FOLDER` jest ścieżką do zbioru danych.
5. Aby przygotować dane w wymaganym formacie, użyj skryptu organize_by_sample.py:
```sh
uv run src/organize_by_sample.py --csv path/to/data.csv --screenshots path/to/screenshots --output path/to/output_directory
```

Każda próbka jest umieszczana w osobnym folderze (sample1, sample2, itd.) bezpośrednio w folderach phishing/trusted_list. Każdy folder próbki zawiera plik info.txt z pełnym URL i informacją o celu (target) oraz shot.png ze zrzutem ekranu


Zbiór danych powinien być w postaci:
- Phishing
    * sample1
        - info.txt # zawierające pełny URL
        - shot.png # zrzut ekranu
    * sample2
        - info.txt
        - shot.png
    * sample3
        - info.txt
        - shot.png
    * targets2.txt
- trusted_list
    * sample1
        - info.txt
        - shot.png
    * sample2
        - info.txt
        - shot.png
    * sample3
        - info.txt
        - shot.png
    * targets.txt

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

5. Aby przygotować dane w wymaganym formacie, użyj skryptu organize_by_target.py:
```sh
uv run organize_by_target.py --csv path/to/data.csv --screenshots path/to/screenshots --output path/to/output_directory
```

Skrypt wymaga pliku CSV z kolumnami: `url`, `fqdn`, `screenshot_object`, `screenshot_hash`, `affected_entity`. Oraz folderu ze screenshotami.
Kolumna `is_phishing` jest opcjonalna - jeśli istnieje, wartość `False` oznacza próbki niezłośliwe (benign).
Jeśli kolumna nie istnieje, wszystkie próbki są traktowane jako phishing.

Nazwy katalogów z targetami są tworzone na podstawie kolumny `affected_entity`:
- Wartość jest konwertowana na małe litery
- Jeśli wartość jest pusta (NaN), używana jest nazwa 'unknown'
- Pliki zrzutów ekranu są pobierane z kolumny `screenshot_object`
- Targety są sortowane alfabetycznie
- Skrypt tworzy pliki `targets2.txt` w folderze phishing i `targets.txt` w folderze trusted_list, zawierające posortowane nazwy targetów

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