prepare-visualphish:
    cd $PROJECT_ROOT_DIR/src/models/visualphishnet && \
    uv sync --frozen

build-common:
    cd src/tools && uv build

copy-tools-to-visualphishnet: build-common
    mkdir -p src/models/visualphishnet/libs
    cp src/tools/dist/*.whl src/models/visualphishnet/libs

copy-tools-to-phishpedia: build-common
    mkdir -p src/models/phishpedia/libs
    cp src/tools/dist/*.whl src/models/phishpedia/libs

copy-tools-to-baseline: build-common
    mkdir -p src/models/baseline/libs
    cp src/tools/dist/*.whl src/models/baseline/libs

copy-tools-for-eval: build-common
    mkdir -p src/eval/libs
    cp src/tools/dist/*.whl src/eval/libs

copy-tools-to-cv: build-common
    mkdir -p src/cross_validation/libs
    cp src/tools/dist/*.whl src/cross_validation/libs

copy-tools-to-data-splitter: build-common
    mkdir -p src/data_splitter/libs
    cp src/tools/dist/*.whl src/data_splitter/libs/

setup-data-splitter: copy-tools-to-data-splitter
    cd src/data_splitter && \
    echo "Setting up 'data_splitter' package environment in src/data_splitter/ ..." && \
    uv sync --frozen --quiet && \
    echo "'data_splitter' environment setup complete."

run-data-split config_path='config.json': setup-data-splitter
    echo "Running data split with config: {{ config_path }}"
    cd src/data_splitter && \
    uv run split_data.py {{ config_path }}

setup-eval: copy-tools-for-eval
    cd src/eval && \
    echo "Setting up 'eval' package environment in src/eval/ ..." && \
    uv sync --quiet && \
    # TODO: all tools should be installed by uv
    uv pip install ./libs/*.whl --quiet
    echo "'eval' environment setup complete."

run-eval config_path='config.json': setup-eval
    echo "Running evaluation with config: {{ config_path }}"
    cd src/eval && \
    uv run eval-run --config {{ config_path }}

evaluate: run-eval

clean-eval:
    rm -rf src/eval/libs
    rm -rf src/eval/.venv
    rm -rf src/eval/critical_difference_analysis_results
    echo "Cleaned 'eval' libs, virtual environment, and results."

clean-data-splitter:
    rm -rf src/data_splitter/libs
    rm -rf src/data_splitter/.venv
    rm -rf src/data_splitter/data_splits
    echo "Cleaned 'data_splitter' libs, virtual environment, and results."

tools: copy-tools-to-visualphishnet copy-tools-to-phishpedia copy-tools-to-baseline copy-tools-to-cv copy-tools-to-data-splitter
    echo "Tools package was built"

run-pp recipe-name:
    just -f src/models/phishpedia/justfile {{ recipe-name }}

docker: tools
    just run-pp setup-models
    docker compose up
