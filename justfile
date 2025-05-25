prepare-visualphish:
    cd src/models/visualphish && \
    uv sync --frozen && \
    uv run --with gdown gdown 1ewejN6qo3Bkb8IYSKeklU4GIlRHqPlUC -O - --quiet | tar zxvf - -C ../../data/raw/VisualPhish

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
    cp src/tools/dist/*.whl src/eval/libs/
    echo "Copied tools wheel to src/eval/libs/"

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

tools: copy-tools-to-visualphishnet copy-tools-to-phishpedia copy-tools-to-baseline
    echo "Tools package was built"

run-pp recipe-name:
    just -f src/models/phishpedia/justfile {{ recipe-name }}

docker: tools
    just run-pp setup-models
    docker compose up
