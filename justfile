prepare-visualphish:
    cd src/models/visualphish && \
    uv sync --frozen && \
    uv run --with gdown gdown 1ewejN6qo3Bkb8IYSKeklU4GIlRHqPlUC -O - --quiet | tar zxvf - -C ../../data/raw/VisualPhish

build-common:
    cd src/tools && uv build

copy-vp: build-common
    mkdir -p src/models/visualphishnet/common
    cp src/tools/dist/*.whl src/models/visualphishnet/common

copy-pp: build-common
    mkdir -p src/models/phishpedia/common
    cp src/tools/dist/*.whl src/models/phishpedia/common

copy-baseline: build-common
    mkdir -p src/models/baseline/common
    cp src/tools/dist/*.whl src/models/baseline/common

tools: copy-vp copy-pp copy-baseline
    echo "Tools package was built"

run-pp recipe-name:
    just -f src/models/phishpedia/justfile {{ recipe-name }}

docker: tools
    just run-pp setup-models
    docker compose up
