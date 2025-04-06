prepare-visualphish:
    cd src/models/visualphish && \
    uv sync --frozen && \
    uv run --with gdown gdown 1ewejN6qo3Bkb8IYSKeklU4GIlRHqPlUC -O - --quiet | tar zxvf - -C ../../data/raw/VisualPhish

build-common:
    cd src/tools && \
    uv build && \
    cp dist/*.whl ../models/common

copy-vp: build-common
    cp src/tools/dist/*.whl src/models/visualphishnet/common

copy-pp: build-common
    cp src/tools/dist/*.whl src/models/phishpedia/common

tools: copy-vp copy-pp
    echo "Tools package was built"

docker: tools
    docker compose up