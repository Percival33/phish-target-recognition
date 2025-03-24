prepare-visualphish:
    cd src/models/visualphish && \
    uv sync --frozen && \
    uv run --with gdown gdown 1ewejN6qo3Bkb8IYSKeklU4GIlRHqPlUC -O - --quiet | tar zxvf - -C ../../data/raw/VisualPhish
