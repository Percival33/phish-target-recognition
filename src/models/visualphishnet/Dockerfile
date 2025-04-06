FROM python:3.9.21-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.5.15 /uv /uvx /bin/

WORKDIR /code

COPY uv.lock /code/uv.lock
COPY pyproject.toml /code/pyproject.toml
COPY common/*.whl /code/common/
RUN uv sync --frozen --no-cache
COPY * /code

CMD ["uv", "run", "VisualPhishServing.py"]