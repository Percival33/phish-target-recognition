FROM python:3.9.21-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.5.15 /uv /uvx /bin/

WORKDIR /code

COPY uv.lock /code/uv.lock
COPY pyproject.toml /code/pyproject.toml
COPY libs/*.whl /code/libs/
RUN uv sync --frozen --no-cache --no-dev
RUN uv add "perception @ git+https://github.com/thorn-oss/perception" --frozen

COPY * /code

CMD ["uv", "run", "--with", "perception", "--no-dev", "BaselineServing.py"]
