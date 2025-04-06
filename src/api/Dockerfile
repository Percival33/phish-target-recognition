FROM python:3.9.21-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.5.15 /uv /uvx /bin/

WORKDIR /code

COPY uv.lock /code/uv.lock
COPY pyproject.toml /code/pyproject.toml

RUN uv sync --frozen
RUN mkdir -p /code/app

COPY main.py /code
COPY routes.py /code
CMD ["uv", "run", "--no-dev", "main.py"]
