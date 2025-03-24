FROM python:3.9.21-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.5.15 /uv /uvx /bin/

WORKDIR /code


COPY ./uv.lock /code/uv.lock
COPY ./pyproject.toml /code/pyproject.toml

RUN uv pip install --system --no-cache-dir -r pyproject.toml

COPY *.py /code/app

CMD ["uv", "run", "python", "-m", "app.main"]