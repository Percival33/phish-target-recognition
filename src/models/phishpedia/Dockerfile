FROM python:3.9.21-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.5.15 /uv /uvx /bin/

WORKDIR /code

COPY uv.lock /code/uv.lock
COPY pyproject.toml /code/pyproject.toml
COPY common/*.whl /code/common/
RUN uv sync --frozen --no-cache --no-dev
RUN uv pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 --index https://download.pytorch.org/whl/cpu
RUN uv pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
COPY * /code

CMD ["uv", "run", "--no-dev", "PhishpediaServing.py"]