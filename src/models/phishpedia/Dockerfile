FROM python:3.9.21-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.5.15 /uv /uvx /bin/

WORKDIR /code

COPY uv.lock /code/uv.lock
COPY pyproject.toml /code/pyproject.toml
COPY libs/*.whl /code/libs/
RUN uv sync --frozen --no-cache --no-dev
RUN uv pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN uv pip install setuptools wheel
RUN uv pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
COPY * /code

CMD ["uv", "run", "--no-dev", "PhishpediaServing.py"]
