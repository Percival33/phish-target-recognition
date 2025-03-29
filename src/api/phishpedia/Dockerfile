FROM python:3.9.21-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.5.15 /uv /uvx /bin/

WORKDIR /code

COPY uv.lock /code/uv.lock
COPY pyproject.toml /code/pyproject.toml

RUN uv pip install --system --no-cache-dir -r pyproject.toml
RUN uv pip install --system --no-cache-dir torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN uv pip install --system --no-cache-dir --prerelease=allow detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html
RUN mkdir -p /code/app


COPY ModelServing.py /code
COPY __init__.py /code
COPY phishpedia/PhishpediaServing.py /code/app

CMD ["python", "-m", "app.PhishpediaServing"]