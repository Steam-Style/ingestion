FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    HF_HOME=/data/hf

WORKDIR /app

RUN apt-get update \
    ; apt-get install -y --no-install-recommends git \
    ; rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY packages ./packages

RUN uv sync --frozen --no-cache

CMD ["uv", "run", "python", "src/main.py"]
