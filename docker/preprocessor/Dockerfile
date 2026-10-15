# syntax=docker/dockerfile:1
# Build from the repository root:
# docker build -f docker/preprocessor/Dockerfile -t pumps-and-dumps-preprocessor:local .
FROM python:3.13-slim-bookworm AS preprocessor
LABEL org.opencontainers.image.title="pumps-and-dumps-preprocessor" \
      org.opencontainers.image.description="Convert Binance trade archives to Hive-partitioned Parquet" \
      org.opencontainers.image.licenses="MIT"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 pumps \
    && useradd --uid 1000 --gid pumps --create-home pumps \
    && mkdir /data \
    && chown pumps:pumps /data

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PUMPS_AND_DUMPS_DATA_DIR=/data \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache
WORKDIR /app
COPY docker/preprocessor/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN python -m pip install --no-cache-dir --no-deps --editable .
COPY resources/ ./resources/
USER pumps
ENTRYPOINT ["pumps-write-hive"]
CMD ["--help"]
