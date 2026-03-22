FROM python:3.10-slim

# ── system deps ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ libgomp1 curl wget git \
    && rm -rf /var/lib/apt/lists/*

# ── python deps ──────────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── project source ───────────────────────────────────────────────────────
COPY src/ ./src/

# ── cache directory (mount persistent volume here) ───────────────────────
RUN mkdir -p /cache/data /cache/models /cache/results /cache/logs
ENV CACHE_DIR=/cache

# ── data paths (override with -e when running) ───────────────────────────
ENV RAW_CAN_PATH=/data/raw/car_hacking.csv
ENV RAW_CIC_PATH=/data/raw/cicids2017.csv

# ── default: run full pipeline ───────────────────────────────────────────
CMD ["python", "src/pipeline.py"]
