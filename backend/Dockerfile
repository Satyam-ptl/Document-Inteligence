# Phase 3's PaddleOCR-VL-1.6 + opencv-python-headless need a few system
# shared libs that python:3.12-slim doesn't ship (libgomp1 for OpenMP,
# used by paddlepaddle's CPU inference; libglib2.0-0/libsm6/libxext6 for
# opencv-python-headless's image codecs, which still dynamically link
# against them despite "headless" only meaning "no GUI/X11 bindings").
# Without these, `import cv2` / `import paddle` fails at container startup
# with an ImportError, not at `pip install` time — this was caught by
# static review in Phase 11 (no Docker daemon was available to actually
# build/run this image in any sandbox session so far; see PROJECT_STATE.md's
# Phase 11 verification log for what could vs couldn't be confirmed here).
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN mkdir -p /app/data/uploads /app/data/processed
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
