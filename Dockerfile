# PPB prediction web service — image for Hugging Face Spaces (Docker SDK).
# Build context is the project root so we can copy src/ and the lean bundle.
FROM python:3.12-slim

# RDKit's drawing module links against X11/OpenMP shared libs not in the slim base.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxrender1 \
        libxext6 \
        libsm6 \
        libgomp1 \
        libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Install inference-only Python deps first for better layer caching.
WORKDIR /app
COPY serve/requirements-serve.txt /app/requirements-serve.txt
RUN pip install --no-cache-dir -r /app/requirements-serve.txt

# App code + library + the served model bundle (lean single XGB hybrid).
COPY serve/ /app/serve/
COPY src/ /app/src/
COPY models/final_xgb_hybrid.joblib /app/models/final_xgb_hybrid.joblib

# ppb_model importable without installing the full package (avoids optuna/matplotlib).
ENV PYTHONPATH=/app/src \
    PPB_BUNDLE=/app/models/final_xgb_hybrid.joblib \
    PORT=7860

# HF Spaces convention: run as non-root uid 1000.
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

WORKDIR /app/serve
EXPOSE 7860
# ${PORT:-7860}: honour a platform-injected PORT, fall back to the Spaces default.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
