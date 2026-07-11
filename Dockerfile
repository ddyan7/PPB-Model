# PPB prediction web service - image for Hugging Face Spaces (Docker SDK) / Azure
# Container Apps. Build context is the project root so we can copy src/ and the
# lean bundle. Multi-stage so the runtime image carries only the installed venv
# (no pip cache, no build-only apt) - smaller image = faster cold-start pulls.

# ---- builder: install inference-only deps into an isolated venv ----
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY serve/requirements-serve.txt /tmp/requirements-serve.txt
# Strip test/__pycache__ trees (RDKit and SciPy ship sizeable ones) to shrink the
# layer copied into the runtime stage.
RUN pip install -r /tmp/requirements-serve.txt \
    && find /opt/venv -type d -name tests -prune -exec rm -rf {} + \
    && find /opt/venv -type d -name '__pycache__' -prune -exec rm -rf {} + \
    && find /opt/venv -name '*.pyc' -delete

# ---- runtime: only runtime shared libs + the venv + app ----
FROM python:3.12-slim
# RDKit's drawing module links against X11/OpenMP shared libs not in the slim base.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxrender1 \
        libxext6 \
        libsm6 \
        libgomp1 \
        libexpat1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# App code + library + the served model bundle (lean single XGB hybrid).
WORKDIR /app
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
