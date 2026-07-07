"""FastAPI service for PPB (plasma protein binding) prediction from SMILES.

Serves the lean XGB hybrid bundle plus a small static frontend. All inference is
delegated to the existing library (`ppb_model.predict`) — this module only wraps
it in an HTTP API.

Run locally (from this folder):
    PYTHONPATH=../src python -m uvicorn app:app --port 7860
In the container `src/` is on PYTHONPATH and PPB_BUNDLE points at the bundle.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ppb_model.predict import load_bundle, predict_smiles

MAX_BATCH = 200
HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"

# Bundle location is env-configurable so the same image works locally and on the
# Space; default resolves to the lean bundle relative to the project root.
DEFAULT_BUNDLE = HERE.parent / "models" / "final_xgb_hybrid.joblib"
BUNDLE_PATH = Path(os.environ.get("PPB_BUNDLE", str(DEFAULT_BUNDLE)))

app = FastAPI(title="PPB Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Columns returned to the client (drops standardised_smiles/uncertainty noise).
_OUT_COLS = [
    "input_smiles",
    "predicted_ppb",
    "predicted_fraction_unbound",
    "max_training_similarity",
    "in_applicability_domain",
    "status",
]

_bundle = None


@app.on_event("startup")
def _load() -> None:
    """Load the model bundle once at startup so requests are cheap."""
    global _bundle
    _bundle = load_bundle(BUNDLE_PATH)


class PredictRequest(BaseModel):
    smiles: list[str] = Field(..., description="SMILES strings to score.")


@app.get("/api/health")
def health() -> dict[str, object]:
    """Liveness/readiness check for the platform and the frontend."""
    return {"status": "ok", "model_loaded": _bundle is not None}


@app.post("/api/predict")
def predict(req: PredictRequest) -> dict[str, object]:
    """Score a batch of SMILES and return per-compound predictions.

    Always check `in_applicability_domain` before trusting a value; rows whose
    SMILES failed standardisation come back with a non-"ok" `status` and null
    predictions.
    """
    smiles = [s.strip() for s in req.smiles if s and s.strip()]
    if not smiles:
        raise HTTPException(status_code=422, detail="Provide at least one SMILES.")
    if len(smiles) > MAX_BATCH:
        raise HTTPException(
            status_code=422,
            detail=f"Too many SMILES ({len(smiles)}); limit is {MAX_BATCH} per request.",
        )

    df = predict_smiles(_bundle, smiles)
    # JSON can't carry NaN/inf; convert to null for a clean payload.
    records = df[_OUT_COLS].replace({np.nan: None}).to_dict(orient="records")
    return {"count": len(records), "predictions": records}


# Static frontend. Mounted last so it doesn't shadow the /api routes.
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(STATIC_DIR / "index.html"))
