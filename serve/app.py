"""FastAPI service for PPB (plasma protein binding) prediction from SMILES.

Serves the lean XGB hybrid bundle plus a small static frontend. All inference is
delegated to the existing library (`ppb_model.predict`) - this module only wraps
it in an HTTP API.

Run locally (from this folder):
    PYTHONPATH=../src python -m uvicorn app:app --port 7860
In the container `src/` is on PYTHONPATH and PPB_BUNDLE points at the bundle.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Make `ppb_model` importable without requiring PYTHONPATH to be set by hand:
# the library lives in ../src relative to this file (the container sets
# PYTHONPATH too, so this is a no-op there).
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rdkit import Chem  # noqa: E402
from rdkit.Chem import Descriptors, rdMolDescriptors  # noqa: E402
from rdkit.Chem.Draw import rdMolDraw2D  # noqa: E402

from ppb_model.predict import load_bundle, predict_smiles  # noqa: E402

# Canonical model provenance/metrics, shared with the static docs/model_info.json
# that the Pages frontend fetches (see scripts/export_model_info.py).
from model_info import MODEL_INFO  # noqa: E402

MAX_BATCH = 200
HERE = Path(__file__).resolve().parent

# Bundle location is env-configurable so the same image works locally and in the
# container; default is the lean single hybrid model. The parity plot and MODEL_INFO
# metrics are generated from this same bundle so predictions and reported
# performance stay consistent (see scripts/export_parity.py).
DEFAULT_BUNDLE = HERE.parent / "models" / "final_xgb_hybrid.joblib"
BUNDLE_PATH = Path(os.environ.get("PPB_BUNDLE", str(DEFAULT_BUNDLE)))

app = FastAPI(title="PPB Prediction API", version="1.0.0")

# The web UI is served separately from GitHub Pages, so requests are cross-origin.
# Restrict to the Pages origin (plus localhost for dev); override in the Container
# App via PPB_ALLOWED_ORIGINS (comma-separated) for a custom domain.
_ALLOWED_ORIGINS = [o.strip() for o in os.environ.get(
    "PPB_ALLOWED_ORIGINS",
    "https://ddyan7.github.io,http://localhost:8000,http://127.0.0.1:8000",
).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
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

# Physicochemical + structure fields added per molecule; kept null when the SMILES
# could not be parsed so the JSON shape is uniform across rows.
_EMPTY_EXTRAS = {"svg": None, "mw": None, "logp": None,
                 "tpsa": None, "hbd": None, "hba": None}


def _mol_extras(std_smiles: str) -> dict[str, object]:
    """2D depiction (SVG) + key physicochemical descriptors for one molecule.

    LogP is highlighted in the UI because it is the dominant PPB driver in this
    model. Returns _EMPTY_EXTRAS if RDKit cannot parse the (already standardised)
    SMILES, which should not normally happen for an "ok" row.
    """
    mol = Chem.MolFromSmiles(std_smiles)
    if mol is None:
        return dict(_EMPTY_EXTRAS)
    drawer = rdMolDraw2D.MolDraw2DSVG(180, 120)
    drawer.drawOptions().clearBackground = False  # blend with page background
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return {
        "svg": drawer.GetDrawingText(),
        "mw": round(Descriptors.MolWt(mol), 1),
        "logp": round(Descriptors.MolLogP(mol), 2),
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 1),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
    }


@app.on_event("startup")
def _load() -> None:
    """Load the model bundle once at startup so requests are cheap."""
    global _bundle
    _bundle = load_bundle(BUNDLE_PATH)


class PredictRequest(BaseModel):
    smiles: list[str] = Field(..., description="SMILES strings to score.")


class PredictionRecord(BaseModel):
    """One compound's result. Field descriptions surface in the /docs schema."""

    input_smiles: str = Field(..., description="The SMILES you submitted.")
    predicted_ppb: float | None = Field(
        None, description="Predicted percent bound to plasma proteins (0-100). "
                          "Null if the SMILES could not be parsed.")
    predicted_fraction_unbound: float | None = Field(
        None, description="Free, pharmacologically active fraction = (100 - PPB)/100.")
    max_training_similarity: float | None = Field(
        None, description="Max Tanimoto similarity (0-1) of the molecule's Morgan "
                          "fingerprint to any training compound.")
    in_applicability_domain: bool = Field(
        ..., description="True if max_training_similarity clears the applicability-"
                         "domain threshold; predictions outside the domain are unreliable.")
    status: str = Field(..., description="Standardisation result: 'ok' or an error "
                                         "code such as 'rdkit_parse_failed'.")
    svg: str | None = Field(None, description="2D structure depiction as inline SVG.")
    mw: float | None = Field(None, description="Molecular weight (g/mol).")
    logp: float | None = Field(
        None, description="Calculated lipophilicity (Crippen MolLogP); the dominant "
                          "PPB driver in this model.")
    tpsa: float | None = Field(None, description="Topological polar surface area (Angstrom^2).")
    hbd: int | None = Field(None, description="Number of hydrogen-bond donors.")
    hba: int | None = Field(None, description="Number of hydrogen-bond acceptors.")


class PredictResponse(BaseModel):
    count: int = Field(..., description="Number of predictions returned.")
    predictions: list[PredictionRecord]


@app.get("/api/health")
def health() -> dict[str, object]:
    """Liveness/readiness check for the platform and the frontend."""
    return {"status": "ok", "model_loaded": _bundle is not None}


@app.get("/api/model-info")
def model_info() -> dict[str, object]:
    """Model provenance, performance, and citation for the UI's details panel."""
    return MODEL_INFO


@app.post("/api/predict", response_model=PredictResponse)
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
    # Attach a 2D depiction + physicochemical props for each parsed molecule.
    for rec, std, status in zip(records, df["standardised_smiles"], df["status"]):
        rec["in_applicability_domain"] = bool(rec["in_applicability_domain"])  # numpy -> py bool
        rec.update(_mol_extras(std) if status == "ok" and std else dict(_EMPTY_EXTRAS))
    return {"count": len(records), "predictions": records}


# This service is API-only; the web UI lives on GitHub Pages. Give the root a small
# JSON pointer so hitting it isn't a bare 404.
@app.get("/")
def root() -> dict[str, str]:
    return {"service": "PPB Prediction API", "docs": "/docs", "health": "/api/health"}
