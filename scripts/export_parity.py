"""Export held-out test predictions for the web app's parity plot.

Predictions are recomputed **from the served model bundle** (not from the
predictions CSV's `predicted_ppb` column), so the plotted performance always
matches exactly what the web app serves. Only the stable identity columns
(SMILES, observed value, split) are read from the frozen predictions table -
this makes the export robust to Google-Drive sync churn on the results CSV.

Writes docs/parity_test.json (tiny), which ships with the static web UI on
GitHub Pages alongside docs/index.html.

Run after (re)training or after changing the served bundle:
    python scripts/export_parity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ppb_model.predict import load_bundle, predict_smiles  # noqa: E402

SRC = ROOT / "reports" / "results" / "predictions_final.csv"
# Must match serve/app.py DEFAULT_BUNDLE (the model the app actually serves).
BUNDLE = ROOT / "models" / "final_xgb_hybrid.joblib"
DST = ROOT / "docs" / "parity_test.json"


def main() -> None:
    df = pd.read_csv(SRC)
    test = df[df["data_split"] == "test"].copy()
    y = test["observed_ppb"].to_numpy()

    # Recompute everything from the served bundle: predictions, similarity, AD.
    pred = predict_smiles(load_bundle(BUNDLE), test["canonical_smiles"].tolist())
    yhat = pred["predicted_ppb"].to_numpy()
    in_ad = pred["in_applicability_domain"].to_numpy().astype(bool)
    sim = pred["max_training_similarity"].to_numpy()

    ae = np.abs(y - yhat)

    def mae(mask: np.ndarray) -> float | None:
        return round(float(ae[mask].mean()), 2) if mask.any() else None

    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    # log10(fraction unbound): the field-standard PPB space (see the web plot).
    lfu_o = np.log10(np.clip((100 - y) / 100, 1e-4, 1))
    lfu_p = np.log10(np.clip((100 - yhat) / 100, 1e-4, 1))
    lfu_res = float(np.sum((lfu_o - lfu_p) ** 2))
    lfu_tot = float(np.sum((lfu_o - lfu_o.mean()) ** 2))

    # Each point: [observed, predicted, in_domain, max_training_similarity].
    points = [[round(float(o), 2), round(float(p), 2), bool(a), round(float(s), 2)]
              for o, p, a, s in zip(y, yhat, in_ad, sim)]

    out = {
        "note": f"Held-out PPBR_AZ scaffold test set - predictions recomputed from the "
                f"served bundle ({BUNDLE.name}). "
                "Point = [observed_ppb, predicted_ppb, in_applicability_domain, "
                "max_training_similarity].",
        "n": len(points),
        "metrics": {
            "MAE": round(float(ae.mean()), 2),
            "RMSE": round(float(np.sqrt(np.mean((y - yhat) ** 2))), 2),
            "R2": round(1 - ss_res / ss_tot, 3),
            "R2_logfu": round(1 - lfu_res / lfu_tot, 3),
            "RMSE_logfu": round(float(np.sqrt(np.mean((lfu_o - lfu_p) ** 2))), 2),
            "Spearman": round(float(spearmanr(y, yhat).statistic), 3),
            "high_binding_MAE": round(float(ae[y >= 90].mean()), 2),
            "MAE_in": mae(in_ad),
            "MAE_out": mae(~in_ad),
            "n_out": int((~in_ad).sum()),
        },
        "points": points,
    }
    DST.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    m = out["metrics"]
    print(f"Wrote {DST} from {BUNDLE.name}: "
          f"MAE={m['MAE']} RMSE={m['RMSE']} R2={m['R2']} Spearman={m['Spearman']} "
          f"high-binding MAE={m['high_binding_MAE']} n={len(points)}")


if __name__ == "__main__":
    main()
