"""CLI: predict PPB (%) for new SMILES using a saved model bundle.

Examples:
    # single SMILES
    python scripts/predict.py --smiles "CC(=O)Oc1ccccc1C(=O)O"
    # a file with one SMILES per line, custom bundle, write CSV
    python scripts/predict.py --input compounds.smi --bundle models/final_consensus.joblib --out preds.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ppb_model.predict import load_bundle, predict_smiles
from ppb_model.utils import project_root


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict PPB for new SMILES.")
    ap.add_argument("--smiles", nargs="*", help="One or more SMILES strings.")
    ap.add_argument("--input", help="Text file with one SMILES per line.")
    ap.add_argument("--bundle", default="models/final_xgb_hybrid.joblib",
                    help="Path to a saved model bundle.")
    ap.add_argument("--out", help="Optional CSV output path.")
    args = ap.parse_args()

    smiles: list[str] = list(args.smiles or [])
    if args.input:
        smiles += [ln.strip() for ln in Path(args.input).read_text(encoding="utf-8").splitlines()
                   if ln.strip()]
    if not smiles:
        ap.error("Provide --smiles and/or --input")

    bundle_path = Path(args.bundle)
    if not bundle_path.is_absolute():
        bundle_path = project_root() / bundle_path
    df = predict_smiles(load_bundle(bundle_path), smiles)

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Wrote {len(df)} predictions to {args.out}")
    else:
        print(df[["input_smiles", "predicted_ppb", "predicted_fraction_unbound",
                  "max_training_similarity", "in_applicability_domain", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
