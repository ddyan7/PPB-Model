"""Stage 6 entry point: precompute and cache raw molecular representations.

Caches per-molecule, structure-only features (leakage-safe — no train/test statistics
involved). The train-only descriptor *cleaning/scaling* is fitted later at model time.

Outputs:
    data/interim/features.npz   - desc_raw, desc_names, morgan, maccs, record_id, y_percent
    reports/results/stage6_features_summary.json

Usage:
    python scripts/build_features.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from ppb_model.features import compute_descriptor_matrix, maccs_matrix, morgan_matrix
from ppb_model.utils import Paths, get_logger, load_config, set_seed


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["project"]["seed"])
    paths = Paths.create()
    logger = get_logger("stage6", log_file="logs/stage6_features.log")

    df = pd.read_csv(paths.processed / "ppbr_az_human_clean.csv")
    smiles = df["canonical_smiles"].tolist()
    logger.info("Building features for %d molecules", len(smiles))

    desc_raw, desc_names = compute_descriptor_matrix(smiles)
    logger.info("Descriptors: %s (raw)", desc_raw.shape)
    radius = cfg["features"]["morgan"]["radius"]
    n_bits = cfg["features"]["morgan"]["n_bits"]
    morgan = morgan_matrix(smiles, radius, n_bits)
    logger.info("Morgan r%d %dbit: %s", radius, n_bits, morgan.shape)
    maccs = maccs_matrix(smiles)
    logger.info("MACCS: %s", maccs.shape)

    cache = paths.interim / "features.npz"
    np.savez_compressed(
        cache,
        desc_raw=desc_raw,
        desc_names=np.array(desc_names, dtype=object),
        morgan=morgan,
        maccs=maccs,
        record_id=df["record_id"].to_numpy(),
        y_percent=df["y_percent"].to_numpy(dtype=float),
        morgan_radius=radius,
        morgan_nbits=n_bits,
    )
    summary = {
        "n_molecules": int(len(smiles)),
        "descriptors_raw": int(desc_raw.shape[1]),
        "morgan": {"radius": radius, "n_bits": n_bits, "mean_bits_per_mol": float(morgan.sum(1).mean())},
        "maccs_bits": int(maccs.shape[1]),
        "cache": str(cache),
    }
    (paths.results / "stage6_features_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Cached features to %s", cache)
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 6: precompute molecular features.")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
