"""Stage 4-5 entry point: create and verify leakage-safe data splits.

Produces:
    data/splits/scaffold_split.csv   - primary (record_id, split, scaffold)
    data/splits/random_split.csv     - secondary reference
    reports/results/stage5_split_summary.json

Leakage guards asserted here:
    * no scaffold appears in more than one partition (scaffold split)
    * no record_id appears in more than one partition (both splits)
    * partitions cover every row exactly once

Usage:
    python scripts/create_splits.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from ppb_model.splitting import (
    assignment_frame,
    bemis_murcko_scaffold,
    random_split,
    scaffold_split,
)
from ppb_model.utils import Paths, get_logger, load_config, set_seed


def _target_stats(y: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "mean": float(np.mean(y)),
        "median": float(np.median(y)),
        "std": float(np.std(y)),
        "pct_ge_90": float(np.mean(y >= 90) * 100),
        "pct_ge_99": float(np.mean(y >= 99) * 100),
    }


def _assert_disjoint(frame: pd.DataFrame, key: str, label: str, logger) -> None:
    dupe = frame.groupby(key)["split"].nunique()
    crossing = dupe[dupe > 1]
    if len(crossing) > 0:
        raise AssertionError(f"{label}: {len(crossing)} {key}(s) cross split boundaries")
    logger.info("%s: no %s crosses split boundaries (OK)", label, key)


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    seed = cfg["split"]["seed"]
    set_seed(seed)
    paths = Paths.create()
    logger = get_logger("stage5", log_file="logs/stage5_splits.log")

    df = pd.read_csv(paths.processed / "ppbr_az_human_clean.csv")
    smiles = df["canonical_smiles"].tolist()
    y = df["y_percent"].to_numpy(dtype=float)
    record_ids = df["record_id"]
    scaffolds = [bemis_murcko_scaffold(s) for s in smiles]
    n_unique_scaffolds = len(set(scaffolds))
    n_acyclic = sum(1 for s in scaffolds if s == "")
    logger.info("%d rows | %d unique scaffolds | %d acyclic (empty scaffold)",
                len(df), n_unique_scaffolds, n_acyclic)

    tf, vf, tsf = cfg["split"]["train_frac"], cfg["split"]["valid_frac"], cfg["split"]["test_frac"]

    # ---- Scaffold split (primary) -------------------------------------------
    sc = scaffold_split(smiles, tf, vf, tsf, seed=seed)
    sc_frame = assignment_frame(sc, record_ids)
    sc_frame["scaffold"] = sc_frame["row_index"].map(lambda i: scaffolds[i])
    _assert_disjoint(sc_frame, "scaffold", "scaffold-split", logger)
    _assert_disjoint(sc_frame, "record_id", "scaffold-split", logger)
    assert len(sc_frame) == len(df), "scaffold split does not cover all rows exactly once"
    sc_path = paths.splits / "scaffold_split.csv"
    sc_frame.to_csv(sc_path, index=False)

    # ---- Random split (secondary) -------------------------------------------
    rnd = random_split(len(df), tf, vf, tsf, seed=seed)
    rnd_frame = assignment_frame(rnd, record_ids)
    _assert_disjoint(rnd_frame, "record_id", "random-split", logger)
    assert len(rnd_frame) == len(df), "random split does not cover all rows exactly once"
    rnd_path = paths.splits / "random_split.csv"
    rnd_frame.to_csv(rnd_path, index=False)

    # ---- Summary -------------------------------------------------------------
    def split_target_stats(frame):
        out = {}
        for name in ("train", "valid", "test"):
            idx = frame.loc[frame["split"] == name, "row_index"].to_numpy()
            out[name] = _target_stats(y[idx])
        return out

    summary = {
        "n_rows": int(len(df)),
        "n_unique_scaffolds": n_unique_scaffolds,
        "n_acyclic": n_acyclic,
        "fractions": {"train": tf, "valid": vf, "test": tsf},
        "seed": seed,
        "scaffold_split": {
            "sizes": {k: int(len(v)) for k, v in sc.items()},
            "unique_scaffolds_per_split": {
                name: int(sc_frame.loc[sc_frame["split"] == name, "scaffold"].nunique())
                for name in ("train", "valid", "test")
            },
            "target_stats": split_target_stats(sc_frame),
        },
        "random_split": {
            "sizes": {k: int(len(v)) for k, v in rnd.items()},
            "target_stats": split_target_stats(rnd_frame),
        },
        "outputs": {"scaffold_csv": str(sc_path), "random_csv": str(rnd_path)},
    }
    out_path = paths.results / "stage5_split_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Wrote %s", sc_path)
    logger.info("Wrote %s", rnd_path)
    logger.info("Wrote %s", out_path)
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 4-5: create scaffold and random splits.")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
