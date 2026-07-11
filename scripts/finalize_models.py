"""Post-Stage-16: build compact, deployable model bundles from the tuned members.

Does NOT re-tune. Loads the already-saved tuned member models, refits the (fast,
deterministic) descriptor cleaner on the primary scaffold-train split, and writes two
self-contained, compressed bundles that can actually predict on new SMILES:

    models/final_xgb_hybrid.joblib   - lean single model (small footprint)
    models/final_consensus.joblib    - full ensemble (re-saved, compressed)

Usage:
    python scripts/finalize_models.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np

from ppb_model.data import load_clean_and_split
from ppb_model.features import DescriptorCleaner
from ppb_model.predict import load_bundle, predict_smiles
from ppb_model.targets import TargetTransformer
from ppb_model.train import FeatureCache
from ppb_model.utils import Paths, get_logger, load_config, set_seed

MEMBER_REPS = {"xgb_hybrid": "hybrid", "rf_desc": "descriptors", "hgb_desc": "descriptors"}


def run(config_path: str, lean_only: bool = False) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["project"]["seed"])
    paths = Paths.create()
    logger = get_logger("finalize", log_file="logs/finalize_models.log")

    data = load_clean_and_split(cfg, "scaffold")
    cache = FeatureCache.load(paths.interim / "features.npz")
    tr = data["idx"]["train"]
    fcfg = cfg["features"]

    # Refit descriptor cleaner on the primary scaffold-train (deterministic, no tuning).
    cleaner = DescriptorCleaner(fcfg["descriptor_variance_threshold"],
                                fcfg["descriptor_corr_threshold"]).fit(cache.desc_raw[tr], cache.desc_names)
    tf = TargetTransformer(cfg["target"]["transform"], float(cfg["target"]["epsilon"]),
                           tuple(cfg["target"]["clip_percent"]))
    morgan_train = cache.morgan[tr]
    morgan_cfg = {"radius": fcfg["morgan"]["radius"], "n_bits": fcfg["morgan"]["n_bits"]}

    # Load AD threshold from the existing consensus bundle (or fall back to 0.25).
    existing = paths.models / "final_consensus.joblib"
    ad_threshold = 0.25
    if existing.is_file():
        try:
            ad_threshold = float(joblib.load(existing).get("ad_threshold", 0.25))
        except Exception:  # noqa: BLE001
            pass

    members = {name: joblib.load(paths.models / "tuned_members" / f"{name}.joblib")
               for name in MEMBER_REPS}

    common = {"cleaner": cleaner, "transformer": tf, "morgan": morgan_cfg,
              "morgan_train": morgan_train.astype(np.uint8), "ad_threshold": ad_threshold}

    lean = {"kind": "single", "model": members["xgb_hybrid"],
            "representation": "hybrid", **common}
    consensus = {"kind": "consensus", "members": members, "member_reps": MEMBER_REPS, **common}

    lean_path = paths.models / "final_xgb_hybrid.joblib"
    cons_path = paths.models / "final_consensus.joblib"
    joblib.dump(lean, lean_path, compress=3)
    joblib.dump(consensus, cons_path, compress=3)

    lean_mb = lean_path.stat().st_size / 1e6
    cons_mb = cons_path.stat().st_size / 1e6
    logger.info("Wrote lean bundle %.1f MB, consensus bundle %.1f MB", lean_mb, cons_mb)

    # Smoke test: both bundles predict on a few known drugs.
    demo = ["CC(=O)Oc1ccccc1C(=O)O",              # aspirin
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",         # ibuprofen (highly bound)
            "OCC1OC(O)C(O)C(O)C1O"]               # glucose (low binding)
    for path, kind in ((lean_path, "single"), (cons_path, "consensus")):
        out = predict_smiles(load_bundle(path), demo)
        assert out["predicted_ppb"].notna().all(), f"{kind} bundle failed to predict"
        logger.info("%s bundle predictions: %s", kind,
                    [round(v, 1) for v in out["predicted_ppb"]])

    print(f"Lean bundle:      {lean_path.name}  ({lean_mb:.1f} MB)")
    print(f"Consensus bundle: {cons_path.name}  ({cons_mb:.1f} MB)")
    out = predict_smiles(load_bundle(lean_path), demo)
    print("\nSmoke-test predictions (lean xgb_hybrid):")
    print(out[["input_smiles", "predicted_ppb", "max_training_similarity",
               "in_applicability_domain"]].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build compact deployable model bundles.")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
