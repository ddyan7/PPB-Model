"""Stage 13 entry point: interpretability and chemical plausibility.

Uses the tuned Random Forest on cleaned descriptors (an interpretable, competitive model)
and reports two importance views:
    * permutation importance on the validation set (model-agnostic, MAE-based)
    * impurity-based importance (tree-internal)
Then checks whether the top features align with known PPB drivers (lipophilicity, size,
polar surface area, ionisation, hydrogen bonding) and lists the largest test errors.

SHAP is not used here to avoid a heavy extra dependency; permutation importance provides
an equivalent, defensible attribution for this tabular descriptor model.

Outputs:
    reports/tables/feature_importance.csv
    reports/figures/stage13_feature_importance.png
    reports/tables/largest_errors.csv
    reports/results/stage13_interpretation.json

Usage:
    python scripts/interpret.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from ppb_model.baselines import make_model
from ppb_model.data import load_clean_and_split
from ppb_model.train import FeatureCache, assemble_representation
from ppb_model.targets import TargetTransformer
from ppb_model.utils import Paths, get_logger, load_config, set_seed

# Known PPB drivers (substring match against RDKit descriptor names) for plausibility check.
KNOWN_DRIVERS = {
    "lipophilicity": ["LogP", "SLogP", "MolLogP"],
    "size": ["MolWt", "HeavyAtom", "ExactMolWt", "LabuteASA", "MolMR"],
    "polar_surface_area": ["TPSA", "PSA"],
    "hydrogen_bonding": ["HDonor", "HAcceptor", "NHOH", "NOCount"],
    "aromaticity": ["Aromatic", "aromatic"],
    "ionisation_charge": ["Charge", "NumBasic", "NumAcidic", "PEOE", "EState"],
}


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    seed = cfg["project"]["seed"]
    set_seed(seed)
    paths = Paths.create()
    logger = get_logger("stage13", log_file="logs/stage13_interpret.log")

    data = load_clean_and_split(cfg, "scaffold")
    cache = FeatureCache.load(paths.interim / "features.npz")
    idx = data["idx"]
    fcfg = cfg["features"]
    mats, n_feat, names = assemble_representation(
        "descriptors", cache, idx, fcfg["descriptor_variance_threshold"], fcfg["descriptor_corr_threshold"])

    tuned = json.loads((paths.results / "tuned_hyperparameters.json").read_text(encoding="utf-8"))
    rf_params = tuned["rf_desc"]["best_params"]
    tf = TargetTransformer(cfg["target"]["transform"], float(cfg["target"]["epsilon"]),
                           tuple(cfg["target"]["clip_percent"]))
    y_tr_t = tf.forward(cache.y_percent[idx["train"]])
    y_va_t = tf.forward(cache.y_percent[idx["valid"]])

    model = make_model("rf", seed=seed, **rf_params)
    model.fit(mats["train"], y_tr_t)

    # Permutation importance (validation, MAE-based).
    perm = permutation_importance(model, mats["valid"], y_va_t, n_repeats=10,
                                  random_state=seed, scoring="neg_mean_absolute_error", n_jobs=-1)
    imp_df = pd.DataFrame({
        "feature": names,
        "permutation_importance": perm.importances_mean,
        "permutation_std": perm.importances_std,
        "impurity_importance": model.feature_importances_,
    }).sort_values("permutation_importance", ascending=False).reset_index(drop=True)
    imp_df.to_csv(paths.tables / "feature_importance.csv", index=False)

    top = imp_df.head(20)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top["feature"][::-1], top["permutation_importance"][::-1],
            xerr=top["permutation_std"][::-1], color="seagreen")
    ax.set_xlabel("Permutation importance (validation, MAE-based)")
    ax.set_title("Top 20 descriptors - RF on descriptors (logit target)")
    fig.tight_layout()
    fig.savefig(paths.figures / "stage13_feature_importance.png", dpi=130)
    plt.close(fig)

    # Chemical plausibility: which driver categories appear in the top 20.
    top_features = set(top["feature"])
    plausibility = {}
    for cat, keys in KNOWN_DRIVERS.items():
        hits = [f for f in top_features if any(k in f for k in keys)]
        plausibility[cat] = hits

    # Largest test errors.
    pred_test = tf.inverse(model.predict(mats["test"]))
    y_test = cache.y_percent[idx["test"]]
    err_df = pd.DataFrame({
        "compound_id": [data["df"].iloc[i]["drug_id"] for i in idx["test"]],
        "canonical_smiles": [data["df"].iloc[i]["canonical_smiles"] for i in idx["test"]],
        "observed_ppb": y_test,
        "predicted_ppb": pred_test,
        "absolute_error": np.abs(y_test - pred_test),
    }).sort_values("absolute_error", ascending=False).reset_index(drop=True)
    err_df.head(20).to_csv(paths.tables / "largest_errors.csv", index=False)

    interp = {
        "model": "rf_descriptors_logit(tuned)",
        "n_features": int(n_feat),
        "top10_permutation": imp_df.head(10)[["feature", "permutation_importance"]].to_dict("records"),
        "chemical_plausibility_top20": plausibility,
        "largest_error_max_pct": float(err_df["absolute_error"].max()),
        "n_test_errors_gt_20pct": int((err_df["absolute_error"] > 20).sum()),
    }
    (paths.results / "stage13_interpretation.json").write_text(json.dumps(interp, indent=2), encoding="utf-8")

    logger.info("Top permutation features: %s", list(top["feature"].head(8)))
    print("Top 12 descriptors by permutation importance:")
    print(imp_df.head(12)[["feature", "permutation_importance", "impurity_importance"]].to_string(index=False))
    print("\nChemical plausibility (known PPB drivers found in top 20):")
    for cat, hits in plausibility.items():
        print(f"  {cat:20s}: {hits}")
    print(f"\nLargest test error: {err_df['absolute_error'].max():.1f} pct; "
          f"{interp['n_test_errors_gt_20pct']} test compounds with error > 20 pct")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 13: interpretability.")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
