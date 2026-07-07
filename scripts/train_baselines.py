"""Stage 7 entry point: train and evaluate the baseline model matrix.

Baselines (all on the primary scaffold split; test set reported, not tuned):
    median            - non-informative floor
    ridge/elasticnet  - linear, descriptors, {raw, logit}
    rf                - descriptors, {raw, logit}
    hgb / xgb         - strong conventional, descriptors, {raw, logit}
    rf                - morgan (logit), maccs (logit) fingerprint references

Outputs:
    reports/results/baseline_results.{csv,md,json}

Usage:
    python scripts/train_baselines.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse

from ppb_model.data import load_clean_and_split
from ppb_model.results import result_row, save_results_table
from ppb_model.train import FeatureCache, run_experiment
from ppb_model.utils import Paths, get_logger, load_config, set_seed

# (model, representation, transform)
BASELINES = [
    ("median", "descriptors", "none"),
    ("ridge", "descriptors", "none"),
    ("ridge", "descriptors", "logit"),
    ("elasticnet", "descriptors", "logit"),
    ("rf", "descriptors", "none"),
    ("rf", "descriptors", "logit"),
    ("hgb", "descriptors", "none"),
    ("hgb", "descriptors", "logit"),
    ("xgb", "descriptors", "none"),
    ("xgb", "descriptors", "logit"),
    ("rf", "morgan", "logit"),
    ("rf", "maccs", "logit"),
]


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["project"]["seed"])
    paths = Paths.create()
    logger = get_logger("stage7", log_file="logs/stage7_baselines.log")

    split_method = cfg["split"]["method"]
    data = load_clean_and_split(cfg, split_method)
    cache = FeatureCache.load(paths.interim / "features.npz")

    rows = []
    for i, (model_name, rep, transform) in enumerate(BASELINES):
        exp_id = f"B{i:02d}_{model_name}_{rep}_{transform}"
        res = run_experiment(model_name=model_name, representation=rep,
                             transform_method=transform, cache=cache,
                             idx=data["idx"], config=cfg)
        rows.append(result_row(res, experiment_id=exp_id, split_method=split_method))
        m = res.test_metrics
        logger.info("%-34s test MAE=%.3f RMSE=%.3f R2=%.3f highMAE=%.3f fuGMFE=%.2f",
                    exp_id, m["MAE"], m["RMSE"], m["R2"], m["high_binding_MAE"], m["fu_GMFE_high"])

    df = save_results_table(rows, paths.results, "baseline_results")
    print(df[["experiment_id", "MAE", "RMSE", "R2", "Spearman",
              "high_binding_MAE", "fraction_unbound_MAE"]].to_string(index=False))
    best = df.loc[df["MAE"].idxmin()]
    print(f"\nBest baseline by test MAE: {best['experiment_id']} (MAE={best['MAE']})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 7: baseline model matrix.")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
