"""Stage 8-9 entry point: tune the proposed model and evaluate it against the baselines.

Proposed model = a logit-target consensus ensemble over diverse, Optuna-tuned learners:
    * XGBoost on the HYBRID representation (descriptors + Morgan)
    * Random Forest on descriptors
    * HistGradientBoosting on descriptors
The ensemble averages member predictions in logit space; an applicability-domain
(max Tanimoto to train) and an ensemble-disagreement uncertainty are attached.

Tuning uses train->valid only; the test set is scored once at the end.

Outputs:
    reports/results/improved_results.{csv,md,json}
    reports/results/tuned_hyperparameters.json
    reports/results/predictions_final.csv
    models/final_consensus.joblib
    models/tuned_members/*.joblib

Usage:
    python scripts/train_improved_model.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from ppb_model.baselines import make_model
from ppb_model.data import load_clean_and_split
from ppb_model.evaluation import regression_metrics
from ppb_model.results import result_row, save_results_table
from ppb_model.targets import TargetTransformer
from ppb_model.train import ExperimentResult, FeatureCache, assemble_representation
from ppb_model.tuning import tune_model
from ppb_model.uncertainty import (
    applicability_domain_flag,
    ensemble_uncertainty,
    max_train_tanimoto,
)
from ppb_model.utils import Paths, get_logger, load_config, set_seed

# Consensus members: (member_key, model_name, representation)
MEMBERS = [
    ("xgb_hybrid", "xgb", "hybrid"),
    ("rf_desc", "rf", "descriptors"),
    ("hgb_desc", "hgb", "descriptors"),
]
# Also tuned for the representation ablation (hybrid vs descriptors on the same learner).
EXTRA_TUNED = [("xgb_desc", "xgb", "descriptors")]


def _fit_predict(model, X, transformer):
    return transformer.inverse(model.predict(X))


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    seed = cfg["project"]["seed"]
    set_seed(seed)
    paths = Paths.create()
    logger = get_logger("stage8", log_file="logs/stage8_improved.log")

    split_method = cfg["split"]["method"]
    data = load_clean_and_split(cfg, split_method)
    idx = data["idx"]
    cache = FeatureCache.load(paths.interim / "features.npz")
    y_pct = cache.y_percent

    transform = cfg["target"]["transform"]
    tf = TargetTransformer(method=transform, epsilon=float(cfg["target"]["epsilon"]),
                           clip_percent=tuple(cfg["target"]["clip_percent"]))
    y_tr_t = tf.forward(y_pct[idx["train"]])
    y_va_pct, y_te_pct = y_pct[idx["valid"]], y_pct[idx["test"]]

    fcfg = cfg["features"]
    ecfg = cfg["evaluation"]
    metric_kwargs = dict(high_binding_threshold=ecfg["high_binding_threshold_percent"],
                         bands=tuple(ecfg["binding_bands_percent"]))
    n_trials = cfg["tuning"]["n_trials"]

    # ---- Assemble each representation once -----------------------------------
    reps = {}
    for rep in ("descriptors", "hybrid"):
        mats, n_feat, _ = assemble_representation(
            rep, cache, idx, fcfg["descriptor_variance_threshold"], fcfg["descriptor_corr_threshold"])
        reps[rep] = {"mats": mats, "n_features": n_feat}

    # ---- Tune each member + extras -------------------------------------------
    tuned: dict[str, dict] = {}
    tuned_params_record = {}
    members_dir = paths.models / "tuned_members"
    members_dir.mkdir(parents=True, exist_ok=True)

    for key, model_name, rep in MEMBERS + EXTRA_TUNED:
        mats = reps[rep]["mats"]
        logger.info("Tuning %s (%s on %s, %d trials)...", key, model_name, rep, n_trials)
        tr = tune_model(model_name, mats["train"], y_tr_t, mats["valid"], y_va_pct, tf,
                        n_trials=n_trials, seed=seed)
        model = make_model(model_name, seed=seed, **tr.best_params)
        model.fit(mats["train"], y_tr_t)
        joblib.dump(model, members_dir / f"{key}.joblib")
        tuned[key] = {"model": model, "rep": rep, "model_name": model_name,
                      "n_features": reps[rep]["n_features"],
                      "valid_pred": _fit_predict(model, mats["valid"], tf),
                      "valid_logit": model.predict(mats["valid"]),
                      "test_pred": _fit_predict(model, mats["test"], tf),
                      "test_logit": model.predict(mats["test"])}
        tuned_params_record[key] = {
            "model": model_name, "representation": rep, "best_params": tr.best_params,
            "best_valid_mae": round(tr.best_valid_mae, 4), "n_trials": tr.n_trials,
            "search_space": tr.search_space}
        logger.info("  best valid MAE=%.3f params=%s", tr.best_valid_mae, tr.best_params)

    (paths.results / "tuned_hyperparameters.json").write_text(
        json.dumps(tuned_params_record, indent=2), encoding="utf-8")

    # ---- Consensus ensemble (average in logit space) -------------------------
    member_keys = [k for k, _, _ in MEMBERS]
    valid_logit_stack = np.vstack([tuned[k]["valid_logit"] for k in member_keys])
    test_logit_stack = np.vstack([tuned[k]["test_logit"] for k in member_keys])
    consensus_valid_pct = tf.inverse(valid_logit_stack.mean(axis=0))
    consensus_test_pct = tf.inverse(test_logit_stack.mean(axis=0))

    # ---- Build results rows (test metrics) -----------------------------------
    def make_result(pred_valid, pred_test, model_label, rep_label, n_feat) -> ExperimentResult:
        meta = {"model": model_label, "representation": rep_label,
                "target_transformation": transform, "seed": seed, "n_features": int(n_feat),
                "train_size": len(idx["train"]), "valid_size": len(idx["valid"]),
                "test_size": len(idx["test"]), "train_time_s": None,
                "predict_time_s": None, "model_params": {}}
        return ExperimentResult(
            meta,
            regression_metrics(y_va_pct, pred_valid, **metric_kwargs),
            regression_metrics(y_te_pct, pred_test, **metric_kwargs),
            pred_test, pred_valid, None, tf)

    rows = []
    results_by_id = {}
    for key in member_keys + [k for k, _, _ in EXTRA_TUNED]:
        info = tuned[key]
        res = make_result(info["valid_pred"], info["test_pred"], info["model_name"],
                          info["rep"], info["n_features"])
        exp_id = f"T_{key}"
        rows.append(result_row(res, experiment_id=exp_id, split_method=split_method,
                               hyperparameter_method="optuna",
                               notes=f"tuned;valid_MAE={res.valid_metrics['MAE']:.3f};"
                                     f"fu_GMFE_high={res.test_metrics['fu_GMFE_high']:.3f}"))
        results_by_id[exp_id] = res

    consensus_res = make_result(consensus_valid_pct, consensus_test_pct,
                                "consensus_ensemble", "hybrid+descriptors",
                                reps["hybrid"]["n_features"])
    rows.append(result_row(consensus_res, experiment_id="T_consensus", split_method=split_method,
                           hyperparameter_method="optuna",
                           notes=f"members={'+'.join(member_keys)};"
                                 f"valid_MAE={consensus_res.valid_metrics['MAE']:.3f};"
                                 f"fu_GMFE_high={consensus_res.test_metrics['fu_GMFE_high']:.3f}"))
    results_by_id["T_consensus"] = consensus_res

    df = save_results_table(rows, paths.results, "improved_results")

    # ---- Model selection by VALIDATION MAE (test stays frozen for selection) --
    valid_maes = {rid: res.valid_metrics["MAE"] for rid, res in results_by_id.items()}
    best_id = min(valid_maes, key=valid_maes.get)
    best_res = results_by_id[best_id]
    logger.info("Selected by validation MAE: %s (valid=%.3f, test=%.3f)",
                best_id, best_res.valid_metrics["MAE"], best_res.test_metrics["MAE"])

    # ---- Applicability domain + uncertainty (on the selected model) ----------
    morgan_train = cache.morgan[idx["train"]]
    # Data-driven AD threshold: 5th percentile of train-to-other-train max similarity.
    ad_threshold = float(np.percentile(_second_best_selfsim(morgan_train), 5))

    def ad_and_uncertainty(split_name):
        rid_idx = idx[split_name]
        max_sim = max_train_tanimoto(cache.morgan[rid_idx], morgan_train)
        logit_stack = np.vstack([tf.inverse(m["model"].predict(
            reps[m["rep"]]["mats"][split_name])) for m in [tuned[k] for k in member_keys]])
        unc = ensemble_uncertainty(logit_stack)
        return max_sim, unc

    # ---- Final prediction file (all splits, selected model) ------------------
    split_df = pd.read_csv(paths.splits / f"{split_method}_split.csv")
    scaffold_map = dict(zip(split_df["row_index"], split_df["scaffold"]))
    selected_pred = _selected_predictions(best_id, tuned, member_keys, tf, reps, idx, consensus=(best_id == "T_consensus"))

    pred_rows = []
    for split_name in ("train", "valid", "test"):
        rid_idx = idx[split_name]
        max_sim, unc = ad_and_uncertainty(split_name)
        in_ad = applicability_domain_flag(max_sim, ad_threshold)
        obs = y_pct[rid_idx]
        pred = selected_pred[split_name]
        for j, row_index in enumerate(rid_idx):
            drow = data["df"].iloc[row_index]
            pred_rows.append({
                "compound_id": drow["drug_id"],
                "canonical_smiles": drow["canonical_smiles"],
                "data_split": split_name,
                "observed_ppb": float(obs[j]),
                "predicted_ppb": float(pred[j]),
                "observed_fraction_unbound": float((100 - obs[j]) / 100),
                "predicted_fraction_unbound": float((100 - pred[j]) / 100),
                "absolute_error": float(abs(obs[j] - pred[j])),
                "scaffold": scaffold_map.get(int(row_index), ""),
                "maximum_training_similarity": float(max_sim[j]),
                "uncertainty_score": float(unc[j]),
                "in_applicability_domain": bool(in_ad[j]),
                "model_name": best_id,
            })
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(paths.results / "predictions_final.csv", index=False)

    # ---- Persist deployable, compressed model bundles ------------------------
    # Refit the (fast, deterministic) descriptor cleaner so the bundle can predict
    # on new SMILES without any external state; include training fingerprints for AD.
    from ppb_model.features import DescriptorCleaner
    cleaner = DescriptorCleaner(fcfg["descriptor_variance_threshold"],
                                fcfg["descriptor_corr_threshold"]).fit(
        cache.desc_raw[idx["train"]], cache.desc_names)
    member_reps = {k: r for k, _m, r in MEMBERS}
    common = {"cleaner": cleaner, "transformer": tf,
              "morgan": {"radius": fcfg["morgan"]["radius"], "n_bits": fcfg["morgan"]["n_bits"]},
              "morgan_train": morgan_train.astype(np.uint8), "ad_threshold": ad_threshold}
    joblib.dump({"kind": "consensus", "members": {k: tuned[k]["model"] for k in member_keys},
                 "member_reps": member_reps, "selected_id": best_id, **common},
                paths.models / "final_consensus.joblib", compress=3)
    joblib.dump({"kind": "single", "model": tuned["xgb_hybrid"]["model"],
                 "representation": "hybrid", **common},
                paths.models / "final_xgb_hybrid.joblib", compress=3)

    # ---- Report vs baseline bar ----------------------------------------------
    baseline_csv = paths.results / "baseline_results.csv"
    baseline_note = ""
    if baseline_csv.is_file():
        bdf = pd.read_csv(baseline_csv)
        b_mae = bdf["MAE"].min()
        b_high = bdf["high_binding_MAE"].min()
        baseline_note = (f"best baseline MAE={b_mae:.3f}, best baseline high-binding MAE={b_high:.3f}")

    print(df[["experiment_id", "MAE", "RMSE", "R2", "Spearman",
              "high_binding_MAE", "fraction_unbound_MAE"]].to_string(index=False))
    print(f"\nSelected (by valid MAE): {best_id}")
    print(f"  test MAE={best_res.test_metrics['MAE']:.3f} | high-binding MAE="
          f"{best_res.test_metrics['high_binding_MAE']:.3f} | Spearman="
          f"{best_res.test_metrics['Spearman']:.3f}")
    print(f"AD threshold (Tanimoto)={ad_threshold:.3f}; "
          f"test in-domain fraction={pred_df.query('data_split==\"test\"')['in_applicability_domain'].mean():.2f}")
    if baseline_note:
        print(baseline_note)


def _second_best_selfsim(morgan_train: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Max Tanimoto of each train molecule to any *other* train molecule (self excluded)."""
    t = (morgan_train > 0).astype(np.float32)
    t_sum = t.sum(axis=1)
    n = len(t)
    out = np.empty(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        qb = t[start:start + batch_size]
        inter = qb @ t.T
        qb_sum = qb.sum(axis=1, keepdims=True)
        union = qb_sum + t_sum[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(union > 0, inter / union, 0.0)
        # zero out self-match
        for r in range(qb.shape[0]):
            sim[r, start + r] = -1.0
        out[start:start + qb.shape[0]] = sim.max(axis=1)
    return out


def _selected_predictions(best_id, tuned, member_keys, tf, reps, idx, consensus: bool):
    """Return {split: pred_pct} for the selected model."""
    out = {}
    for split_name in ("train", "valid", "test"):
        if consensus:
            stack = np.vstack([tuned[k]["model"].predict(reps[tuned[k]["rep"]]["mats"][split_name])
                               for k in member_keys])
            out[split_name] = tf.inverse(stack.mean(axis=0))
        else:
            key = best_id.replace("T_", "")
            m = tuned[key]
            out[split_name] = tf.inverse(m["model"].predict(reps[m["rep"]]["mats"][split_name]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 8-9: tuned improved model + consensus.")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
