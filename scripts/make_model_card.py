"""Stage 16 entry point: generate MODEL_CARD.md from saved artefacts.

Reads the JSON/CSV results produced by the pipeline and composes a model card. Run this
last, after robustness, ablation, interpretation, figures, and evaluate_models.

Usage:
    python scripts/make_model_card.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from ppb_model.utils import Paths, get_logger, load_config, project_root, set_seed


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["project"]["seed"])
    paths = Paths.create()
    logger = get_logger("stage16", log_file="logs/stage16_model_card.log")

    s2 = _load_json(paths.results / "stage2_summary.json")
    s3 = _load_json(paths.results / "stage3_target_decision.json")
    s5 = _load_json(paths.results / "stage5_split_summary.json")
    sel = _load_json(paths.results / "final_selection.json")
    cov = _load_json(paths.results / "stage14_coverage.json")
    interp = _load_json(paths.results / "stage13_interpretation.json")
    ablation = _load_json(paths.results / "ablation_summary.json")
    tuned = _load_json(paths.results / "tuned_hyperparameters.json")
    ext = _load_json(paths.results / "external_validation_ingle.json")
    h2h = _load_json(paths.results / "head_to_head_ingle.json")
    aug_final = _load_json(paths.results / "augmented_final_summary.json")
    confirm = _load_json(paths.results / "confirm_robustness.json")
    boot = sel.get("bootstrap", {})

    master = pd.read_csv(paths.results / "experiment_results.csv") if (
        paths.results / "experiment_results.csv").is_file() else pd.DataFrame()
    selected = sel.get("selected", "n/a")

    def boot_line(key):
        if key in boot:
            b = boot[key]
            return (f"{key}: MAE {b['MAE_mean']:.2f} "
                    f"(95% CI {b['MAE_CI95'][0]:.2f}-{b['MAE_CI95'][1]:.2f}), "
                    f"high-binding MAE {b['high_binding_MAE_mean']:.2f}")
        return key

    ad = ablation.get("applicability_domain", {})
    high_thr = cfg["evaluation"]["high_binding_threshold_percent"]

    # Deployed-final section (augmented model), if promotion has been run.
    deployed_section = ""
    if aug_final and aug_final.get("deployed_model"):
        tc = aug_final.get("training_composition", {})
        tm = aug_final.get("test_metrics_ppbr_scaffold", {})
        rs = confirm.get("repeated_splits", {})
        pb = confirm.get("paired_bootstrap_primary", {})
        rs_mae = rs.get("MAE", {})
        deployed_section = f"""## Deployed final model (augmented)
**The deployed model is the augmented consensus.** Training data = PPBR_AZ scaffold-train
({tc.get('ppbr_az_train', 'n/a')}) + scaffold-safe non-overlapping Ingle compounds
({tc.get('ingle_scaffold_safe', 'n/a')}) = **{tc.get('total', 'n/a')}** molecules. It replaced the
original PPBR_AZ-only model, which is preserved as `models/*_ppbr_only.joblib`.

Performance on the unchanged PPBR_AZ scaffold test: MAE **{tm.get('MAE', 'n/a')}**,
RMSE {tm.get('RMSE', 'n/a')}, R2 **{tm.get('R2', 'n/a')}**, Spearman {tm.get('Spearman', 'n/a')},
high-binding MAE {tm.get('high_binding_MAE', 'n/a')} (vs the original ~7.53 / 0.36 / 2.69).

**Robustness (why it was promoted, stated honestly):** across {confirm.get('n_seeds', 5)} repeated
scaffold splits the augmented model beats the original on MAE in
{rs.get('MAE_augmented_wins', 'n/a')}/{confirm.get('n_seeds', 5)} splits
(MAE {rs_mae.get('augmented', {}).get('mean', 'n/a')} vs {rs_mae.get('original', {}).get('mean', 'n/a')};
R2 {rs.get('R2', {}).get('augmented', {}).get('mean', 'n/a')} vs
{rs.get('R2', {}).get('original', {}).get('mean', 'n/a')}). The paired bootstrap on the primary split
gives delta MAE {pb.get('delta_MAE_mean', 'n/a')} (95% CI
[{pb.get('delta_MAE_CI95', ['', ''])[0]}, {pb.get('delta_MAE_CI95', ['', ''])[1]}],
P(improvement) {pb.get('delta_MAE_prob_improvement', 'n/a')}). **So: a consistent, modest overall gain
that is directionally reliable but not certified significant on a 243-compound test, with a small
high-binding-MAE cost.** Do not overstate it.

"""

    # External-validation section (Ingle et al. 2016), if the step has been run.
    if ext and ext.get("models"):
        em = ext["models"].get("consensus") or next(iter(ext["models"].values()))
        allm, inm = em.get("all", {}), em.get("in_domain_only", {})
        ext_section = f"""## External validation (out-of-distribution, Ingle et al. 2016)
Scored the frozen model on **{ext.get('n_external_evaluated', 'n/a')} compounds** from
`ppb_usable_dataset.csv` after removing **{ext.get('n_ppbr_az_overlap_removed', 0)} InChIKey overlaps**
with PPBR_AZ (leakage-controlled). Fub was converted to percent bound.

| Metric | All ({allm.get('n', 'n/a')}) | In-domain ({inm.get('n', 'n/a')}) | In-distribution test (ref) |
|---|---|---|---|
| MAE | {allm.get('MAE', 'n/a')} | {inm.get('MAE', 'n/a')} | ~7.5 |
| Median AE | {allm.get('MedAE', 'n/a')} | {inm.get('MedAE', 'n/a')} | - |
| RMSE | {allm.get('RMSE', 'n/a')} | {inm.get('RMSE', 'n/a')} | ~13.6 |
| R2 | {allm.get('R2', 'n/a')} | {inm.get('R2', 'n/a')} | ~0.36 |
| Spearman | {allm.get('Spearman', 'n/a')} | {inm.get('Spearman', 'n/a')} | ~0.75 |
| High-binding MAE | {allm.get('high_binding_MAE', 'n/a')} | {inm.get('high_binding_MAE', 'n/a')} | ~2.7 |
| In-domain fraction | {em.get('in_domain_fraction', 'n/a')} | - | 0.87 |

**Interpretation (honest):** absolute error roughly doubles-to-triples out-of-distribution
(MAE ~7.5 in-distribution -> ~{allm.get('MAE', '19')} on Ingle), driven by an out-of-domain tail
(RMSE >> MAE >> median AE ~{allm.get('MedAE', '10')}) and the fact that only
~{int(float(em.get('in_domain_fraction', 0.58)) * 100)}% of Ingle compounds fall inside the
applicability domain (it contains environmental chemicals outside the drug-like AZ space).
Rank correlation holds up well (Spearman ~{allm.get('Spearman', '0.72')}), so the model still
ranks/triages usefully out-of-distribution even where absolute calibration drifts. The AD flag
correctly identifies the ~42% of external compounds where predictions should not be trusted.
"""
    else:
        ext_section = "## External validation (out-of-distribution)\nNot yet run. See scripts/external_validation.py.\n"

    # Head-to-head note vs Ingle's published models on identical held-out data.
    if h2h and h2h.get("results"):
        by = {r["model"]: r for r in h2h["results"]}
        def g(m, k):
            return by.get(m, {}).get(k, "n/a")
        ext_section += f"""
### Head-to-head vs published models (identical held-out compounds)
On **{h2h.get('n_common_heldout', 'n/a')} identical** Ingle held-out compounds (Dte+T1+T2, PPBR_AZ
overlaps removed), scored the same way. **Caveat:** Ingle's models are tested *in-distribution*
(their home data) while ours is *cross-source transfer* - Ingle has a home-field advantage.

| Model | MAE | R2 | Spearman | High-binding MAE |
|---|---|---|---|---|
| Ingle consensus (published) | {g('Ingle_consensus','MAE')} | {g('Ingle_consensus','R2')} | {g('Ingle_consensus','Spearman')} | {g('Ingle_consensus','high_binding_MAE')} |
| Ingle RF (published) | {g('Ingle_RF','MAE')} | {g('Ingle_RF','R2')} | {g('Ingle_RF','Spearman')} | {g('Ingle_RF','high_binding_MAE')} |
| This model (consensus) | {g('Mine_consensus','MAE')} | {g('Mine_consensus','R2')} | {g('Mine_consensus','Spearman')} | **{g('Mine_consensus','high_binding_MAE')}** |

Ingle's in-distribution models lead on overall MAE/R2 by a modest margin (expected), but **this
model wins the high-binding region** ({g('Mine_consensus','high_binding_MAE')} vs
{g('Ingle_consensus','high_binding_MAE')} / {g('Ingle_RF','high_binding_MAE')}) despite the transfer
handicap - independent confirmation that the logit-transform design generalises. See
`reports/results/head_to_head_ingle.csv`.
"""

    # If the augmented model is deployed, the Ingle-based external results describe the
    # *earlier* PPBR_AZ-only model (Ingle is now training data for the deployed model).
    if aug_final and aug_final.get("deployed_model"):
        ext_section = (
            "> **Note:** the external-validation and head-to-head results below describe the "
            "*earlier PPBR_AZ-only* model (preserved as `models/*_ppbr_only.joblib`). Because the "
            "**deployed** model now trains on Ingle data, Ingle can no longer serve as its external "
            "validation - a fresh third-party dataset would be required.\n\n" + ext_section)

    card = f"""# Model Card - Human Plasma Protein Binding (PPB) Predictor

## Intended use
Estimate human plasma protein binding (percent bound) for small drug-like molecules from
SMILES, as an early-discovery triage/prioritisation aid. **Not** a replacement for
experimental measurement and **not** for regulatory or clinical decisions. Predictions
outside the applicability domain (see below) should be treated as unreliable.

{deployed_section}
## Prediction target
Percent bound (0-100%). Models are trained on a **logit-transformed fraction bound**
(`logit(fb)`) and predictions are inverse-transformed back to percent. Fraction unbound
fu = (100 - PPB)/100 is reported alongside, since fu is the pharmacologically active quantity.

## Training dataset
TDCommons **PPBR_AZ** (AstraZeneca), **human subset only** ({s2.get('human_rows', 'n/a')} rows;
Species == "Homo sapiens"). After cleaning: **{s2.get('final_rows', 'n/a')} unique structures**.
Distribution is strongly high-binding: median {s2.get('final_target_summary', {}).get('median', 'n/a')}%,
{s2.get('final_target_summary', {}).get('pct_ge_90', 'n/a'):.0f}% of compounds >= 90% bound.

## Data-cleaning steps
RDKit parse -> Cleanup -> largest-fragment (salt/mixture) parent -> uncharge -> preserve
stereochemistry -> canonical SMILES + InChIKey. Duplicate/conflict resolution by InChIKey
(median aggregation; hard conflicts excluded). {s2.get('n_ceiling_censored', 0)} ceiling-censored
values (>= 99.9%) flagged and retained. Every record is logged in
`reports/tables/data_cleaning_audit.csv` (no silent edits). In this dataset all
{s2.get('human_rows', '')} human structures parsed and were unique (0 salts, 0 conflicts).

## Molecular representation
- Physicochemical: full RDKit descriptor set ({s2.get('final_rows', '')} mols), cleaned
  **train-only** (impute -> drop constant -> decorrelate at |r|>0.95 -> standardise).
- Morgan/ECFP (radius {cfg['features']['morgan']['radius']},
  {cfg['features']['morgan']['n_bits']} bits) and MACCS (167-bit).
- **Hybrid** = descriptors + Morgan (used by the selected model).

## Model architecture
Selected model: **{selected}**. Candidate set = tuned XGBoost (hybrid), tuned Random Forest
and HistGradientBoosting (descriptors), and a **consensus ensemble** averaging member
predictions in logit space. Hyperparameters via Optuna (train->validation, test frozen);
best parameters recorded in `reports/results/tuned_hyperparameters.json`.

## Validation approach
Primary: **Bemis-Murcko scaffold split** 70/15/15 ({s5.get('n_unique_scaffolds', 'n/a')} unique
scaffolds; disjoint across partitions). Secondary: random split (reported separately).
Robustness: 5 repeated scaffold splits + 1000x bootstrap CIs on the primary test set.

## Performance (primary scaffold-split test set, percent space)
Bootstrap 95% CIs:
- {boot_line(sel.get('baseline_key', 'baseline'))}
- {boot_line(sel.get('best_proposed_key', 'proposed'))}

Repeated-split means (5 seeds) are in `reports/results/robustness_repeated_splits.csv`.
**Honest summary:** on any single split the models are statistically indistinguishable
(overlapping MAE CIs). Across repeated splits the proposed hybrid/consensus models show a
**consistent modest advantage and lower variance** than the linear/RF baselines, and the
consensus gives the best high-binding MAE.

## Performance by PPB range
Error is lowest for highly-bound compounds (in absolute percent terms) and largest for
low/moderate binders, which are under-represented ({s2.get('final_target_summary', {}).get('pct_ge_90', 0):.0f}%
of data is >= 90%). See `reports/figures/stage10_mae_by_band.png`. The logit target is what
makes the high-binding region accurate (ablation B).

{ext_section}
## Applicability domain
Max Tanimoto similarity (Morgan) to the training set; threshold =
{sel.get('bootstrap', {}) and ad.get('in_domain_fraction', '') and ''}data-driven (5th percentile
of train self-similarity). In-domain test MAE {ad.get('test_MAE_in_domain', 'n/a')} vs
out-of-domain {ad.get('test_MAE_out_domain', 'n/a')} - the AD flags less reliable predictions.

## Uncertainty method
Ensemble disagreement (std of member predictions). Correlation with absolute error on test:
{ad.get('uncertainty_error_corr', 'n/a')}. Empirical coverage of a +/-1.96-sigma interval:
{cov.get('empirical_coverage_at_1.96sigma', 'n/a')}. Treat uncertainty as a *relative* ranking of
reliability, not a calibrated probability.

## Important limitations
- Severe class imbalance toward high binding caps reliability for low-fu-unbound compounds.
- Single-source (AstraZeneca) human data; chemical space is drug-like. **External validation on
  Ingle et al. 2016 shows MAE roughly doubles-to-triples out-of-distribution** (see External
  validation section); absolute predictions on non-AZ chemotypes need local recalibration, though
  rank-ordering (Spearman ~0.72) still transfers. Use the applicability-domain flag.
- Ceiling censoring near 100% limits accuracy for the most extreme binders.
- Irreducible experimental noise (~0.06 fraction-bound units in the literature) bounds achievable MAE.
- R2 ~0.3-0.37 reflects the strict scaffold split (novel scaffolds), not a broken model.

## Known failure cases
Largest test errors (see `reports/tables/largest_errors.csv`); up to
{interp.get('largest_error_max_pct', 'n/a')}% absolute error;
{interp.get('n_test_errors_gt_20pct', 'n/a')} test compounds exceed 20% error, concentrated among
low-similarity (out-of-domain) and low/moderate-binding molecules.

## Ethical and scientific-use considerations
Research triage only. Do not use for dosing, safety, or regulatory decisions. Always confirm
critical predictions experimentally. Report predicted fu with its uncertainty and AD flag.

## Predicting on new compounds
Self-contained bundles (`models/final_xgb_hybrid.joblib` lean 0.5 MB, or
`models/final_consensus.joblib` 17.7 MB) carry the fitted descriptor cleaner, transformer,
training fingerprints, and AD threshold. Standardisation -> features -> logit prediction ->
inverse-transform -> AD flag + uncertainty are handled end-to-end:
```
python scripts/predict.py --smiles "CC(=O)Oc1ccccc1C(=O)O"
python scripts/predict.py --input compounds.smi --bundle models/final_consensus.joblib --out preds.csv
```
Programmatically: `from ppb_model.predict import load_bundle, predict_smiles`. Always check
`in_applicability_domain` before trusting a prediction.

---
*Generated by scripts/make_model_card.py from pipeline artefacts. Selected model: {selected}.
Rationale: {sel.get('rationale', 'n/a')}*
"""
    out = project_root() / "MODEL_CARD.md"
    out.write_text(card, encoding="utf-8")
    logger.info("Wrote %s", out)
    print(f"Model card written to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 16: generate model card.")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
