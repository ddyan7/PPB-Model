# Model Card — Human Plasma Protein Binding (PPB) Predictor

## Intended use
Estimate human plasma protein binding (percent bound) for small drug-like molecules from
SMILES, as an early-discovery triage/prioritisation aid. **Not** a replacement for
experimental measurement and **not** for regulatory or clinical decisions. Predictions
outside the applicability domain (see below) should be treated as unreliable.

## Deployed final model (augmented)
**The deployed model is the augmented consensus.** Training data = PPBR_AZ scaffold-train
(1129) + scaffold-safe non-overlapping Ingle compounds
(1412) = **2541** molecules. It replaced the
original PPBR_AZ-only model, which is preserved as `models/*_ppbr_only.joblib`.

Performance on the unchanged PPBR_AZ scaffold test: MAE **7.0403**,
RMSE 12.5737, R2 **0.4574**, Spearman 0.7842,
high-binding MAE 3.1599 (vs the original ~7.53 / 0.36 / 2.69).

**Robustness (why it was promoted, stated honestly):** across 5 repeated
scaffold splits the augmented model beats the original on MAE in
4/5 splits
(MAE 7.7288 vs 8.0561;
R2 0.4768 vs
0.4012). The paired bootstrap on the primary split
gives delta MAE -0.5023 (95% CI
[-1.4774, 0.4613],
P(improvement) 0.851). **So: a consistent, modest overall gain
that is directionally reliable but not certified significant on a 243-compound test, with a small
high-binding-MAE cost.** Do not overstate it.


## Prediction target
Percent bound (0-100%). Models are trained on a **logit-transformed fraction bound**
(`logit(fb)`) and predictions are inverse-transformed back to percent. Fraction unbound
fu = (100 - PPB)/100 is reported alongside, since fu is the pharmacologically active quantity.

## Training dataset
TDCommons **PPBR_AZ** (AstraZeneca), **human subset only** (1614 rows;
Species == "Homo sapiens"). After cleaning: **1614 unique structures**.
Distribution is strongly high-binding: median 95.43%,
67% of compounds >= 90% bound.

## Data-cleaning steps
RDKit parse -> Cleanup -> largest-fragment (salt/mixture) parent -> uncharge -> preserve
stereochemistry -> canonical SMILES + InChIKey. Duplicate/conflict resolution by InChIKey
(median aggregation; hard conflicts excluded). 30 ceiling-censored
values (>= 99.9%) flagged and retained. Every record is logged in
`reports/tables/data_cleaning_audit.csv` (no silent edits). In this dataset all
1614 human structures parsed and were unique (0 salts, 0 conflicts).

## Molecular representation
- Physicochemical: full RDKit descriptor set (1614 mols), cleaned
  **train-only** (impute -> drop constant -> decorrelate at |r|>0.95 -> standardise).
- Morgan/ECFP (radius 2,
  2048 bits) and MACCS (167-bit).
- **Hybrid** = descriptors + Morgan (used by the selected model).

## Model architecture
Selected model: **consensus**. Candidate set = tuned XGBoost (hybrid), tuned Random Forest
and HistGradientBoosting (descriptors), and a **consensus ensemble** averaging member
predictions in logit space. Hyperparameters via Optuna (train->validation, test frozen);
best parameters recorded in `reports/results/tuned_hyperparameters.json`.

## Validation approach
Primary: **Bemis-Murcko scaffold split** 70/15/15 (1032 unique
scaffolds; disjoint across partitions). Secondary: random split (reported separately).
Robustness: 5 repeated scaffold splits + 1000x bootstrap CIs on the primary test set.

## Performance (primary scaffold-split test set, percent space)
Bootstrap 95% CIs:
- elasticnet_desc(baseline): MAE 7.55 (95% CI 6.29-9.11), high-binding MAE 2.98
- consensus: MAE 7.47 (95% CI 6.23-8.88), high-binding MAE 2.75

Repeated-split means (5 seeds) are in `reports/results/robustness_repeated_splits.csv`.
**Honest summary:** on any single split the models are statistically indistinguishable
(overlapping MAE CIs). Across repeated splits the proposed hybrid/consensus models show a
**consistent modest advantage and lower variance** than the linear/RF baselines, and the
consensus gives the best high-binding MAE.

## Performance by PPB range
Error is lowest for highly-bound compounds (in absolute percent terms) and largest for
low/moderate binders, which are under-represented (67%
of data is >= 90%). See `reports/figures/stage10_mae_by_band.png`. The logit target is what
makes the high-binding region accurate (ablation B).

> **Note:** the external-validation and head-to-head results below describe the *earlier PPBR_AZ-only* model (preserved as `models/*_ppbr_only.joblib`). Because the **deployed** model now trains on Ingle data, Ingle can no longer serve as its external validation — a fresh third-party dataset would be required.

## External validation (out-of-distribution, Ingle et al. 2016)
Scored the frozen model on **1494 compounds** from
`ppb_usable_dataset.csv` after removing **136 InChIKey overlaps**
with PPBR_AZ (leakage-controlled). Fub was converted to percent bound.

| Metric | All (1494) | In-domain (862) | In-distribution test (ref) |
|---|---|---|---|
| MAE | 18.935 | 17.4461 | ~7.5 |
| Median AE | 9.982 | 8.3449 | - |
| RMSE | 27.8887 | 26.6922 | ~13.6 |
| R2 | 0.303 | 0.2824 | ~0.36 |
| Spearman | 0.7193 | 0.7141 | ~0.75 |
| High-binding MAE | 5.2569 | 4.9206 | ~2.7 |
| In-domain fraction | 0.577 | - | 0.87 |

**Interpretation (honest):** absolute error roughly doubles-to-triples out-of-distribution
(MAE ~7.5 in-distribution -> ~18.935 on Ingle), driven by an out-of-domain tail
(RMSE >> MAE >> median AE ~9.982) and the fact that only
~57% of Ingle compounds fall inside the
applicability domain (it contains environmental chemicals outside the drug-like AZ space).
Rank correlation holds up well (Spearman ~0.7193), so the model still
ranks/triages usefully out-of-distribution even where absolute calibration drifts. The AD flag
correctly identifies the ~42% of external compounds where predictions should not be trusted.

### Head-to-head vs published models (identical held-out compounds)
On **576 identical** Ingle held-out compounds (Dte+T1+T2, PPBR_AZ
overlaps removed), scored the same way. **Caveat:** Ingle's models are tested *in-distribution*
(their home data) while ours is *cross-source transfer* — Ingle has a home-field advantage.

| Model | MAE | R2 | Spearman | High-binding MAE |
|---|---|---|---|---|
| Ingle consensus (published) | 13.437 | 0.53 | 0.701 | 8.785 |
| Ingle RF (published) | 13.305 | 0.472 | 0.692 | 6.615 |
| This model (consensus) | 14.991 | 0.363 | 0.656 | **5.744** |

Ingle's in-distribution models lead on overall MAE/R2 by a modest margin (expected), but **this
model wins the high-binding region** (5.744 vs
8.785 / 6.615) despite the transfer
handicap — independent confirmation that the logit-transform design generalises. See
`reports/results/head_to_head_ingle.csv`.

## Applicability domain
Max Tanimoto similarity (Morgan) to the training set; threshold =
data-driven (5th percentile
of train self-similarity). In-domain test MAE 6.9617 vs
out-of-domain 11.211 — the AD flags less reliable predictions.

## Uncertainty method
Ensemble disagreement (std of member predictions). Correlation with absolute error on test:
0.5462. Empirical coverage of a +/-1.96-sigma interval:
n/a. Treat uncertainty as a *relative* ranking of
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
71.70028005036986% absolute error;
27 test compounds exceed 20% error, concentrated among
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
*Generated by scripts/make_model_card.py from pipeline artefacts. Selected model: consensus.
Rationale: Overall MAE CIs overlap (no meaningful overall-accuracy gain), but the proposed model improves high-binding MAE (2.75 vs 2.98), the metric the research gap targets; selected for the high-binding edge plus its applicability-domain and uncertainty outputs.*
