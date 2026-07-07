# PPB Prediction — Modelling Plan (Stage 1)

**Dataset:** TDCommons PPBR_AZ, human-only subset (`Species == "Homo sapiens"`, n = 1,614)
**Target column:** `Y` = plasma-protein-binding **percent bound** (range in data: 11.18–99.95)
**Environment:** Python 3.12, venv at `C:\Users\dandan\.venvs\ppb-model` (outside Google Drive)
**Status:** Plan documented *before* any model training, per project brief.

This plan operationalises the completed PPB literature review. It does not repeat the review; it
translates its findings into concrete, testable modelling decisions.

---

## 0. Verified dataset facts (recomputed from `data/raw/ppbr_az.csv`, not assumed)

| Fact | Value |
|---|---|
| Total rows / species | 2,828 across 5 species (human 1,614, rat 717, dog 244, mouse 162, guinea pig 91) |
| Human n | 1,614 |
| Human target mean / median / std | 88.07 / 95.43 / 16.73 (% bound) |
| Skew: ≥80% / ≥90% / ≥95% / ≥99% | 80.2% / 66.9% / 52.1% / 18.5% |
| Compounds < 50% bound | 84 (5.2%) |
| Raw duplicate SMILES within human | 0 (canonical/parent dedup still required — Stage 2) |
| SMILES measured in >1 species | 701 (cross-species leakage risk if species mixed) |
| Values ≥ 99.9% | 30 (ceiling-censoring suspected) |

**Implication:** the dataset is severely right-skewed toward high binding. The scientifically important
and hardest region — fraction unbound near zero — is exactly where the review reports current models fail.

---

## 1. Primary research question
Does target transformation (logit/lnKa vs. untransformed %), combined with the best-performing molecular
representation, **meaningfully improve human PPB prediction on PPBR_AZ under a scaffold split** — in
particular for highly bound compounds (≥90%) — relative to a strong untransformed descriptor baseline?

## 2. Primary hypothesis
A **logit-transformed target** paired with a **hybrid (physicochemical descriptors ⊕ Morgan fingerprints)**
gradient-boosting model will reduce **high-binding-subset MAE** (and MAE in fraction-unbound space) without
degrading overall MAE, because the logit transform redistributes error into the fu,p-sensitive region.
*Supported by:* Han et al. 2025 (LogIt consensus, decisive for PPB>99%), Watanabe et al. 2018 (log-fu improved
low-fu accuracy); consensus/ensemble superiority (Yuan 2020, Sun 2018, Han 2025).

## 3. Primary evaluation metric
**MAE on PPB % on the scaffold-split test set.** Headline and directly comparable to the literature.

## 4. Secondary / decisive metrics
- **High-binding MAE (PPB ≥ 90%)** — the region the research gap targets (decisive co-metric).
- **MAE in fraction-unbound space** — free-drug-relevant; small % error at high binding = large relative fu error.
- RMSE, R², Spearman, Pearson, median-AE (overall and per binding band: <50, 50–80, 80–90, 90–95, 95–99, ≥99).
- Prediction-interval coverage & width (Stage 14).
- Robustness: mean ± bootstrap CI across repeated scaffold splits.

> "Improvement" is claimed only if the proposed model beats the best baseline on the **frozen test set** on the
> primary metric OR on high-binding/fu-space MAE **without materially worsening** overall MAE — judged against
> bootstrap CIs, not point estimates.

## 5. Data-cleaning strategy (Stage 2)
Filter human → RDKit parse → salt-strip to parent → neutralise unambiguous charges → preserve stereo →
canonical SMILES + InChIKey → detect canonical duplicates and intra-structure conflicting measurements →
flag censored/limit values. **Every removed/modified/aggregated record is logged in an audit CSV. No silent edits.**

## 6. Target-transformation options compared (Stage 3)
`percent` · `fraction_bound` · `fraction_unbound` · `log(fu)` · **`logit(fb)`** · `lnKa = 0.5·ln(fb/fu)` · `clipped`.
Selection criteria: numerical stability, scientific meaning, residual distribution, **high-binding performance**,
interpretability, literature consistency — never a single metric. Boundary guard: clip to [0.1%, 99.9%], ε = 0.001.

## 7. Molecular representations compared (Stage 6)
- RDKit **physicochemical descriptors** (train-only variance filter, correlation prune, impute, scale).
- **Morgan** fingerprints (radius 2/3, 1024/2048 bits — tuned/ablated).
- **MACCS** keys.
- **Hybrid** (descriptors ⊕ Morgan) — the proposed representation.
- GNN / pretrained embeddings: **not planned** — n=1,614 is small; the review shows no clear GNN advantage at
  this scale and warns against complexity-for-its-own-sake. Will revisit only if baselines plateau with evidence.

## 8. Baseline models (Stage 7)
1. **Median predictor** (non-informative floor).
2. **Ridge / ElasticNet** on descriptors (linear reference).
3. **Random Forest** (descriptors).
4. **HistGradientBoosting / XGBoost** (strong conventional).
5. **Fingerprint RF** (Morgan-based).
All baselines use identical splits and the same evaluation harness.

## 9. Proposed improved model (Stage 8)
**Hybrid-representation gradient boosting trained on the logit target**, wrapped with an
**applicability-domain + ensemble-uncertainty** layer; optional high-binding-weighted-loss / two-stage variant.
- *Problem it solves:* poor accuracy + no reliability signal in the high-binding region.
- *Why baselines may not:* untransformed % under-weights fu-sensitive errors; single models give no AD/uncertainty.
- *Evidence:* Han 2025, Watanabe 2018 (transformation); Sun 2018, Han 2025 (AD halves out-of-domain error).
- *Counts as improvement:* lower high-binding / fu-space MAE at equal-or-better overall MAE, CI-supported.
- *Counts as no help:* overlapping CIs vs. best baseline → report honestly and keep the simpler model.

## 10. Validation & statistical comparison (Stages 4–5, 11)
**Primary:** Bemis–Murcko **scaffold split 70/15/15**, disjoint scaffolds, saved assignments, fixed seed.
**Secondary:** random split (reported separately, never mixed). Repeated scaffold splits (n=5) + bootstrap CIs.
Test set frozen until final model + hyperparameters are locked.

## 11. Ablation studies (Stage 12)
descriptors-only vs FP-only vs hybrid · raw vs logit target · single vs ensemble · scaffold vs random split ·
with/without AD filtering · standard vs high-binding-weighted loss · Morgan radius/bit-size.

## 12. Risks & limitations
Extreme skew caps low-binding reliability; ceiling censoring near 100%; small n discourages deep models;
irreducible measurement noise (~0.06 fb units, Sun 2018) sets an error floor; logit ε handling at boundaries;
Google-Drive sync of large artefacts (mitigated: venv/outputs kept lean, venv outside Drive).

---

## Leakage-prevention checklist (Stages 4)
- [ ] Human-only scope avoids the 701 cross-species shared SMILES.
- [ ] Salt-strip to parent before dedup and before splitting.
- [ ] Disjoint Bemis–Murcko scaffolds across train/val/test.
- [ ] Conflicting replicate measurements resolved before splitting.
- [ ] Scaling / imputation / feature-selection fit on **train only**.
- [ ] Hyperparameters tuned on train+val (CV); test untouched.
- [ ] `ppb_usable` (Ingle) checked for InChIKey overlap before any external-validation use.
