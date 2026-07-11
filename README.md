# PPB Prediction — Human Plasma Protein Binding on TDCommons PPBR_AZ

Reproducible machine-learning pipeline for predicting small-molecule human plasma protein binding (PPB)
from SMILES, built on the AstraZeneca PPBR_AZ dataset. Modelling decisions are driven by a completed
literature review (see `MODELLING_PLAN.md`).

## Research gap
Systematic evaluation of **target transformation** (untransformed % vs. logit/lnKa) × **molecular
representation** (descriptors vs. Morgan vs. hybrid) for human PPB on PPBR_AZ, focused on the highly bound
(≥90%) region where the free-drug-relevant fraction unbound is hardest to predict.

## Environment (Python 3.12)
RDKit has no PyPI wheels for Python 3.14, so this project pins **Python 3.12**. The venv lives **outside**
Google Drive to avoid sync churn.

```powershell
py -3.12 -m venv C:\Users\dandan\.venvs\ppb-model
C:\Users\dandan\.venvs\ppb-model\Scripts\python -m pip install -r requirements.txt
```

Install ppb_model as editable install
```powershell
C:\Users\dandan\.venvs\ppb-model\Scripts\python -m pip install -e "C:\Users\dandan\My Drive\Portfolio\Projects\PPB-Model"
```

Run anything with that interpreter, e.g.:
```powershell
C:\Users\dandan\.venvs\ppb-model\Scripts\python scripts\prepare_data.py --config configs\default.yaml
```

## Layout
```
PPB-Model/
├── MODELLING_PLAN.md        # Stage 1: the experimental plan (read this first)
├── configs/default.yaml     # all pipeline settings
├── data/{raw,interim,processed,splits}/
├── src/ppb_model/           # reusable library code
├── scripts/                 # stage entry points
├── reports/{figures,tables,results}/
├── models/                  # trained pipelines
├── tests/                   # unit tests
└── logs/
```

## Pipeline stages
1. Plan (`MODELLING_PLAN.md`) · 2. Clean + audit · 3. Target analysis · 4–5. Leakage-safe splits ·
6. Features · 7. Baselines · 8. Improved model · 9. Tuning · 10–12. Eval/robustness/ablation ·
13. Interpretability · 14. Applicability domain / uncertainty · 15. Selection · 16. Outputs + model card.

## Running the full pipeline (in order)


```powershell
$py = "C:\Users\dandan\.venvs\ppb-model\Scripts\python"
& $py scripts\prepare_data.py          --config configs\default.yaml   # Stage 2
& $py scripts\analyze_target.py        --config configs\default.yaml   # Stage 3
& $py scripts\create_splits.py         --config configs\default.yaml   # Stages 4-5
& $py scripts\build_features.py        --config configs\default.yaml   # Stage 6
& $py scripts\train_baselines.py       --config configs\default.yaml   # Stage 7
& $py scripts\train_improved_model.py  --config configs\default.yaml   # Stages 8-9
& $py scripts\robustness.py            --config configs\default.yaml   # Stage 11
& $py scripts\ablation.py              --config configs\default.yaml   # Stage 12
& $py scripts\interpret.py             --config configs\default.yaml   # Stage 13
& $py scripts\make_figures.py          --config configs\default.yaml   # Stages 10, 14
& $py scripts\evaluate_models.py       --config configs\default.yaml   # Stages 15-16
& $py scripts\make_model_card.py       --config configs\default.yaml   # Stage 16
& $py -m pytest tests\ -q                                              # unit tests (24)
```

### Optional: external validation + the improvement experiment
```powershell
& $py scripts\external_validation.py  --config configs\default.yaml   # score frozen model on Ingle (leakage-controlled)
& $py scripts\head_to_head.py         --config configs\default.yaml   # vs Ingle's published models, same held-out data
& $py scripts\train_augmented.py      --config configs\default.yaml   # augment train w/ Ingle + grouped-CV tuning
& $py scripts\confirm_robustness.py   --config configs\default.yaml   # repeated splits + paired bootstrap
& $py scripts\promote_augmented.py    --config configs\default.yaml   # promote augmented model to deployed bundles
& $py scripts\make_model_card.py      --config configs\default.yaml   # regenerate card with augmented + robustness
```
The **deployed** model is the augmented consensus (PPBR_AZ train + scaffold-safe Ingle); the
original PPBR_AZ-only bundles are preserved as `models/*_ppbr_only.joblib`. Note: once Ingle is
folded into training, it can no longer serve as external validation for the deployed model.

## Headline results (scaffold split, frozen test set)
- **Target transformation is the decisive lever:** logit vs raw target roughly halves
  high-binding (≥90%) MAE (≈6.4 → ≈2.8) — the core research-gap finding, isolated in ablation B.
- **Descriptors ≫ fingerprints** on novel scaffolds (Morgan/MACCS R²≈0); MolLogP dominates
  importance, matching the PPB literature.
- **Proposed hybrid/consensus ≈ tuned baseline on overall MAE** (bootstrap CIs overlap) but with
  a consistent edge in high-binding MAE and lower variance across 5 repeated splits — honest,
  modest improvement, not a complexity win.
- **Reliability layer works:** in-domain test MAE 6.96 vs out-of-domain 11.21; split-conformal
  90% interval achieves ~93% empirical coverage. Selected model: consensus (see `MODEL_CARD.md`).

## Predicting on new compounds
Two self-contained bundles carry the fitted cleaner, transformer, training fingerprints, and
AD threshold, so they predict from raw SMILES end-to-end:
- `models/final_xgb_hybrid.joblib` — lean single model (**0.5 MB**)
- `models/final_consensus.joblib` — full ensemble with uncertainty (**17.7 MB**, compressed from 61 MB)

```powershell
& $py scripts\predict.py --smiles "CC(=O)Oc1ccccc1C(=O)O"
& $py scripts\predict.py --input compounds.smi --bundle models\final_consensus.joblib --out preds.csv
```
Rebuild bundles without re-tuning: `& $py scripts\finalize_models.py --config configs\default.yaml`.
Always check the `in_applicability_domain` flag before trusting a prediction.

## Reproducibility
Fixed seeds (`configs/default.yaml`), train-only preprocessing, frozen test set, saved split assignments,
and a single experiment-results table (`reports/results/`) in CSV/MD/JSON.

## Data sources, licensing & attribution
The **code** in this repository is released under the **MIT** license. The
**datasets** carry their own licenses and must be attributed:

- **PPBR_AZ** (training + test) — an AstraZeneca ADME assay deposited in
  [ChEMBL](https://www.ebi.ac.uk/chembl/), licensed **CC BY-SA 3.0**
  (DOI `10.6019/CHEMBL3301361`), accessed via
  [Therapeutics Data Commons](https://tdcommons.ai/single_pred_tasks/adme/).
  Attribution required; redistributing the *data* triggers ShareAlike.
- **Ingle et al. 2016** (augmentation/validation) — Ingle, Tornero-Velez,
  Nichols & Veber, *J. Chem. Inf. Model.* 2016, 56(11):2243–2252; dataset
  published by the [US EPA](https://catalog.data.gov/dataset/qsars-for-plasma-protein-binding-source-data-and-predictions)
  (public domain).

The shipped model bundles and the web app contain only *derived* artifacts
(fitted models, Morgan fingerprints for the applicability-domain check) and
aggregate metrics — not the raw datasets — so the practical obligation is
attribution, not ShareAlike redistribution.

## To run Web Server and FASTAPI
```powershell
cd serve
$py -m uvicorn app:app --port 7860
```
