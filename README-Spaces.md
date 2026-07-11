---
title: PPB Predictor
emoji: 💊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# PPB Predictor

Predict human plasma protein binding (% bound) from a SMILES string, using an
XGBoost model trained on the AstraZeneca PPBR_AZ dataset and augmented with the
Ingle et al. (2016) set. Each prediction carries an **applicability-domain
flag** — always check it before trusting a value.

- Web UI: open the Space and paste one SMILES per line.
- JSON API: `POST /api/predict` with `{"smiles": ["CC(=O)Oc1ccccc1C(=O)O"]}`.
- Health check: `GET /api/health`.

## Data sources, licensing & attribution

The **code** is released under the MIT license (the `license: mit` tag above).
The **training/test data** carry their own licenses and are credited here:

- **PPBR_AZ** (training + test): an AstraZeneca ADME assay deposited in
  [ChEMBL](https://www.ebi.ac.uk/chembl/), licensed
  [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)
  (DOI `10.6019/CHEMBL3301361`), accessed via Therapeutics Data Commons.
  Attribution required; redistribution of the *data* would be ShareAlike.
- **Ingle et al. 2016** (augmentation): Ingle, Tornero-Velez, Nichols & Veber,
  *J. Chem. Inf. Model.* 2016, 56(11):2243–2252. Dataset published by the
  [US EPA](https://catalog.data.gov/dataset/qsars-for-plasma-protein-binding-source-data-and-predictions)
  (public domain).

This app ships only *derived* artifacts (a fitted model and Morgan fingerprints
for the applicability-domain check) and aggregate performance numbers — not the
raw datasets — so the practical obligation is attribution, satisfied above and in
the app footer.

> **Deploying this Space:** Hugging Face reads the YAML header above from a file
> named `README.md`. When you create the Space, copy this file to `README.md` in
> the Space repo (see `serve/DEPLOY.md`).
