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
XGBoost model trained on the AstraZeneca PPBR_AZ dataset. Each prediction carries
an **applicability-domain flag** — always check it before trusting a value.

- Web UI: open the Space and paste one SMILES per line.
- JSON API: `POST /api/predict` with `{"smiles": ["CC(=O)Oc1ccccc1C(=O)O"]}`.
- Health check: `GET /api/health`.

> **Deploying this Space:** Hugging Face reads the YAML header above from a file
> named `README.md`. When you create the Space, copy this file to `README.md` in
> the Space repo (see `serve/DEPLOY.md`).
