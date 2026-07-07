# Deploying the PPB Predictor to Hugging Face Spaces (free)

The app is a Docker container that serves both the JSON API and the web page on
port **7860**. Hugging Face Spaces' **Docker SDK** runs it for free (16 GB RAM,
no credit card). It sleeps after ~48h idle and wakes on the next visit.

## What the Space needs
A Space is its own git repo. It must contain, at the repo root:

- `README.md` — **with the YAML header** (copy from `README-Spaces.md`; the
  `sdk: docker` / `app_port: 7860` fields configure the Space).
- `Dockerfile`
- `serve/` (app + static frontend + `requirements-serve.txt`)
- `src/` (the `ppb_model` library)
- `models/final_xgb_hybrid.joblib` (0.5 MB — commit directly, no Git LFS needed)

`.dockerignore` already trims everything else from the image build.

## One-time deploy

1. Create the Space: https://huggingface.co/new-space → pick **Docker → Blank**,
   set visibility to **Public** (so it works as a portfolio link).
2. Clone it and copy the files in:
   ```bash
   git clone https://huggingface.co/spaces/<user>/ppb-predictor
   cd ppb-predictor
   # from the project root, copy the needed pieces:
   cp -r ../Dockerfile ../serve ../src .
   mkdir -p models && cp ../models/final_xgb_hybrid.joblib models/
   cp ../README-Spaces.md README.md
   ```
3. Commit and push — the Space builds the image automatically:
   ```bash
   git add -A && git commit -m "Deploy PPB predictor" && git push
   ```
4. Watch the **Logs** tab until the build finishes, then open the Space URL.

## Smoke test the live Space
```bash
curl https://<user>-ppb-predictor.hf.space/api/health
curl -X POST https://<user>-ppb-predictor.hf.space/api/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles": ["CC(=O)Oc1ccccc1C(=O)O"]}'
```

## Updating later
Push new commits to the Space repo; it rebuilds on every push. To swap in the
consensus bundle instead, change `PPB_BUNDLE` in the `Dockerfile` and copy that
`.joblib` in (note it is ~18 MB).
