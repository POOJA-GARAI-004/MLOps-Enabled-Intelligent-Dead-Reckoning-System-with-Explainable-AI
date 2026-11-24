
# MLOps fast start — Vfa01

Open a terminal in this project root.

## Install
```bash
pip install -r requirements-mlops.txt
```

## Align the two CSVs (run once)
```bash
python -m mlops.align_vfa01
```

## Generate all four Vfa01 explainability/monitoring outputs
```bash
python -m mlops.explain_shap --output-index 0 --samples 30
python -m mlops.explain_lime --output-index 0
python -m mlops.explain_integrated_gradients --output-index 0
python -m mlops.monitor_drift
```

## MLflow experiment and registry
```bash
python -m mlops.log_existing_model
```
Copy the printed Model URI. In a second terminal:
```bash
mlflow ui --host 127.0.0.1 --port 5000
```
Open http://127.0.0.1:5000. Then register:
```bash
python -m mlops.register_model --model-uri "runs:/YOUR_RUN_ID/model" --name IntelligentDeadReckoning
```

## GitHub Actions
Push the project to GitHub and open the Actions tab. Workflow:
`.github/workflows/mlops-ci.yml`

## Output paths
- `results/mlops/Vfa01_alignment_report.json`
- `results/mlops/shap_vfa01_output_0.png`
- `results/mlops/lime_vfa01_output_0.png`
- `results/mlops/integrated_gradients_vfa01_output_0.png`
- `results/mlops/evidently_vfa01_early_vs_late.html`

## Verified locally
- Vfa01 timestamps aligned with no extrapolation.
- Repository preprocessing prepared 1,053 model windows.
- Existing checkpoint loaded successfully.
- Model produced finite output with shape `(4, 11)` on a four-window smoke test.
- All newly added Python files passed compile checks.

Note: SHAP/LIME background uses early Vfa01 windows because the raw Vta2 training CSVs
are not present in the supplied ZIP. Evidently compares early and late Vfa01 portions.
Do not describe these as training-vs-production drift.
