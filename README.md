# Athlete Injury Risk Prediction Platform

Production-oriented ML pipeline for predicting athlete injury risk probability from multimodal sports datasets. This is a risk screening system, not a diagnosis system.

## What It Builds

- Unified ingestion for `data/multimodal`, `data/sirp600`, `data/runners`, and `data/optional`
- Leakage-aware temporal feature engineering
- Robust preprocessing with imputation, outlier clipping, categorical encoding, and scaling
- Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost model comparison
- Weighted ensemble, Platt scaling, isotonic calibration, threshold optimization
- Precision-first reports, calibration curves, PR curves, confusion matrices, and SHAP plots
- FastAPI inference service with single and batch prediction endpoints

## Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt
```

## Train

```powershell
python -m src.training.train --config config.yaml
```

Artifacts are written to `models/` and reports to `outputs/`.

`config.yaml` enables GPU training for XGBoost, LightGBM, and CatBoost with `training.use_gpu: true`. Scikit-learn preprocessing, Logistic Regression, Random Forest, calibration, and report generation still run on CPU.

Monitor GPU usage while training:

```powershell
nvidia-smi -l 1
```

## Serve

```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Health check:

```powershell
curl http://localhost:8000/health
```

## Deploy To Vercel

The repository includes a Vercel entrypoint at `api/index.py`. Vercel installs the lean runtime dependencies from `requirements.txt`; full training dependencies are kept in `requirements-train.txt`.

```powershell
npm i -g vercel
vercel login
vercel
```

For production:

```powershell
vercel --prod
```

The deployment serves:

- `/` frontend
- `/health`
- `/predict`
- `/batch_predict`

The trained artifact `models/injury_risk_pipeline.joblib` must be present before deploying.

## Diagnosis Engine

The diagnosis subsystem is a sports injury decision-support engine. It estimates likely injuries from questionnaire answers, symptom matching, rule tracing, and optional injury-risk model output. It never forces a diagnosis: responses below the confidence threshold include an insufficient-confidence message.

Endpoints:

- `POST /diagnose`
- `POST /confirm-diagnosis`
- `POST /feedback`
- `GET /assessment/{id}`

Example:

```powershell
curl -X POST http://localhost:8000/diagnose -H "Content-Type: application/json" -d "@sample_requests/diagnose.json"
```

Continuous learning:

- Every assessment can be stored.
- Confirmed diagnoses are stored as future labels.
- Low-confidence cases are placed in the review queue.
- `src/retraining/pipeline.py` can build a labeled dataset and version future models under `models/diagnosis/v*/`.

For production PostgreSQL storage, set:

```powershell
$env:DATABASE_URL="postgresql+psycopg2://user:password@host:5432/dbname"
```

Prediction:

```powershell
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@sample_requests/predict.json"
```

## Precision-First Behavior

The trainer selects models using validation precision, false positive rate, and calibration quality. The deployed decision threshold is optimized separately from the model probability so the system can reduce false positives without distorting calibrated probabilities.

## Outputs

- `outputs/model_comparison.csv`
- `outputs/metrics_report.json`
- `outputs/feature_importance.csv`
- `outputs/confusion_matrix.png`
- `outputs/precision_recall_curve.png`
- `outputs/calibration_curve.png`
- `outputs/shap_global_importance.png`
- `models/injury_risk_pipeline.joblib`

## Notes

- Any positive target class is treated as risk-positive because this system predicts injury risk probability, not injury type.
- Temporal feature windows use shifted historical values within athlete groups to avoid current-row leakage.
- Optional dependencies such as XGBoost, LightGBM, CatBoost, Optuna, and MLflow are used when installed.
