# Titanic Survival Prediction API — End-to-End ML Deployment

**Week 4 Task (Capstone): AI Project Deployment & Model Serving** (Internship project by Rabi Narayan Patra)

The final project of the four-week series: the Titanic survival model — built through
[Week 1 preprocessing](https://github.com/rabinarayanpatra350-biotech/titanic-data-preprocessing),
the [Week 2 model comparison](https://github.com/rabinarayanpatra350-biotech/titanic-supervised-models), and
the [Week 3 evaluation](https://github.com/rabinarayanpatra350-biotech/titanic-clustering-evaluation) —
is **serialized with joblib and served as a live prediction API by Flask**.

## What this system does

| Program | Responsibility |
|---------|----------------|
| `train_and_serialize.py` | Preprocess → train → evaluate → **serialize to `model.pkl`** (joblib) |
| `app.py` | Load `model.pkl` at startup and **serve predictions** as a Flask JSON API |
| `test_api.py` | **End-to-end tests** of every endpoint (Flask test client) |

## Quickstart

```bash
pip install -r requirements.txt
python train_and_serialize.py    # builds model.pkl (once)
python app.py                   # API live at http://127.0.0.1:5000
```

Then predict a passenger:

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"pclass": 3, "sex": "male", "age": 22, "sibsp": 1,
          "parch": 0, "fare": 7.25, "embarked": "S", "title": "Mr"}'
```

Response (real output from this system):

```json
{
  "survival_prediction": 0,
  "survival_probability": 0.076,
  "verdict": "does not survive",
  "model": "LogisticRegression (scikit-learn)"
}
```

## API endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | documentation + usage example |
| `/health` | GET | status + model metadata (for monitoring) |
| `/predict` | POST | one passenger → prediction, probability, verdict |

Invalid bodies return **HTTP 400** with a helpful message; optional fields default to sensible values.

## Results

**Deployed model (held-out test set):** accuracy 84.9%, precision 0.828, recall 0.768, F1 0.797, ROC-AUC 0.876.

**Live API tests** (`python test_api.py` — all passed):

| Passenger | Survival probability | Prediction |
|-----------|---------------------|------------|
| 3rd-class man, 22, £7.25 ticket | 0.076 | does not survive |
| 1st-class woman, 38, £71.28 ticket | 0.960 | survives |
| Boy aged 7 travelling with family | 0.553 | survives |
| Invalid request body | HTTP 400 | graceful error |

## Design notes

- **Feature parity**: the client sends raw, human-friendly fields; the API applies
  *exactly* the training-time transformations (FamilySize, IsAlone, one-hot encoding,
  the serialized fare scaler) — the classic training/serving skew failure mode is designed out.
- **Serialization as a bundle**: `model.pkl` (2.2 KB) carries the model, the scaler,
  the ordered feature list, and the metrics, and is round-trip verified on save.
- **Interpretable model**: the coefficients show what drives survival — `Mr` (−1.34),
  male sex (−1.29), 3rd class (−1.14) lower it; `Master` (+1.20), `Mrs` (+0.72) raise it.

## Repository contents

| Where | File | What it is |
|-------|------|------------|
| repo | `train_and_serialize.py` | Training + evaluation + joblib serialization |
| repo | `app.py` | The Flask prediction API |
| repo | `test_api.py` | Automated endpoint tests |
| repo | `train_log.txt`, `api_test_log.txt` | Real run logs |
| repo | `requirements.txt`, `README.md` | Setup and documentation |
| [Release v1.0](https://github.com/rabinarayanpatra350-biotech/titanic-ml-api/releases/tag/v1.0) | `model.pkl` | The serialized trained model (ready to serve) |
| [Release v1.0](https://github.com/rabinarayanpatra350-biotech/titanic-ml-api/releases/tag/v1.0) | `titanic.csv` | Raw input dataset |
| [Release v1.0](https://github.com/rabinarayanpatra350-biotech/titanic-ml-api/releases/tag/v1.0) | `Capstone_Deployment_Report.pdf` | Full written report (Week 4 deliverable) |
| [Release v1.0](https://github.com/rabinarayanpatra350-biotech/titanic-ml-api/releases/tag/v1.0) | `Capstone_Presentation.pptx` | The project presentation |
| [Release v1.0](https://github.com/rabinarayanpatra350-biotech/titanic-ml-api/releases/tag/v1.0) | `fig_*.png`, `model_metrics.csv` | Generated figures and metrics |

## Purpose

Educational internship submission (Week 4 capstone) demonstrating an end-to-end machine-learning
application: data preprocessing, model training, testing, **model serialization with joblib**,
and **deployment of a simple prediction API with Flask**, with full documentation and presentation.
