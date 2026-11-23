# SCSE3040 MLOps Mini Project --- Group G172

## Project: MP255 --- OPD Waiting Time (Surat)

- **Client:** Surat District Hospital (Out-Patient Department)
- **Problem Statement:** Predict OPD waiting time (`wait_min`) from patient token to doctor consultation.
- **Evaluation Metric:** MAE (Mean Absolute Error) in minutes.
- **Faculty:** Dr. Richa Sharma
- **Batch:** 29

---

## Group Members

| Enrolment No. | Name | GitHub Username | Role |
| :--- | :--- | :--- | :--- |
| **S24CSEU2867** | Retiksingh | | Data Lead |
| **S24CSEU0606** | NishikaUpadhyay | nishika841 | Model Lead |
| **S24CSEU0571** | SupriyaSharma | | Service Lead |

---

## Repository Architecture

```
├── .dvc/                            # DVC initialization files
│   ├── config
│   └── .gitignore
├── .github/workflows/ci.yml         # CI pipeline running pytest on push
├── app/
│   └── main.py                      # FastAPI inference service (POST /predict, GET /health)
├── data/
│   ├── raw/
│   │   ├── train.csv.dvc            # DVC tracked raw training dataset
│   │   ├── later.csv.dvc            # DVC tracked raw later dataset
│   │   ├── holdout_features.csv     # Final holdout features
│   │   └── .gitignore
│   └── processed/                   # Cleaned datasets
├── k8s/
│   ├── deployment.yaml              # Kubernetes Deployment (mp255-api, 3 replicas)
│   └── service.yaml                 # Kubernetes Service (mp255-svc, port 80 -> 8000)
├── models/
│   └── model.joblib                 # Champion model artifact
├── predictions/
│   └── holdout_predictions.csv      # Output predictions on holdout_features.csv
├── reports/
│   └── findings.yaml                # Standard findings document
├── src/
│   └── mp255/
│       ├── __init__.py
│       ├── config.py                # Config loader reading params.yaml safely
│       └── data_check.py            # Code hunting and verifying the 7 data anomalies
├── tests/
│   ├── __init__.py
│   └── test_*.py                    # Unit and integration tests
├── .dockerignore
├── Dockerfile
├── findings.yaml                    # Submission findings
├── mlflow.db                        # Tracked MLflow SQLite database
├── params.yaml                      # Project parameters
├── README.md                        # Documentation
└── requirements.txt                 # Pinned dependencies
```

---

## Stage 1: Data Hunting & Anomaly Findings

Run the automated data check script:
```bash
python -m src.mp255.data_check
```

### Summary of Data Findings
1. **Duplicate Rows:** **Present** (52 duplicate rows detected in `train.csv`).
2. **Stuck Sensor:** **Present** (`avg_consult_min` stuck at `14.0` from `2026-05-03` to `2026-05-17`).
3. **Spelling Variants:** **Present** (`department` contains variants such as `'Pae.'`, `'Gen.'`, `' PAEDIATRICS'`; 3 true categories).
4. **Leaking Column:** **Absent** (No post-consultation columns present).
5. **Unit Mixup:** **Absent** (No unit discrepancies between wings).
6. **Sentinel Value:** **Absent** (No sentinel numbers like -1 or 999).
7. **Target Typos:** **Absent** (All `wait_min` targets fall within normal range 19–136 min).
