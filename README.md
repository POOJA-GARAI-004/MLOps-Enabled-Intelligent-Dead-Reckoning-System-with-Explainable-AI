# MLOps-Enabled Intelligent Dead Reckoning System with Explainable AI

An Intelligent Dead Reckoning (IDR) navigation system enhanced with Machine Learning Operations (MLOps), Explainable Artificial Intelligence (XAI), experiment tracking, model versioning, and data drift monitoring.

## 1. Project Overview

The **MLOps-Enabled Intelligent Dead Reckoning System** aims to improve smartphone-based navigation in GNSS-denied environments using inertial sensor data and a machine-learning-assisted navigation pipeline.

The system uses smartphone IMU measurements, including:

- Accelerometer (X, Y, Z)
- Gyroscope (X, Y, Z)

A Dilated Convolutional Neural Network (CNN) adapter learns to estimate 11 adaptation parameters that support the navigation pipeline's Invariant Extended Kalman Filter (IEKF).

The project integrates MLOps practices to improve model experiment management, reproducibility, explainability, version control, and monitoring.

## 2. Key Features

### Intelligent Dead Reckoning
- Smartphone IMU-based navigation.
- Inertial sensor preprocessing.
- CNN-based adaptive parameter estimation.
- IEKF-based navigation pipeline.
- Navigation in GNSS-denied environments.

### MLOps Integration
- MLflow experiment tracking.
- MLflow model registration and version management.
- Model artifact logging.
- Automated code checks using GitHub Actions.
- Data drift monitoring using Evidently.
- Model documentation using a Model Card.

### Explainable AI (XAI)
- SHAP: Global feature contribution analysis.
- LIME: Local model prediction explanations.
- Integrated Gradients: Time-series sensor attribution analysis.

These explainability methods help understand how sensor measurements influence the CNN adapter's outputs.

## 3. System Architecture

The project follows this general workflow:

1. Smartphone IMU data collection.
2. Data preprocessing and alignment.
3. Sensor window generation.
4. CNN adapter inference.
5. Adaptive parameter estimation.
6. IEKF-based navigation.
7. Model evaluation and analysis.
8. Explainability and monitoring.
9. MLflow experiment and model management.

## 4. Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Machine Learning | Scikit-learn |
| Navigation | Invariant Extended Kalman Filter (IEKF) |
| Experiment Tracking | MLflow |
| Explainable AI | SHAP, LIME, Captum |
| Drift Monitoring | Evidently |
| Data Processing | NumPy, Pandas, SciPy |
| Visualization | Matplotlib |
| Version Control | Git, GitHub |
| CI Automation | GitHub Actions |

## 5. Project Structure

```text
Intelligent-Dead-Reckoning/
│
├── .github/
│   └── workflows/
│       └── mlops-ci.yml
│
├── android/
│
├── configs/
│   └── adapter_cnn.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   ├── model_card_mlops.md
│   └── MLOPS_VFA01_PROCESS.md
│
├── mlops/
│   ├── align_vfa01.py
│   ├── vfa01_data.py
│   ├── explain_shap.py
│   ├── explain_lime.py
│   ├── explain_integrated_gradients.py
│   ├── monitor_drift.py
│   ├── log_existing_model.py
│   └── register_model.py
│
├── notebooks/
├── results/
│   ├── models/
│   └── mlops/
│
├── scripts/
├── src/
├── team/
├── tests/
│
├── train.py
├── requirements.txt
├── requirements-mlops.txt
├── pytest.ini
├── MLOPS_FAST_START.md
└── README.md
```

## 6. Installation and Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/POOJA-GARAI-004/MLOps-Enabled-Intelligent-Dead-Reckoning-System-with-Explainable-AI.git
```

### Step 2: Navigate to the project

```bash
cd MLOps-Enabled-Intelligent-Dead-Reckoning-System-with-Explainable-AI
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

Activate on Windows:

```powershell
venv\Scripts\activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-mlops.txt
```

## 7. MLOps Implementation

### 7.1 MLflow Experiment Tracking

MLflow is used to record model information, experiment runs, parameters, and artifacts.

Run:

```bash
python -m mlops.log_existing_model
```

Start the MLflow dashboard:

```bash
mlflow ui --port 5000
```

Open:

```text
http://localhost:5000
```

### 7.2 MLflow Model Registry

The trained CNN adapter can be registered and managed through MLflow Model Registry.

```bash
python -m mlops.register_model --model-uri "YOUR_MODEL_URI" --name IntelligentDeadReckoning
```

Replace `YOUR_MODEL_URI` with the actual model URI from your MLflow run.

### 7.3 Explainable AI

**SHAP**

```bash
python -m mlops.explain_shap
```

Used to analyze global feature contributions.

**LIME**

```bash
python -m mlops.explain_lime
```

Used to explain individual model outputs.

**Integrated Gradients**

```bash
python -m mlops.explain_integrated_gradients
```

Used to visualize the contribution of individual sensor channels and time steps.

### 7.4 Data Drift Monitoring

```bash
python -m mlops.monitor_drift
```

The current Vfa01 monitoring script compares early and late portions of one recording. It is an initial within-recording drift analysis, not yet a production monitoring system.

The generated HTML report is saved under:

```text
results/mlops/
```

### 7.5 GitHub Actions CI

The GitHub Actions workflow is located at:

```text
.github/workflows/mlops-ci.yml
```

It checks Python syntax and verifies the presence of important MLOps scripts, model configuration, and model checkpoint files.

Workflow execution can be monitored from the repository's **Actions** tab.

## 8. Model Information

The project uses a Dilated CNN adapter.

| Model property | Description |
|---|---|
| Model architecture | Dilated CNN |
| Input shape | 32 × 6 |
| Input channels | 3-axis gyroscope and 3-axis accelerometer |
| Output parameters | 11 |
| Framework | PyTorch |
| Navigation integration | IEKF |

The CNN adapter estimates navigation adaptation parameters rather than directly predicting GPS coordinates.

## 9. Model Evaluation

The existing project evaluation includes navigation metrics such as:

- Endpoint position error.
- Position RMSE.
- Drift percentage.
- Validation loss.

The Vta08 evaluation recorded the following CNN-adapted IEKF results:

| Evaluation metric | Result |
|---|---:|
| Endpoint position error | 29.770 m |
| Position RMSE | 27.845 m |
| Drift | 20.966% |
| Model parameters | 10,411 |

These are results from the existing project evaluation, not a claim of performance on every dataset or environment.

## 10. Responsible AI and Model Documentation

The project includes a Model Card documenting:

- Model purpose and intended use.
- Model architecture.
- Input and output information.
- Evaluation methodology.
- Explainability methods.
- Limitations and potential risks.

SHAP, LIME, and Integrated Gradients provide complementary explanations of model behavior.

The system is intended for research and experimental navigation development, not safety-critical navigation without further validation.

## 11. Future Enhancements

- Automated model retraining pipeline.
- Production-level data drift monitoring.
- Model performance dashboards.
- Automated model validation and deployment.
- Additional datasets and cross-device evaluation.
- Advanced model explainability and monitoring.

## 12. Conclusion

This project combines Intelligent Dead Reckoning with MLOps and Explainable AI to create a more manageable, interpretable, and reproducible machine-learning-assisted navigation pipeline.

By integrating PyTorch, MLflow, SHAP, LIME, Integrated Gradients, Evidently, and GitHub Actions, the project demonstrates important practices for managing and evaluating machine-learning systems throughout their lifecycle.

---

**Project:** MLOps-Enabled Intelligent Dead Reckoning System with Explainable AI

**GitHub:** https://github.com/POOJA-GARAI-004/MLOps-Enabled-Intelligent-Dead-Reckoning-System-with-Explainable-AI