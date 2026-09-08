# MLflow Demo — Iris Classifier MLOps Pipeline

A hands-on MLOps project demonstrating how to take a machine learning model from training to registration, evaluation, promotion, API serving, Docker, and CI/CD.

The project uses the Iris dataset and is designed to demonstrate production-style ML workflows locally and through GitHub Actions.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [ML Lifecycle](#ml-lifecycle)
- [Dataset Versioning](#dataset-versioning)
- [Model Training](#model-training)
- [MLflow Tracking](#mlflow-tracking)
- [MLflow Model Registry](#mlflow-model-registry)
- [Automated Model Promotion](#automated-model-promotion)
- [FastAPI Prediction API](#fastapi-prediction-api)
- [Testing](#testing)
- [Docker](#docker)
- [CI/CD with GitHub Actions](#cicd-with-github-actions)
- [Running the Project Locally](#running-the-project-locally)
- [Running Training in Docker](#running-training-in-docker)
- [Checking a Model's Dataset](#checking-a-models-dataset)
- [Environment Differences](#environment-differences)
- [Important Design Decisions](#important-design-decisions)
- [Future Production Architecture](#future-production-architecture)
- [Learning Outcomes](#learning-outcomes)

---

# Overview

This project demonstrates a complete machine learning workflow:

```text
Dataset
   │
   ▼
Dataset Hash / Version
   │
   ▼
Model Training
   │
   ▼
Metrics
   │
   ▼
MLflow Tracking
   │
   ▼
MLflow Model Registry
   │
   ▼
Quality Gate
   │
   ▼
Champion Model
   │
   ▼
FastAPI
   │
   ▼
Docker
```

The project also integrates automated testing and CI/CD:

```text
Developer
    │
    ▼
Git Push / Pull Request
    │
    ▼
GitHub Actions
    │
    ├── Run Tests
    │
    ├── Start Temporary MLflow
    │
    ├── Train Model
    │
    ├── Evaluate Model
    │
    └── Promote if Quality Gate Passes
```

The goal is not to build the most sophisticated machine learning model.

The goal is to understand the engineering surrounding a machine learning model.

---

# Architecture

## Local Development Architecture

```text
                     ┌──────────────────────┐
                     │      Iris Dataset    │
                     │     data/iris.csv    │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ Dataset Hashing      │
                     │ SHA-256              │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ Versioned Dataset    │
                     │ data/versions/       │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ Training             │
                     │ scikit-learn         │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ MLflow Tracking      │
                     │ Parameters           │
                     │ Metrics              │
                     │ Artifacts            │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ MLflow Model Registry│
                     │ iris-classifier-model│
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ Quality Gate         │
                     │ Accuracy             │
                     │ Precision            │
                     │ Recall               │
                     │ F1                   │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ champion Alias      │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ FastAPI              │
                     │ /predict             │
                     └──────────────────────┘
```

---

# Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.12 | Application and ML code |
| scikit-learn | Machine learning |
| MLflow | Experiment tracking and model registry |
| FastAPI | Model serving API |
| Pydantic | API request validation |
| pytest | Automated testing |
| Docker | Containerization |
| Git | Version control |
| GitHub Actions | CI/CD |
| SQLite | Local MLflow backend during CI simulation |

---

# Project Structure

```text
MLflow Demo/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── predictor.py
│   └── schemas.py
│
├── data/
│   ├── iris.csv
│   └── versions/
│
├── tests/
│   └── test_api.py
│
├── training/
│   ├── __init__.py
│   ├── train.py
│   ├── promote.py
│   ├── pipeline.py
│   ├── get_dataset.py
│   └── Dockerfile
│
├── pytest.ini
├── requirements.txt
├── Dockerfile
└── .gitignore
```

---

# ML Lifecycle

The project follows this lifecycle:

```text
1. Load dataset
       ↓
2. Calculate dataset SHA-256 hash
       ↓
3. Store immutable dataset version
       ↓
4. Split dataset
       ↓
5. Train model
       ↓
6. Calculate evaluation metrics
       ↓
7. Log parameters and metrics to MLflow
       ↓
8. Register model
       ↓
9. Apply quality gate
       ↓
10. Compare against current champion
       ↓
11. Promote candidate if approved
       ↓
12. Serve champion through FastAPI
```

---

# Dataset Versioning

A dataset can change while the model code remains unchanged.

To make the training process traceable, the project calculates a SHA-256 hash of the dataset file.

Example:

```text
Dataset hash:
c6d77a4cae3212dbecb49bb2ea9224ed336ffb3d9fb4543cdd8842da196ce5ea
```

The hash is used to create an immutable dataset copy:

```text
data/versions/
└── c6d77a4cae3212dbecb49bb2ea9224ed336ffb3d9fb4543cdd8842da196ce5ea.csv
```

The training run also records:

```text
dataset_file
dataset_hash
versioned_dataset_path
```

This allows a model version to be associated with the exact dataset version used during training.

## Why this matters

A model should not only answer:

> Which code produced this model?

It should also answer:

> Which data produced this model?

The combination of:

```text
Model
+
Code
+
Parameters
+
Metrics
+
Dataset version
```

provides much better reproducibility and traceability.

---

# Model Training

The training implementation is located at:

```text
training/train.py
```

The project currently uses:

```python
GradientBoostingClassifier
```

with:

```text
n_estimators = 100
max_depth = 5
random_state = 42
```

The dataset is split using:

```python
train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)
```

The following metrics are calculated:

- Accuracy
- Precision
- Recall
- F1 Score

The metrics are logged to MLflow.

Example:

```text
accuracy: 1.0000
precision: 1.0000
recall: 1.0000
f1: 1.0000
```

---

# MLflow Tracking

MLflow is used for:

- Experiment tracking
- Parameter tracking
- Metric tracking
- Dataset tracking
- Model artifacts
- Model registration

The experiment name is:

```text
iris-classifier
```

The registered model name is:

```text
iris-classifier-model
```

---

# MLflow Model Registry

Every successful training run can register a new model version.

For example:

```text
iris-classifier-model
    │
    ├── Version 1
    ├── Version 2
    ├── Version 3
    ├── ...
    └── Version N
```

The application does not need to know a specific model version.

Instead, the API uses the MLflow alias:

```text
models:/iris-classifier-model@champion
```

This provides an important separation:

```text
Application
    │
    ▼
champion
    │
    ▼
Current approved model
```

The application can continue using the same URI even when the underlying model version changes.

---

# Automated Model Promotion

Model promotion is handled by:

```text
training/promote.py
```

The candidate model must first pass a quality gate.

Current thresholds:

```text
Accuracy  >= 0.90
Precision >= 0.90
Recall    >= 0.90
F1        >= 0.90
```

Conceptually:

```text
                  Candidate Model
                        │
                        ▼
              ┌──────────────────┐
              │ Quality Gate     │
              └────────┬─────────┘
                       │
                ┌──────┴──────┐
                │             │
              FAIL           PASS
                │             │
                ▼             ▼
          Reject Model    Compare F1
                              │
                              ▼
                     Current Champion
                              │
                       ┌──────┴──────┐
                       │             │
                  Candidate       Candidate
                  worse           better/equal
                       │             │
                       ▼             ▼
                    Reject        Promote
```

A candidate that fails the quality gate does not replace the champion.

Example:

```text
accuracy: 0.3867
precision: 0.4240
recall: 0.3867
f1: 0.3934

Candidate failed quality gate.
Champion will remain unchanged.
```

This prevents an obviously poor model from automatically becoming the production model.

---

# Champion Alias

The project uses the MLflow alias:

```text
champion
```

When a model is promoted:

```text
iris-classifier-model
        │
        └── champion → Version N
```

If a better model is trained:

```text
iris-classifier-model
        │
        └── champion → Version N+1
```

The API does not need to change.

This is one of the important ideas behind model registries:

> The serving application points to a logical model role rather than a hard-coded model version.

---

# FastAPI Prediction API

The serving application is located in:

```text
app/
```

Main application:

```text
app/main.py
```

Prediction logic:

```text
app/predictor.py
```

Request schemas:

```text
app/schemas.py
```

## Home Endpoint

```http
GET /
```

Example response:

```json
{
  "message": "Iris prediction API is running"
}
```

## Prediction Endpoint

```http
POST /predict
```

Example request:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Example response:

```json
{
  "prediction": 0
}
```

The API validates incoming data using Pydantic.

Invalid input returns:

```text
HTTP 422
```

---

# Lazy Model Loading

The model predictor uses lazy loading.

The model is not loaded when the Python module is imported.

Instead, it is loaded when prediction is actually requested.

Conceptually:

```text
Application starts
      │
      ▼
FastAPI loads
      │
      ▼
No MLflow model request yet
      │
      ▼
POST /predict
      │
      ▼
Load champion model
      │
      ▼
Make prediction
```

This design is particularly useful for testing and CI because the application can be imported without requiring an available MLflow server.

---

# Testing

Tests are located in:

```text
tests/test_api.py
```

The project tests:

### Home endpoint

```text
GET /
```

### Prediction endpoint

```text
POST /predict
```

### Invalid input

The API should reject invalid feature values.

Example:

```json
{
  "sepal_length": "hello",
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Expected:

```text
HTTP 422
```

---

# Test Isolation

The API tests do not require a real MLflow model.

A fake model is injected:

```python
class FakeModel:
    def predict(self, features):
        return [0]
```

This allows the API layer to be tested independently of:

- MLflow
- Model Registry
- Network connectivity
- A running model server

This is an important testing principle:

> Unit/API tests should not depend unnecessarily on external infrastructure.

---

# Docker

The project contains Docker support for both the application and training.

## Training Dockerfile

Located at:

```text
training/Dockerfile
```

Build:

```powershell
docker build -f training\Dockerfile -t iris-training .
```

Run:

```powershell
docker run --rm iris-training
```

The container runs the training pipeline.

---

# Running the Project Locally

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd "MLflow Demo"
```

---

## 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
```

Activate:

```powershell
venv\Scripts\activate
```

---

## 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# Start MLflow

The local MLflow server runs on port `5000`.

Start it with:

```powershell
mlflow server --host 0.0.0.0 --port 5000 --allowed-hosts "host.docker.internal:5000,localhost:5000,127.0.0.1:5000"
```

MLflow UI:

```text
http://127.0.0.1:5000
```

Keep the MLflow server running while training or serving the model locally.

---

# Run the Training Pipeline

From the project root:

```powershell
python training/pipeline.py
```

The pipeline:

1. Calculates the dataset hash
2. Creates a dataset version if needed
3. Trains the model
4. Calculates metrics
5. Logs the run to MLflow
6. Registers the model
7. Applies the quality gate
8. Compares the candidate with the champion
9. Promotes the model if approved

Example successful output:

```text
Dataset hash: c6d77a4cae3212dbecb49bb2ea9224ed336ffb3d9fb4543cdd8842da196ce5ea

accuracy: 1.0000
precision: 1.0000
recall: 1.0000
f1: 1.0000

Candidate passed quality gate.
Version promoted to champion.

PIPELINE COMPLETED
```

---

# Run the API

Start the API from the **project root**.

```powershell
python -m uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

Open `/docs` in a browser to interact with the API.

---

# Test the API

Using Swagger:

```text
http://127.0.0.1:8000/docs
```

Or with PowerShell:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

---

# Run Tests

From the project root:

```powershell
python -m pytest
```

Expected result:

```text
3 passed
```

---

# Running Training in Docker

Build the training image:

```powershell
docker build -f training\Dockerfile -t iris-training .
```

Run:

```powershell
docker run --rm iris-training
```

When running training inside Docker on Windows, the container uses:

```text
host.docker.internal
```

to reach the MLflow server running on the host machine.

For example:

```python
mlflow.set_tracking_uri(
    "http://host.docker.internal:5000"
)
```

---

# Checking a Model's Dataset

The project includes:

```text
training/get_dataset.py
```

This script looks up the dataset information associated with a registered model version.

Usage:

```powershell
python training/get_dataset.py <model_version>
```

Example:

```powershell
python training/get_dataset.py 1
```

Example output:

```text
Model version: 1
Run ID: ...
Dataset hash: ...
Dataset path: data/versions/<hash>.csv

Exact dataset found.
```

This demonstrates model-to-data traceability.

---

# CI/CD with GitHub Actions

The GitHub Actions workflow is:

```text
.github/workflows/ci.yml
```

The workflow performs automated testing and training.

The workflow includes:

```text
Checkout repository
        ↓
Set up Python
        ↓
Install dependencies
        ↓
Run pytest
        ↓
Start temporary MLflow server
        ↓
Run training pipeline
        ↓
Evaluate candidate
        ↓
Apply promotion logic
```

---

# GitHub Actions MLflow Environment

GitHub Actions runs on a separate runner.

Therefore, it cannot access the MLflow server running on your personal computer.

For CI, the workflow starts a temporary MLflow server inside the GitHub Actions runner:

```bash
mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts
```

This allows the training pipeline to communicate with MLflow during the CI run.

---

# Important CI/CD Limitation

The MLflow server started by GitHub Actions is temporary.

Each workflow run starts with a fresh environment.

Therefore, model versions in CI are not persistent between workflow runs.

For example, one workflow may show:

```text
Created version '1'
No champion currently exists.
Version 1 promoted to champion.
```

A later workflow may again show:

```text
Created version '1'
No champion currently exists.
Version 1 promoted to champion.
```

This is expected.

The GitHub Actions workflow is currently a **CI/CD simulation**, not a persistent production MLflow deployment.

A real production setup would use a persistent MLflow backend and artifact storage.

---

# Branch Workflow

The GitHub Actions workflow is configured to run on pull requests and pushes to the `main` branch.

Conceptually:

```text
Pull Request
     │
     ▼
Run tests
```

And:

```text
Push to main
     │
     ▼
Run tests
     │
     ▼
Run training pipeline
     │
     ▼
Evaluate candidate
     │
     ▼
Promote if approved
```

This provides a basic foundation for separating code validation from model delivery.

---

# Environment Differences

There are three important environments in this project.

## Local Python

When running training directly from Windows:

```text
http://127.0.0.1:5000
```

is used for MLflow.

---

## Docker on Windows

When the training container needs to communicate with MLflow running on the Windows host:

```text
http://host.docker.internal:5000
```

is used.

`host.docker.internal` is a Docker-specific hostname that allows containers to reach services running on the host machine.

---

## GitHub Actions

GitHub Actions runs on a remote Linux runner.

It cannot reach:

```text
127.0.0.1
```

on your personal computer.

Instead, the workflow starts its own temporary MLflow server:

```text
GitHub Actions Runner
        │
        ├── MLflow
        │
        ├── SQLite backend
        │
        └── Training pipeline
```

This distinction becomes important when moving from local development to cloud infrastructure.

---

# Important Design Decisions

## 1. Dataset Hash

The dataset receives a SHA-256 hash.

This provides a reproducible identifier for the exact dataset file.

---

## 2. Immutable Dataset Copies

The actual dataset is copied into:

```text
data/versions/
```

using the hash as the filename.

A hash alone does not contain the original dataset.

The actual versioned file must therefore be stored somewhere.

---

## 3. MLflow Registry

Models are registered using:

```text
iris-classifier-model
```

This gives every trained model a version.

---

## 4. Champion Alias

The API uses:

```text
models:/iris-classifier-model@champion
```

instead of a hard-coded model version.

This means the serving code does not need to change whenever a new model is promoted.

---

## 5. Quality Gates

A model must satisfy minimum quality requirements before promotion.

Current thresholds:

```text
Accuracy  >= 0.90
Precision >= 0.90
Recall    >= 0.90
F1        >= 0.90
```

---

## 6. Candidate vs Champion

A newly trained model is treated as a candidate.

It does not automatically replace the existing champion.

The candidate must:

1. Pass the quality gate
2. Be compared with the current champion
3. Meet the promotion criteria

---

## 7. Lazy Model Loading

The FastAPI application does not load the MLflow model during import.

The model is loaded only when needed.

This keeps application tests independent of MLflow infrastructure.

---

## 8. Tests Use a Fake Model

The API tests inject a fake model.

This avoids requiring MLflow during unit/API testing.

The actual MLflow model is tested separately through the training pipeline.

---

# Future Production Architecture

This project intentionally stays local and does not require paid cloud services.

A future cloud architecture could look like:

```text
                   Git Repository
                        │
                        ▼
                CI/CD Pipeline
                        │
              ┌─────────┴─────────┐
              │                   │
          Run Tests          Train Model
                                  │
                                  ▼
                           Evaluate Model
                                  │
                                  ▼
                            Quality Gate
                                  │
                                  ▼
                          Model Registry
                                  │
                                  ▼
                         Promote Champion
                                  │
                                  ▼
                          Container Image
                                  │
                                  ▼
                         Cloud Deployment
                                  │
                                  ▼
                            FastAPI API
```

The underlying concepts remain the same.

Only the infrastructure changes.

For example:

| Local | Production |
|---|---|
| Local MLflow server | Hosted MLflow |
| Local SQLite | Managed database |
| Local artifacts | Cloud object storage |
| Docker locally | Container registry |
| Uvicorn locally | Managed container service |
| GitHub Actions simulation | Production CI/CD |
| Local dataset versions | Object storage / data lake |
| MLflow alias | Production model deployment |

The purpose of this project is to understand these concepts before introducing cloud infrastructure.

---

# Learning Outcomes

By completing this project, the following MLOps concepts have been implemented.

## Machine Learning

- Dataset loading
- Train/test split
- Model training
- Model evaluation
- Multiple evaluation metrics

## Experiment Tracking

- MLflow experiments
- Parameters
- Metrics
- Dataset tracking
- Model artifacts

## Model Management

- Model Registry
- Model versions
- Model aliases
- Champion model
- Candidate model
- Automated promotion

## Data Management

- SHA-256 dataset hashing
- Dataset versioning
- Model-to-dataset traceability

## Software Engineering

- Python packages
- FastAPI
- Pydantic validation
- Automated tests
- Dependency management
- Lazy loading
- Separation of concerns

## DevOps

- Git
- GitHub
- GitHub Actions
- Docker
- CI/CD

---

# Key Takeaway

The most important concept demonstrated by this project is that an ML system is more than a trained model.

A production-oriented ML workflow connects:

```text
Code
 │
 ├── Dataset
 │
 ├── Training
 │
 ├── Parameters
 │
 ├── Metrics
 │
 ├── Model Version
 │
 ├── Quality Gate
 │
 ├── Promotion
 │
 ├── API
 │
 ├── Container
 │
 └── CI/CD
```

The model is only one part of the system.

This project provides a small but complete example of how those pieces fit together.

---

## License

This project is intended for educational and demonstration purposes.