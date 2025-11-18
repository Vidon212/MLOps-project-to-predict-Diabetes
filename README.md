# 🩺 Diabetes Prediction Model – Your First MLOps Project (FastAPI + Docker + K8s)

This project helps you learn **Building and Deploying an ML Model** using a simple and real-world use case: predicting whether a person is diabetic based on health metrics. We’ll go from:

- ✅ Model Training
- ✅ Building the Model locally
- ✅ API Deployment with FastAPI
- ✅ Dockerization
- ✅ Kubernetes Deployment

---

## 📊 Problem Statement

Predict if a person is diabetic based on:

- Pregnancies
- Glucose
- Blood Pressure
- BMI
- Age

We use a Random Forest Classifier trained on the **Pima Indians Diabetes Dataset**.

---

## 🚀 Quick Start

### 1. Clone the Repo

```bash
git clone https://github.com/Vidon212/MLOps-project-to-predict-Diabetes.git
cd MLOps-project-to-predict-Diabetes
```

### 2. Create Virtual Environment

```bash
python3 -m venv .mlops
source .mlops/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Train the Model (produces `diabetes_model.pkl` + `diabetes_model_meta.json`)

```bash
python train.py
# or with overrides:
# DATA_URL, MODEL_PATH and MODEL_METADATA_PATH are optional environment variables
# DATA_URL=https://your-source/diabetes.csv MODEL_PATH=diabetes_model.pkl MODEL_METADATA_PATH=diabetes_model_meta.json python train.py
```

## Run the API Locally

```bash
export MODEL_PATH=diabetes_model.pkl
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# in another terminal
curl -s http://localhost:8000/
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"Pregnancies":2,"Glucose":130,"BloodPressure":70,"BMI":28.5,"Age":45}'
```

### Sample Input for /predict

```json
{
  "Pregnancies": 2,
  "Glucose": 130,
  "BloodPressure": 70,
  "BMI": 28.5,
  "Age": 45
}
```

## Dockerize the API

### Build the Docker Image

```bash
# Use your own Docker Hub or Artifact Registry path to match k8s manifest
docker build -t your-dockerhub-username/diabetes-api:latest .
```

### Run the Container

```bash
docker run --rm -p 8000:8000 your-dockerhub-username/diabetes-api:latest
```

## Deploy to Kubernetes

```bash
# Ensure k8s-deploy.yml image matches what you built/pushed
kubectl apply -f k8s-deploy.yml
```
