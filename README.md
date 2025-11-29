# MLOps Kubeflow Pipeline Project

## Project Overview
This project demonstrates an end-to-end MLOps workflow using Kubeflow Pipelines, Minikube, and DVC for versioned data management. The pipeline automates the process of **data extraction, preprocessing, model training, evaluation, and deployment**.  
The goal is to showcase a reproducible and scalable ML workflow that integrates version control, containerized environments, and CI/CD practices.

**Key Components:**
- **Data Versioning:** DVC to manage datasets and pipeline outputs.
- **Pipeline Orchestration:** Kubeflow Pipelines for automating ML workflow.
- **Local Deployment:** Minikube for running Kubeflow Pipelines locally.
- **Continuous Integration:** Jenkins/GitHub Actions for automated testing and pipeline compilation.

---

## Setup Instructions

### 1. Minikube
1. Install Minikube on Ubuntu:

```bash
sudo apt update
sudo apt install -y curl conntrack
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```
2. Start Minikube:

``` bash
minikube start --driver=docker --cpus=4 --memory=8192
```
3.Verify Minikube is running:

```bash
minikube status
```

### 2. Kubeflow Pipelines (Standalone)

1. Deploy KFP:

``` bash
export PIPELINE_VERSION=2.14.4
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=120s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=$PIPELINE_VERSION"
```

2. Check all pods are running:

``` bash
kubectl get pods -n kubeflow
```

3. Access the Kubeflow Pipelines UI:

``` bash
minikube service ml-pipeline-ui -n kubeflow
```
### 3. DVC Remote Storage
Initialize DVC in your project:

``` bash
dvc init
dvc remote add -d myremote s3://mybucket/path
dvc push
```

## Pipeline Walkthrough
1. Define Pipeline

All pipeline components are defined in pipeline.py using kfp.dsl.pipeline.

Components can be linked using outputs from one step as inputs to the next.

Example:

``` bash
@dsl.pipeline(
    name='ML Pipeline',
    description='An end-to-end ML pipeline.'
)
def my_pipeline():
    preprocess = preprocess_op()
    train = train_op(preprocess.output)
    evaluate = evaluate_op(train.output)
```

2. Compile Pipeline

``` bash
python pipeline.py --compile
```

This generates pipeline.yaml.

3. Run Pipeline

Upload pipeline.yaml to the KFP UI.

Trigger the pipeline run.

Monitor execution of all steps: data extraction → preprocessing → training → evaluation.

View outputs such as trained model metrics and evaluation results.
