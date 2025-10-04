# Kubernetes Manifests

This directory contains reference Kubernetes manifests that align with the ML engineering workflows in this repository. They are intentionally split into composable components so you can generate environment-specific overlays with Kustomize or a GitOps tool. Every manifest references configuration via ConfigMaps, Secrets, and `.env`-style variables to avoid leaking sensitive data into version control.

## Directory Contents

- `mlflow-config.yaml`: ConfigMap for shared, non-sensitive MLflow configuration.
- `mlflow-secret.example.yaml`: Template Secret that mirrors `ops/mlflow_server/mlflow.env.example`.
- `mlflow-statefulset.yaml`: StatefulSet for the MLflow tracking server with persistent storage.
- `mlflow-service.yaml`: ClusterIP service exposing the MLflow pod.
- `mlflow-ingress.yaml`: HTTPS ingress definition including annotations for authentication proxies.
- `training-cronjob.yaml`: Schedules the training container built from `ops/docker/Dockerfile.training`.
- `inference-cronjob.yaml`: Schedules the inference container built from `ops/docker/Dockerfile.inference`.

## Usage Pattern

1. Create environment overlays (e.g., `overlays/dev`, `overlays/prod`) that patch images, resource limits, and domain names.
2. Store Secrets outside git by rendering `mlflow-secret.example.yaml` with your secret manager (e.g., `kubectl create secret generic mlflow-secrets --from-env-file=mlflow.env`).
3. Deploy via GitOps or CI/CD pipelines defined in `ops/ci_cd/`. Pipelines should template the manifests using the `.env` files produced for each environment.

## Configuration Merge Strategy

- For cluster-wide defaults, extend `config/settings.yaml` with Kubernetes-specific keys and inject them via ConfigMaps.
- For environment overrides, load `settings.<env>.yaml` inside your job containers (both Dockerfiles already honour `SETTINGS_PATH`).
- For secrets, rely on Kubernetes Secrets generated from `.env` files—never commit the populated files.

