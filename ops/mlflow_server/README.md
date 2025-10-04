# MLflow Tracking Server Operations Guide

This directory contains reference infrastructure and operational guidance for hosting a production-grade MLflow Tracking server that integrates with the existing configuration patterns used in this repository.

## Architecture Overview

The recommended deployment uses:

- **MLflow server** running behind an ingress or reverse proxy that enforces HTTPS and handles authentication.
- **Object storage** (e.g., Amazon S3, MinIO, or Azure Blob Storage) for MLflow artifact storage.
- **Relational database** (e.g., PostgreSQL, MySQL) for the MLflow backend store to ensure transactional integrity.
- **Centralized secret management** via `.env` files checked into version control as `*.example` templates plus environment-specific `settings.yaml` overrides stored in a secure secret manager (HashiCorp Vault, AWS Secrets Manager, etc.).

The high-level layout is illustrated below:

```
client -> ingress -> mlflow server -> backend database
                                \-> artifact storage bucket
```

## Configuration Strategy

1. Copy `mlflow.env.example` to `mlflow.env` for local execution or inject the variables via your CI/CD secret manager. Never commit populated secret files.
2. Apply environment-specific overrides by extending `config/settings.yaml` with an `mlflow` section or by providing a sibling `settings.<env>.yaml` that is merged at runtime by your orchestration scripts.
3. Sync the same environment variables to the infrastructure defined in `ops/docker/` and `ops/k8s/` to ensure consistent credentials everywhere the MLflow client is used.

### Required Environment Variables

The server honours the variables shown below (see `mlflow.env.example`):

- `MLFLOW_HOST` / `MLFLOW_PORT`: listening address for the MLflow server.
- `MLFLOW_BACKEND_STORE_URI`: SQLAlchemy connection string for the backend database. Store this in your secret manager and surface it as an environment variable at deploy time.
- `MLFLOW_DEFAULT_ARTIFACT_ROOT`: S3/MinIO/Blob path for artifact storage.
- `MLFLOW_S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`: optional values when using S3-compatible storage (e.g., MinIO on-premises).
- `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`: optional HTTP Basic Auth credentials when not delegating auth to the ingress.
- `MLFLOW_UI_OAUTH_CLIENT_ID`, `MLFLOW_UI_OAUTH_CLIENT_SECRET`, `MLFLOW_UI_OAUTH_SCOPES`: optional fields for OAuth2 proxies.

All secrets should remain outside version control—use `.env` for local development and the CI/CD variables defined under `ops/ci_cd/` for hosted environments.

## Local Development with Docker Compose

A reference Docker Compose file is provided in `../docker/docker-compose.mlflow.yaml`. It launches:

- MLflow server container.
- MinIO for artifact storage.
- PostgreSQL for the backend store.
- Traefik reverse proxy that terminates TLS (self-signed for local testing) and enforces HTTP Basic Auth.

To start the stack locally:

```bash
cp mlflow.env.example mlflow.env
# Populate mlflow.env with local secrets
cp ../docker/mlflow-minio.env.example ../docker/mlflow-minio.env
cp ../docker/mlflow-postgres.env.example ../docker/mlflow-postgres.env
cd ..
docker compose -f docker/docker-compose.mlflow.yaml up --build
```

After the containers are ready, open `https://localhost:8443` and authenticate with the credentials defined in `mlflow.env` or the proxy configuration.

## Kubernetes Deployment

Kubernetes manifests live under `../k8s/`. The key resources are:

- `mlflow-config.yaml`: ConfigMap for non-secret settings.
- `mlflow-secret.example.yaml`: template for Kubernetes Secret referencing the same variables defined in `mlflow.env.example`.
- `mlflow-statefulset.yaml`: MLflow server pods with persistent volumes for local artifact caching.
- `mlflow-ingress.yaml`: Ingress resource configured with TLS and optional external authentication annotations.

Use your GitOps or CI/CD workflow (see `ops/ci_cd/`) to render environment-specific manifests by injecting the correct `.env` values at deploy time.

## Backup and Disaster Recovery

- **Database**: enable automated backups using the managed database provider or scheduled `pg_dump` jobs stored in encrypted object storage.
- **Artifact storage**: ensure the bucket has lifecycle policies and cross-region replication if required.
- **Configuration**: keep versioned copies of `settings.yaml` and `.env` templates in the repository. Sensitive runtime values should live in the secret manager with audit logging enabled.

## Monitoring and Alerts

- Scrape MLflow server metrics via sidecar exporters or integrate Traefik/Ingress logs with your logging pipeline.
- Track error rates and request latency from the reverse proxy to detect service degradations.
- Configure alerts for storage and database availability, plus TLS certificate expiry reminders if you manage certificates manually.

## Maintenance Checklist

- Rotate API keys and database credentials according to compliance requirements.
- Patch container base images monthly and redeploy using the CI/CD pipelines.
- Review access logs for unusual activity and update RBAC rules as needed.
- Periodically test disaster recovery by restoring from backups in a non-production environment.

