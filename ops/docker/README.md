# Container Tooling

This directory provides container assets for local development and CI workloads. All images inherit the `.env` and `settings.yaml` configuration strategy used throughout the repository. Copy the `*.example` files to their secret-bearing counterparts and populate them with environment-specific values before executing any commands.

## Images

- `Dockerfile.training`: builds a container that runs the `training.build_labels` job to prepare supervised learning datasets. Mount `storage/` volumes when running locally so outputs persist across runs.
- `Dockerfile.inference`: builds a lightweight scoring container that executes `pipelines.score_baseline` by default. Override the command or set `SCORING_DATE` (via `.env`) to control inference windows.

## Docker Compose Stacks

### Training

`docker-compose.training.yaml` orchestrates the training job alongside supporting MLflow, PostgreSQL, and MinIO services. Usage:

```bash
cd ops/docker
cp ../mlflow_server/mlflow.env.example mlflow.env
cp mlflow-minio.env.example mlflow-minio.env
cp mlflow-postgres.env.example mlflow-postgres.env
cp training.env.example training.env
cp ../../config/secrets.example.env ../../config/secrets.env
# Populate the copied files with non-default secrets.
docker compose -f docker-compose.training.yaml up --build
```

### Inference

`docker-compose.inference.yaml` runs the inference job while reusing the shared MLflow configuration. Usage:

```bash
cd ops/docker
cp ../mlflow_server/mlflow.env.example mlflow.env
cp inference.env.example inference.env
cp ../../config/secrets.example.env ../../config/secrets.env
# Populate secrets, then run the scoring workflow.
docker compose -f docker-compose.inference.yaml up --build
```

### MLflow Stack

`docker-compose.mlflow.yaml` exposes MLflow behind a Traefik reverse proxy with TLS termination. Provide certificates under `traefik/certs/` and copy `traefik/users.htpasswd.example` to `users.htpasswd` with real credentials before bootstrapping the stack.

The Compose files expect the repository root to be mounted at `/app` inside each container, preserving access to `config/settings.yaml`. Secrets should *never* be committed—only the `.example` templates belong in source control.

