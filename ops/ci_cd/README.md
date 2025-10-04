# CI/CD Pipelines

This directory contains reference automation for the project. The pipelines assume GitHub Actions, but the stages map cleanly to other orchestrators (GitLab CI, Azure DevOps, etc.). Each stage sources configuration from `.env` templates and the shared `config/settings.yaml` file to ensure secrets never reside directly in the workflow definition.

## Workflow Overview

The canonical workflow is defined in `github-actions.yaml` and implements the following stages:

1. **Static analysis** – Install dependencies, run `ruff` linting, and type checks with `mypy`.
2. **Unit tests** – Execute the pytest suite with coverage reporting. Data-contract checks run against the fixtures in `tests/`.
3. **Build artifacts** – Build and push versioned Docker images for training, inference, and MLflow using the Dockerfiles in `ops/docker/`.
4. **Data contract validation** – Run schema validation against the latest datasets using `great_expectations` (or a placeholder command until the contract suite is implemented).
5. **Staged rollout** – Promote artifacts through `dev`, `staging`, and `prod` environments with manual approvals, templating the Kubernetes manifests under `ops/k8s/`.

Environment-specific secrets live in `environments/<env>.env.example`. Copy the relevant file, populate secrets in your CI secret store, and load them at runtime using the `env` directive in GitHub Actions.

## Adding New Checks

Extend the `quality` job if additional code quality tools are required, or add standalone jobs for integration/end-to-end tests. Always ensure new jobs reference the same `.env`/`settings.yaml` patterns to keep secret handling consistent.

