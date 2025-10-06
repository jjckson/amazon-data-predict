"""HTTP service definitions for the inference API."""
from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, status

from inference.schemas import KeywordClusterRequestSchema, SummaryRequestSchema

app = FastAPI(title="Inference Service")


def get_ai_client() -> None:
    """Placeholder dependency for an external AI provider client."""

    # TODO(manual): 注入真实 LLM 客户端与鉴权逻辑
    return None


@app.post("/v1/ai/summarise")
async def summarise_endpoint(
    payload: SummaryRequestSchema,  # noqa: B008 - FastAPI dependency injection
    ai_client: None = Depends(get_ai_client),
) -> None:
    """Summarise product insights via an external AI provider."""

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="AI summarisation endpoint is not yet implemented.",
    )


@app.post("/v1/ai/keywords")
async def keyword_cluster_endpoint(
    payload: KeywordClusterRequestSchema,  # noqa: B008 - FastAPI dependency injection
    ai_client: None = Depends(get_ai_client),
) -> None:
    """Cluster keywords leveraging an external AI provider."""

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Keyword clustering endpoint is not yet implemented.",
    )
