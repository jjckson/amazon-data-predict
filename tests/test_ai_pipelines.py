"""Tests for AI pipeline utilities."""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from ai.models import EmbeddingRequest, EmbeddingResponse, GenerationRequest, GenerationResponse
from ai.pipelines import cluster_keywords, generate_comment_summaries
from ai.clients.base import EmbeddingClient, LLMClient


class StubLLMClient(LLMClient):
    """Stub LLM client capturing requests for assertions."""

    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
        return GenerationResponse(
            text=f"summary-{len(self.requests)}",
            model="stub-llm",
        )

    def chat(self, request):  # pragma: no cover - not needed for tests
        raise NotImplementedError


class StubEmbeddingClient(EmbeddingClient):
    """Stub embedding client returning deterministic vectors."""

    def __init__(self, vector_map: dict[str, list[float]]) -> None:
        self.vector_map = vector_map
        self.requests: list[EmbeddingRequest] = []

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.requests.append(request)
        embeddings = [self.vector_map[item] for item in request.inputs]
        return EmbeddingResponse(
            embeddings=embeddings,
            model="stub-embedding",
        )


def test_generate_comment_summaries(tmp_path: Path) -> None:
    comments = [
        "Love the durability of this item.",
        "Shipping was slower than expected.",
        "Customer support resolved my issue quickly.",
    ]
    client = StubLLMClient()

    output_path = generate_comment_summaries(
        comments,
        llm_client=client,
        batch_size=2,
        model="stub-llm-v1",
        window_start=datetime(2023, 1, 1),
        window_end=datetime(2023, 1, 7, 23, 59),
        report_date=date(2023, 1, 7),
        output_root=tmp_path,
    )

    assert output_path.exists()
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["model_version"] == "stub-llm-v1"
    assert data["summary_count"] == 2
    assert len(data["summaries"]) == 2

    prompts = [request.prompt for request in client.requests]
    assert len(prompts) == 2
    assert "Love the durability of this item." in prompts[0]
    assert "Customer support resolved my issue quickly." in "\n".join(prompts)

    expected_dir = tmp_path / "20230107" / "ai"
    assert output_path.parent == expected_dir


def test_cluster_keywords(tmp_path: Path) -> None:
    keywords = [
        "Battery life",
        "and",
        "Delivery speed",
        "Prime",
    ]
    vector_map = {
        "Battery life": [0.0, 0.0],
        "Delivery speed": [10.0, 0.0],
        "Prime": [0.0, 10.0],
    }
    client = StubEmbeddingClient(vector_map)

    output_path = cluster_keywords(
        keywords,
        embedding_client=client,
        num_clusters=2,
        model="embed-v1",
        window_start=datetime(2023, 2, 1),
        window_end=datetime(2023, 2, 7, 23, 59),
        report_date=date(2023, 2, 7),
        output_root=tmp_path,
        random_seed=0,
    )

    assert output_path.exists()
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["model_version"] == "embed-v1"
    assert data["num_clusters"] == 2

    captured_inputs = client.requests[0].inputs
    assert captured_inputs == ["Battery life", "Delivery speed", "Prime"]

    clustered_keywords = sorted(
        keyword["keyword"]
        for cluster in data["clusters"]
        for keyword in cluster["keywords"]
    )
    assert clustered_keywords == ["Battery life", "Delivery speed", "Prime"]

    expected_dir = tmp_path / "20230207" / "ai"
    assert output_path.parent == expected_dir
