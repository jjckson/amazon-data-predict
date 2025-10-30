"""Tests for AI pipeline utilities."""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from ai.models import EmbeddingRequest, EmbeddingResponse, GenerationRequest, GenerationResponse
from ai.pipelines import build_trading_prompt, cluster_keywords, generate_comment_summaries
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


def test_build_trading_prompt() -> None:
    system_state = {"minutes_since_start": 42, "current_time": "2024-04-01T12:00:00Z", "invocation_count": 5}
    market_state = [
        {
            "symbol": "BTC",
            "current_price": 114775.5,
            "current_ema20": 114457.686,
            "current_macd": 53.823,
            "current_rsi_7": 86.769,
            "open_interest_latest": 1234.0,
            "open_interest_avg": 987.0,
            "funding_rate": 0.0005,
            "short_term": {
                "interval": "5min",
                "mid_prices": [114339.5, 114500.0],
                "ema_series_20": [114329.3, 114480.5],
                "macd_series": [-60.9, -12.1],
                "rsi_series_7": [61.9, 65.2],
                "rsi_series_14": [51.1, 54.0],
            },
            "long_term": {
                "interval": "4h",
                "ema_20_4h": 113453.764,
                "ema_50_4h": 112031.947,
                "atr_3_4h": 319.999,
                "atr_14_4h": 523.419,
                "current_volume": 8.146,
                "avg_volume": 4618.058,
                "macd_series_4h": [1082.2, 950.1],
                "rsi_series_14_4h": [72.327, 70.1],
            },
        }
    ]
    account_info = {
        "total_return_percent": 12.3,
        "available_cash": 1000.0,
        "account_value": 12345.6,
        "positions": [
            {
                "symbol": "BTC",
                "quantity": 0.5,
                "entry_price": 100000.0,
                "current_price": 114775.5,
                "unrealized_pnl": 7387.75,
                "leverage": 3,
                "exit_plan": {
                    "profit_target": 120000.0,
                    "stop_loss": 95000.0,
                    "invalidation_condition": "Breaks below 90k",
                },
                "confidence": "high",
                "risk_usd": 500.0,
                "notional_usd": 57387.75,
            }
        ],
    }
    performance = {"sharpe_ratio": 1.23}

    prompt = build_trading_prompt(system_state, market_state, account_info, performance_metrics=performance)

    assert "🧭 SYSTEM STATE" in prompt
    assert "SYMBOL: BTC" in prompt
    assert "current_price = 114775.5" in prompt
    assert "Positions:" in prompt
    assert "Sharpe Ratio: 1.23" in prompt
