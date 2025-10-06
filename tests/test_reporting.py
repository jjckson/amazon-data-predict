from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from reporting import dry_run_alerts, dry_run_dashboard_queries
from reporting.alerts import AlertConfig, AlertManager
from reporting.weekly_report import (
    ReportConfig,
    ScheduleConfig,
    WeeklyReportArtifacts,
    WeeklyReportGenerator,
    schedule_weekly_report,
)


class DummyQueryRunner:
    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames

    def run(self, sql: str, params: dict[str, object] | None = None) -> pd.DataFrame:
        if "GROUP BY snapshot_date, category" in sql:
            return self.frames["category_deltas"].copy()
        if "FROM pred_rank_daily" in sql:
            return self.frames["anomalies"].copy()
        return self.frames["top"].copy()


class DummyS3Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def upload_file(self, Filename: str, Bucket: str, Key: str) -> None:  # noqa: N803
        self.calls.append((Filename, Bucket, Key))


class DummySMTPClient:
    def __init__(self) -> None:
        self.sent_messages: list[Any] = []

    def send_message(self, message: Any) -> None:
        self.sent_messages.append(message)

    def quit(self) -> None:
        pass


@pytest.fixture()
def sample_frames() -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2023-01-01", periods=7)
    top = pd.DataFrame(
        {
            "snapshot_date": dates,
            "asin": [f"ASIN{i}" for i in range(len(dates))],
            "site": ["US"] * len(dates),
            "category": ["Electronics"] * len(dates),
            "predicted_revenue": [1500 + i * 10 for i in range(len(dates))],
            "predicted_units": [25 + i for i in range(len(dates))],
            "confidence_score": [0.85] * len(dates),
            "revenue_rank": list(range(1, len(dates) + 1)),
            "quality_rank": list(range(1, len(dates) + 1)),
            "ai_comment_summary": [
                "Auto summary emphasizes conversion uplift" if i % 2 == 0 else None
                for i in range(len(dates))
            ],
            "ai_keyword_cluster": [
                "electronics accessories" if i % 2 == 0 else "home gadgets"
                for i in range(len(dates))
            ],
        }
    )
    category = pd.DataFrame(
        {
            "snapshot_date": dates,
            "category": ["Electronics"] * len(dates),
            "predicted_revenue": [2000 + i * 5 for i in range(len(dates))],
        }
    )
    anomalies = pd.DataFrame(
        {
            "snapshot_date": dates,
            "site": ["US"] * len(dates),
            "category": ["Electronics"] * len(dates),
            "predicted_revenue": [1000, 1020, 1030, 1600, 1040, 1050, 1060],
        }
    )
    return {"top": top, "category_deltas": category, "anomalies": anomalies}


def test_dashboard_dry_run_executes_views() -> None:
    outputs = dry_run_dashboard_queries.run_dry_run()
    assert set(outputs) == {"vw_top_candidates_daily", "vw_features_latest", "pred_rank_daily"}
    for frame in outputs.values():
        assert not frame.empty


def test_weekly_report_generation(tmp_path: Path, sample_frames: dict[str, pd.DataFrame]) -> None:
    schedule = ScheduleConfig(weekday=0, run_time=dt.time(hour=0, minute=0))
    config = ReportConfig(
        output_dir=tmp_path,
        schedule=schedule,
        s3_bucket="example-bucket",
        email_recipients=["team@example.com"],
        email_sender="reports@example.com",
    )
    runner = DummyQueryRunner(sample_frames)
    generator = WeeklyReportGenerator(runner, config)
    artifacts = generator.generate_report(report_date=dt.date(2023, 1, 7))
    assert isinstance(artifacts, WeeklyReportArtifacts)
    assert artifacts.excel_path.exists()
    assert artifacts.pdf_path.exists()
    assert len(artifacts.top_candidates) == len(sample_frames["top"])
    assert artifacts.ai_comment_summaries is None
    assert artifacts.ai_keyword_clusters is None

    s3_client = DummyS3Client()
    smtp_client = DummySMTPClient()
    generator.deliver_report(
        artifacts,
        s3_client=s3_client,
        smtp_factory=lambda: smtp_client,
    )
    assert len(s3_client.calls) == 2
    assert len(smtp_client.sent_messages) == 1


def test_weekly_report_ai_sections(tmp_path: Path, sample_frames: dict[str, pd.DataFrame]) -> None:
    schedule = ScheduleConfig(weekday=0, run_time=dt.time(hour=0, minute=0))
    config = ReportConfig(
        output_dir=tmp_path,
        schedule=schedule,
        ai_enabled=True,
    )
    runner = DummyQueryRunner(sample_frames)
    generator = WeeklyReportGenerator(runner, config)
    artifacts = generator.generate_report(report_date=dt.date(2023, 1, 7))
    assert artifacts.ai_comment_summaries is not None
    assert artifacts.ai_keyword_clusters is not None
    assert config.ai_placeholder_text in artifacts.ai_comment_summaries.iloc[1].tolist()


def test_weekly_report_ai_section_fallback(tmp_path: Path, sample_frames: dict[str, pd.DataFrame]) -> None:
    frames_without_ai = sample_frames.copy()
    frames_without_ai["top"] = frames_without_ai["top"].drop(columns=["ai_comment_summary", "ai_keyword_cluster"])
    schedule = ScheduleConfig(weekday=0, run_time=dt.time(hour=0, minute=0))
    config = ReportConfig(output_dir=tmp_path, schedule=schedule, ai_enabled=True)
    runner = DummyQueryRunner(frames_without_ai)
    generator = WeeklyReportGenerator(runner, config)
    artifacts = generator.generate_report(report_date=dt.date(2023, 1, 7))
    assert artifacts.ai_comment_summaries is None
    assert artifacts.ai_keyword_clusters is None


def test_schedule_weekly_report(tmp_path: Path, sample_frames: dict[str, pd.DataFrame]) -> None:
    schedule = ScheduleConfig(weekday=0, run_time=dt.time(hour=0, minute=0))
    config = ReportConfig(output_dir=tmp_path, schedule=schedule)
    runner = DummyQueryRunner(sample_frames)
    generator = WeeklyReportGenerator(runner, config)
    now = dt.datetime(2023, 1, 2, 0, 1)
    artifacts = schedule_weekly_report(
        generator,
        now=now,
        s3_client=None,
        smtp_factory=None,
    )
    assert artifacts is not None
    assert generator.last_run_at is not None

    # Subsequent invocation within same week should be skipped
    artifacts_again = schedule_weekly_report(
        generator,
        now=dt.datetime(2023, 1, 2, 1, 0),
        s3_client=None,
        smtp_factory=None,
    )
    assert artifacts_again is None


def test_alert_manager_triggers_notifications(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = dry_run_alerts.build_sample_frames()
    config = AlertConfig(
        email_sender="alerts@example.com",
        email_recipients=["ops@example.com"],
        wecom_webhook="https://wecom.test",  # no network thanks to monkeypatch
        dingtalk_webhook="https://dingtalk.test",
    )
    smtp_client = DummySMTPClient()
    posts: list[tuple[str, dict[str, Any]]] = []

    def fake_post(url: str, json: dict[str, Any], timeout: int) -> Any:
        posts.append((url, json))

        class _Response:
            def raise_for_status(self) -> None:
                return None

        return _Response()

    monkeypatch.setattr("reporting.alerts.requests.post", fake_post)
    manager = AlertManager(config, smtp_factory=lambda: smtp_client)
    alerts = manager.run(**frames)
    assert alerts
    assert len(smtp_client.sent_messages) == 1
    assert len(posts) == 2
