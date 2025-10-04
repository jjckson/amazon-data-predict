"""Dry run helper to execute alert evaluation logic."""
from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

from reporting.alerts import AlertConfig, AlertManager


class _DummySMTPClient:
    def send_message(self, message: Any) -> None:  # pragma: no cover - simple stub
        print("Email message prepared:")
        print(message)

    def quit(self) -> None:  # pragma: no cover - simple stub
        pass


def _smtp_factory() -> _DummySMTPClient:
    return _DummySMTPClient()


def build_sample_frames() -> dict[str, pd.DataFrame]:
    today = dt.date.today()
    idx = pd.date_range(end=today, periods=7)
    predictions = pd.DataFrame(
        {
            "asin": [f"ASIN{i}" for i in range(len(idx))],
            "site": ["US"] * len(idx),
            "snapshot_date": idx,
            "predicted_revenue": [1000 + i * 10 for i in range(len(idx))],
            "confidence_score": [0.8] * len(idx),
        }
    )
    actuals = predictions.copy()
    actuals["actual_revenue"] = predictions["predicted_revenue"] * 0.4
    features_current = pd.DataFrame(
        {
            "feature_a": [0.1, 0.2, 0.3],
            "feature_b": [1.0, 1.1, 0.9],
        }
    )
    features_baseline = pd.DataFrame(
        {
            "feature_a": [0.1, 0.19, 0.28],
            "feature_b": [1.05, 1.0, 1.02],
        }
    )
    quality_frame = predictions[["asin", "site", "snapshot_date"]].copy()
    quality_frame["predicted_revenue"] = predictions["predicted_revenue"]
    quality_frame.loc[0, "predicted_revenue"] = pd.NA
    sla_events = pd.DataFrame(
        {
            "pipeline": ["weekly_report"],
            "scheduled_at": [dt.datetime.now() - dt.timedelta(minutes=200)],
            "completed_at": [dt.datetime.now()],
        }
    )
    return {
        "quality_frame": quality_frame,
        "predictions": predictions,
        "actuals": actuals,
        "feature_current": features_current,
        "feature_baseline": features_baseline,
        "sla_events": sla_events,
    }


def main() -> None:
    frames = build_sample_frames()
    config = AlertConfig(
        email_sender="alerts@example.com",
        email_recipients=["data-team@example.com"],
        wecom_webhook=None,
        dingtalk_webhook=None,
    )
    manager = AlertManager(config, smtp_factory=_smtp_factory)
    alerts = manager.run(**frames)
    if not alerts:
        print("No alerts triggered in dry run")
    else:
        print(f"Triggered {len(alerts)} alerts")
        for alert in alerts:
            print(f"- {alert.category}: {alert.message}")


if __name__ == "__main__":
    main()
