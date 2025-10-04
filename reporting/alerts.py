"""Operational alerting for reporting pipelines."""
from __future__ import annotations

from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Callable, Iterable, Protocol, Sequence

import pandas as pd
import requests

from utils.backoff import RetryPolicy, retryable, sleep_with_backoff
from utils.logging import get_logger


LOGGER = get_logger(__name__)


class SMTPClient(Protocol):
    """Minimal interface for SMTP interactions."""

    def send_message(self, message: EmailMessage) -> None:  # noqa: D401
        ...

    def quit(self) -> None:  # noqa: D401
        ...


@dataclass(slots=True)
class AlertConfig:
    """Configuration for alert thresholds and notification channels."""

    data_quality_threshold: float = 0.97
    performance_threshold: float = 0.75
    drift_threshold: float = 0.2
    sla_threshold_minutes: int = 90
    email_sender: str | None = None
    email_recipients: Sequence[str] = field(default_factory=list)
    wecom_webhook: str | None = None
    dingtalk_webhook: str | None = None
    retry_policy: RetryPolicy = field(
        default_factory=lambda: RetryPolicy(
            max_attempts=3,
            base_delay_ms=500,
            jitter_ms=250,
            max_delay_ms=4000,
        )
    )


@dataclass(slots=True)
class AlertResult:
    """Represents a single alert finding."""

    category: str
    severity: str
    message: str
    details: dict[str, float | str]


class AlertEvaluator:
    """Encapsulates metric checks for operational monitoring."""

    def __init__(self, config: AlertConfig) -> None:
        self._config = config

    def evaluate_data_quality(self, frame: pd.DataFrame, dataset_name: str) -> list[AlertResult]:
        if frame.empty:
            return []
        missing_ratio = frame.isna().mean().mean()
        LOGGER.debug("Data quality missing ratio for %s: %.4f", dataset_name, missing_ratio)
        if missing_ratio <= 1 - self._config.data_quality_threshold:
            return []
        return [
            AlertResult(
                category="data_quality",
                severity="high",
                message=f"{dataset_name} missing ratio {missing_ratio:.2%} exceeds threshold",
                details={"missing_ratio": float(missing_ratio)},
            )
        ]

    def evaluate_performance(
        self,
        predictions: pd.DataFrame,
        actuals: pd.DataFrame,
        *,
        join_keys: Iterable[str] = ("asin", "site", "snapshot_date"),
    ) -> list[AlertResult]:
        if predictions.empty or actuals.empty:
            return []
        merged = predictions.merge(actuals, on=list(join_keys), suffixes=("_pred", "_act"))
        if merged.empty:
            return []
        if "predicted_revenue" not in merged or "actual_revenue" not in merged:
            return []
        actual = merged["actual_revenue"].astype(float)
        predicted = merged["predicted_revenue"].astype(float)
        denom = actual.replace(0, pd.NA).abs()
        mape = (predicted.subtract(actual).abs() / denom).dropna().mean()
        accuracy = 1 - mape if pd.notna(mape) else 1.0
        LOGGER.debug("Model accuracy computed as %.4f", accuracy)
        if accuracy >= self._config.performance_threshold:
            return []
        return [
            AlertResult(
                category="performance",
                severity="high",
                message=f"Model accuracy dropped to {accuracy:.2%}",
                details={"accuracy": float(accuracy)},
            )
        ]

    def evaluate_feature_drift(
        self,
        current: pd.DataFrame,
        baseline: pd.DataFrame,
        *,
        numeric_columns: Iterable[str] | None = None,
    ) -> list[AlertResult]:
        if current.empty or baseline.empty:
            return []
        numeric_columns = list(numeric_columns or current.select_dtypes("number").columns)
        if not numeric_columns:
            return []
        drifts: list[AlertResult] = []
        for column in numeric_columns:
            current_mean = current[column].astype(float).mean()
            baseline_mean = baseline[column].astype(float).mean()
            denominator = abs(baseline_mean) if baseline_mean else 1.0
            drift_score = abs(current_mean - baseline_mean) / denominator
            LOGGER.debug(
                "Drift score for %s: %.4f (current=%.4f baseline=%.4f)",
                column,
                drift_score,
                current_mean,
                baseline_mean,
            )
            if drift_score >= self._config.drift_threshold:
                drifts.append(
                    AlertResult(
                        category="drift",
                        severity="medium",
                        message=f"Feature {column} drift score {drift_score:.2f} exceeds threshold",
                        details={"feature": column, "drift_score": float(drift_score)},
                    )
                )
        return drifts

    def evaluate_sla(self, events: pd.DataFrame) -> list[AlertResult]:
        if events.empty:
            return []
        required_columns = {"scheduled_at", "completed_at", "pipeline"}
        if not required_columns.issubset(events.columns):
            return []
        events = events.copy()
        events["scheduled_at"] = pd.to_datetime(events["scheduled_at"])
        events["completed_at"] = pd.to_datetime(events["completed_at"])
        events["delay_minutes"] = (events["completed_at"] - events["scheduled_at"]).dt.total_seconds() / 60
        breaches = events[events["delay_minutes"] > self._config.sla_threshold_minutes]
        LOGGER.debug("Found %s SLA breaches", len(breaches))
        if breaches.empty:
            return []
        return [
            AlertResult(
                category="sla",
                severity="high",
                message="SLA breach detected",
                details={
                    "pipeline": row.pipeline,
                    "delay_minutes": float(row.delay_minutes),
                },
            )
            for row in breaches.itertuples()
        ]


class NotificationService:
    """Send alerts through configured channels."""

    def __init__(
        self,
        config: AlertConfig,
        smtp_factory: Callable[[], SMTPClient] | None = None,
    ) -> None:
        self._config = config
        self._smtp_factory = smtp_factory

    def _build_email(self, alerts: Sequence[AlertResult]) -> EmailMessage:
        message = EmailMessage()
        message["From"] = self._config.email_sender
        message["To"] = ", ".join(self._config.email_recipients)
        message["Subject"] = "Reporting alert notification"
        lines = ["The following alerts were triggered:"]
        for alert in alerts:
            lines.append(f"- [{alert.severity.upper()}] {alert.category}: {alert.message}")
        message.set_content("\n".join(lines))
        return message

    def send_email(self, alerts: Sequence[AlertResult]) -> None:
        if not self._config.email_sender or not self._config.email_recipients:
            LOGGER.debug("Email channel not configured; skipping")
            return
        if self._smtp_factory is None:
            raise ValueError("smtp_factory is required when email is enabled")
        client = self._smtp_factory()
        try:
            client.send_message(self._build_email(alerts))
        finally:
            client.quit()

    @retryable(retry_policy=RetryPolicy(max_attempts=3, base_delay_ms=250, jitter_ms=100, max_delay_ms=2000))
    def send_wecom(self, payload: dict[str, object]) -> None:
        if not self._config.wecom_webhook:
            LOGGER.debug("WeCom webhook not configured; skipping")
            return
        response = requests.post(self._config.wecom_webhook, json=payload, timeout=5)
        response.raise_for_status()

    @retryable(retry_policy=RetryPolicy(max_attempts=3, base_delay_ms=250, jitter_ms=100, max_delay_ms=2000))
    def send_dingtalk(self, payload: dict[str, object]) -> None:
        if not self._config.dingtalk_webhook:
            LOGGER.debug("DingTalk webhook not configured; skipping")
            return
        response = requests.post(self._config.dingtalk_webhook, json=payload, timeout=5)
        response.raise_for_status()


class AlertManager:
    """Coordinates evaluation and notification of alerts."""

    def __init__(
        self,
        config: AlertConfig,
        smtp_factory: Callable[[], SMTPClient] | None = None,
    ) -> None:
        self._config = config
        self._evaluator = AlertEvaluator(config)
        self._notifications = NotificationService(config, smtp_factory=smtp_factory)

    def run(
        self,
        *,
        quality_frame: pd.DataFrame,
        predictions: pd.DataFrame,
        actuals: pd.DataFrame,
        feature_current: pd.DataFrame,
        feature_baseline: pd.DataFrame,
        sla_events: pd.DataFrame,
    ) -> list[AlertResult]:
        alerts: list[AlertResult] = []
        alerts.extend(self._evaluator.evaluate_data_quality(quality_frame, "predictions_daily"))
        alerts.extend(self._evaluator.evaluate_performance(predictions, actuals))
        alerts.extend(
            self._evaluator.evaluate_feature_drift(
                feature_current,
                feature_baseline,
            )
        )
        alerts.extend(self._evaluator.evaluate_sla(sla_events))
        if not alerts:
            LOGGER.info("No alerts triggered during this evaluation")
            return []
        LOGGER.warning("Triggering %s alerts", len(alerts))
        self._notify(alerts)
        return alerts

    def _notify(self, alerts: Sequence[AlertResult]) -> None:
        retry_policy = self._config.retry_policy
        payload = {
            "msgtype": "text",
            "text": {
                "content": "\n".join(
                    f"[{alert.severity.upper()}] {alert.category}: {alert.message}" for alert in alerts
                )
            },
        }
        for attempt in range(retry_policy.max_attempts):
            try:
                self._notifications.send_email(alerts)
                self._notifications.send_wecom(payload)
                self._notifications.send_dingtalk(payload)
                return
            except Exception:  # noqa: BLE001
                LOGGER.exception("Notification attempt %s failed", attempt + 1)
                if attempt >= retry_policy.max_attempts - 1:
                    raise
                sleep_with_backoff(attempt, retry_policy)


__all__ = [
    "AlertConfig",
    "AlertEvaluator",
    "AlertManager",
    "AlertResult",
    "SMTPClient",
]
