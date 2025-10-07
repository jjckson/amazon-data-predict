"""Weekly merchandising report generation utilities."""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence
from xml.sax.saxutils import escape
import zipfile

import pandas as pd

from utils.backoff import RetryPolicy, sleep_with_backoff
from utils.logging import get_logger


LOGGER = get_logger(__name__)


def _column_letter(index: int) -> str:
    """Convert a zero-based column index into Excel column letters."""

    result = ""
    while True:
        index, remainder = divmod(index, 26)
        result = chr(ord("A") + remainder) + result
        if index == 0:
            break
        index -= 1
    return result


class SimpleXLSXBuilder:
    """Create a minimal XLSX workbook without external dependencies."""

    def __init__(self) -> None:
        self._shared_strings: dict[str, int] = {}
        self._strings: list[str] = []

    def _string_index(self, value: str) -> int:
        if value not in self._shared_strings:
            self._shared_strings[value] = len(self._strings)
            self._strings.append(value)
        return self._shared_strings[value]

    @staticmethod
    def _normalise(value: object) -> object:
        if isinstance(value, (dt.datetime, dt.date)):
            return value.isoformat()
        if pd.isna(value):
            return None
        return value

    def _sheet_rows(self, frame: pd.DataFrame) -> list[list[object]]:
        rows = [list(frame.columns)]
        for row in frame.itertuples(index=False):
            rows.append([self._normalise(value) for value in row])
        return rows

    def _sheet_xml(self, frame: pd.DataFrame) -> bytes:
        rows = self._sheet_rows(frame)
        xml_rows: list[str] = []
        for row_idx, values in enumerate(rows, start=1):
            cells: list[str] = []
            for col_idx, value in enumerate(values):
                ref = f"{_column_letter(col_idx)}{row_idx}"
                if value is None:
                    continue
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if isinstance(value, float) and math.isnan(value):
                        continue
                    cell = f'<c r="{ref}" t="n"><v>{value}</v></c>'
                else:
                    string_value = escape(str(value))
                    index = self._string_index(string_value)
                    cell = f'<c r="{ref}" t="s"><v>{index}</v></c>'
                cells.append(cell)
            xml_rows.append(f'<row r="{row_idx}">' + "".join(cells) + "</row>")
        xml = (
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">"
            "<sheetData>"
            + "".join(xml_rows)
            + "</sheetData></worksheet>"
        )
        return xml.encode("utf-8")

    def _shared_strings_xml(self) -> bytes:
        entries = "".join(f'<si><t>{string}</t></si>' for string in self._strings)
        xml = (
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            f"<sst xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" count=\"{len(self._strings)}\" uniqueCount=\"{len(self._strings)}\">"
            + entries
            + "</sst>"
        )
        return xml.encode("utf-8")

    def write(self, path: Path, sheets: dict[str, pd.DataFrame]) -> None:
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(
                "[Content_Types].xml",
                """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>
"""
                + "".join(
                    f'  <Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                    for idx in range(1, len(sheets) + 1)
                )
                + "\n</Types>",
            )
            zf.writestr(
                "_rels/.rels",
                """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>""",
            )
            zf.writestr(
                "xl/_rels/workbook.xml.rels",
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
                "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
                + "".join(
                    f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
                    for idx in range(1, len(sheets) + 1)
                )
                + "</Relationships>",
            )
            zf.writestr(
                "xl/workbook.xml",
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
                "<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"
                "<sheets>"
                + "".join(
                    f'<sheet name="{escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>'
                    for idx, name in enumerate(sheets, start=1)
                )
                + "</sheets></workbook>",
            )
            zf.writestr(
                "xl/styles.xml",
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?><styleSheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\"><fonts count=\"1\"><font><sz val=\"11\"/><name val=\"Calibri\"/></font></fonts><fills count=\"1\"><fill><patternFill patternType=\"none\"/></fill></fills><borders count=\"1\"><border/></borders><cellStyleXfs count=\"1\"><xf/></cellStyleXfs><cellXfs count=\"1\"><xf xfId=\"0\" applyNumberFormat=\"0\"/></cellXfs><cellStyles count=\"1\"><cellStyle name=\"Normal\" xfId=\"0\" builtinId=\"0\"/></cellStyles></styleSheet>",
            )
            for idx, (name, frame) in enumerate(sheets.items(), start=1):
                zf.writestr(f"xl/worksheets/sheet{idx}.xml", self._sheet_xml(frame))
            zf.writestr("xl/sharedStrings.xml", self._shared_strings_xml())


class SimplePDFBuilder:
    """Minimal PDF builder that supports simple text tables."""

    def __init__(self) -> None:
        self._lines: list[str] = []

    @staticmethod
    def _escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    def add_table(self, title: str, frame: pd.DataFrame, max_rows: int = 40) -> None:
        self._lines.append(title)
        if frame.empty:
            self._lines.append("No data available")
            self._lines.append("")
            return
        header = " | ".join(str(col) for col in frame.columns)
        self._lines.append(header)
        preview = frame.head(max_rows)
        for _, row in preview.iterrows():
            cells: list[str] = []
            for value in row:
                if isinstance(value, (dt.datetime, dt.date)):
                    cells.append(value.isoformat())
                elif isinstance(value, float):
                    cells.append(f"{value:,.2f}")
                else:
                    cells.append(str(value))
            self._lines.append(" | ".join(cells))
        self._lines.append("")

    def write(self, path: Path) -> None:
        if not self._lines:
            self._lines.append("Weekly report")
        commands = ["BT", "/F1 12 Tf", "14 TL", "72 760 Td"]
        for index, line in enumerate(self._lines):
            escaped_line = self._escape(line)
            if index == 0:
                commands.append(f"({escaped_line}) Tj")
            else:
                commands.extend(["T*", f"({escaped_line}) Tj"])
        commands.append("ET")
        content = "\n".join(commands).encode("utf-8")
        objects: list[bytes] = []
        objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
        objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        )
        objects.append(b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"\nendstream")
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        with path.open("wb") as handle:
            handle.write(b"%PDF-1.4\n")
            offsets = [0]
            for index, obj in enumerate(objects, start=1):
                offsets.append(handle.tell())
                handle.write(f"{index} 0 obj\n".encode("ascii"))
                handle.write(obj)
                handle.write(b"\nendobj\n")
            xref_position = handle.tell()
            handle.write(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
            handle.write(b"0000000000 65535 f \n")
            for offset in offsets[1:]:
                handle.write(f"{offset:010} 00000 n \n".encode("ascii"))
            handle.write(b"trailer\n")
            handle.write(f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode("ascii"))
            handle.write(b"startxref\n")
            handle.write(f"{xref_position}\n".encode("ascii"))
            handle.write(b"%%EOF\n")


class QueryRunner(Protocol):
    """Protocol for objects capable of executing SQL queries."""

    def run(self, sql: str, params: Mapping[str, object] | None = None) -> pd.DataFrame:  # noqa: D401
        ...


class ObjectStorageClient(Protocol):
    """Protocol covering the subset of the S3 client that we require."""

    def upload_file(self, Filename: str, Bucket: str, Key: str) -> None:  # noqa: N803, D401
        ...


class SMTPClient(Protocol):
    """Minimal interface for SMTP interactions."""

    def send_message(self, message: EmailMessage) -> None:  # noqa: D401
        ...

    def quit(self) -> None:  # noqa: D401
        ...


@dataclass(slots=True)
class ScheduleConfig:
    """Configuration describing when the report should run."""

    weekday: int
    run_time: dt.time
    timezone: dt.tzinfo | None = None


@dataclass(slots=True)
class ReportConfig:
    """Runtime configuration for the weekly report job."""

    output_dir: Path
    schedule: ScheduleConfig
    s3_bucket: str | None = None
    s3_prefix: str = "reports/weekly"
    email_recipients: Sequence[str] = field(default_factory=list)
    email_sender: str | None = None
    email_subject: str = "Weekly Amazon performance report"
    max_top_candidates: int = 100
    anomaly_z_threshold: float = 2.5
    retry_policy: RetryPolicy = field(
        default_factory=lambda: RetryPolicy(
            max_attempts=3,
            base_delay_ms=500,
            jitter_ms=250,
            max_delay_ms=4000,
        )
    )
    ai_enabled: bool = False
    ai_summary_sheet_name: str = "AI 摘要"
    ai_keywords_sheet_name: str = "AI 关键词"
    ai_summary_column_label: str = "AI 摘要"
    ai_keywords_column_label: str = "AI 关键词"
    ai_summary_pdf_title: str = "AI Comment Summaries"
    ai_keywords_pdf_title: str = "AI Keyword Clusters"
    ai_placeholder_text: str = "待运营审核"


@dataclass(slots=True)
class WeeklyReportArtifacts:
    """Artifacts generated for a single run."""

    report_date: dt.date
    excel_path: Path
    pdf_path: Path
    top_candidates: pd.DataFrame
    category_deltas: pd.DataFrame
    anomalies: pd.DataFrame
    ai_comment_summaries: pd.DataFrame | None = None
    ai_keyword_clusters: pd.DataFrame | None = None


class WeeklyReportGenerator:
    """Service class that orchestrates data extraction and delivery."""

    def __init__(self, query_runner: QueryRunner, config: ReportConfig) -> None:
        self._query_runner = query_runner
        self._config = config
        self._last_run_at: dt.datetime | None = None
        self._config.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def last_run_at(self) -> dt.datetime | None:
        return self._last_run_at

    def _fetch_top_candidates(self, report_date: dt.date) -> pd.DataFrame:
        LOGGER.debug("Fetching top candidates for %s", report_date)
        sql = """
        SELECT *
        FROM vw_top_candidates_daily
        WHERE snapshot_date BETWEEN :start_date AND :end_date
        ORDER BY snapshot_date DESC, quality_rank ASC
        LIMIT :limit
        """
        params = {
            "start_date": report_date - dt.timedelta(days=6),
            "end_date": report_date,
            "limit": self._config.max_top_candidates,
        }
        frame = self._query_runner.run(sql, params)
        return frame.reset_index(drop=True)

    def _fetch_category_deltas(self, report_date: dt.date) -> pd.DataFrame:
        LOGGER.debug("Computing category deltas for %s", report_date)
        sql = """
        SELECT snapshot_date, category, SUM(predicted_revenue) AS predicted_revenue
        FROM vw_top_candidates_daily
        WHERE snapshot_date BETWEEN :start_date AND :end_date
        GROUP BY snapshot_date, category
        """
        params = {
            "start_date": report_date - dt.timedelta(days=13),
            "end_date": report_date,
        }
        frame = self._query_runner.run(sql, params)
        if frame.empty:
            return frame
        frame = frame.copy()
        frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"]).dt.date
        pivot = frame.pivot_table(
            index="category",
            columns="snapshot_date",
            values="predicted_revenue",
            aggfunc="sum",
        ).sort_index(axis=1)
        if pivot.shape[1] < 2:
            pivot["delta"] = 0.0
            pivot["delta_pct"] = pd.NA
            return pivot.reset_index()
        last_col, prev_col = pivot.columns[-1], pivot.columns[-2]
        pivot["delta"] = pivot[last_col] - pivot[prev_col]

        def _compute_delta_pct(row: pd.Series) -> object:
            previous_value = row[prev_col]
            if pd.isna(previous_value) or previous_value == 0:
                return pd.NA
            return (row[last_col] - previous_value) / previous_value

        pivot["delta_pct"] = pivot.apply(_compute_delta_pct, axis=1)
        return pivot.reset_index()

    def _fetch_anomalies(self, report_date: dt.date) -> pd.DataFrame:
        LOGGER.debug("Detecting anomalies for %s", report_date)
        sql = """
        SELECT snapshot_date, site, category, predicted_revenue
        FROM pred_rank_daily
        WHERE snapshot_date BETWEEN :start_date AND :end_date
        """
        params = {
            "start_date": report_date - dt.timedelta(days=6),
            "end_date": report_date,
        }
        frame = self._query_runner.run(sql, params)
        if frame.empty:
            return frame
        frame = frame.copy()
        frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"])
        grouped = frame.groupby(["site", "category"], group_keys=False)
        anomalies = []
        for (site, category), group in grouped:
            revenue = group["predicted_revenue"].astype(float)
            mean = revenue.mean()
            std = revenue.std(ddof=0)
            if std == 0:
                continue
            z_scores = (revenue - mean) / std
            mask = z_scores.abs() >= self._config.anomaly_z_threshold
            if mask.any():
                outliers = group.loc[mask, ["snapshot_date", "predicted_revenue"]]
                outliers = outliers.assign(
                    site=site,
                    category=category,
                    z_score=z_scores.loc[mask].round(3),
                )
                anomalies.append(outliers)
        if not anomalies:
            return pd.DataFrame(
                columns=["snapshot_date", "predicted_revenue", "site", "category", "z_score"]
            )
        result = pd.concat(anomalies, ignore_index=True)
        result["snapshot_date"] = pd.to_datetime(result["snapshot_date"]).dt.date
        return result

    def _prepare_ai_sections(
        self, top_candidates: pd.DataFrame
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        if not self._config.ai_enabled:
            return None, None

        required_columns = {"snapshot_date", "asin", "site", "ai_comment_summary", "ai_keyword_cluster"}
        missing_columns = required_columns - set(top_candidates.columns)
        if missing_columns:
            raise KeyError(f"Missing AI columns in top candidates frame: {sorted(missing_columns)}")

        def _format_section(source_column: str, column_label: str) -> pd.DataFrame:
            frame = top_candidates[["snapshot_date", "asin", "site", source_column]].copy()
            frame.rename(columns={source_column: column_label}, inplace=True)
            if frame.empty:
                return pd.DataFrame({column_label: [self._config.ai_placeholder_text]})
            frame[column_label] = frame[column_label].fillna(self._config.ai_placeholder_text)
            frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"]).dt.date
            frame = frame.drop_duplicates(ignore_index=True)
            return frame

        summaries = _format_section("ai_comment_summary", self._config.ai_summary_column_label)
        keywords = _format_section("ai_keyword_cluster", self._config.ai_keywords_column_label)

        if summaries.empty:
            summaries = pd.DataFrame({self._config.ai_summary_column_label: [self._config.ai_placeholder_text]})
        if keywords.empty:
            keywords = pd.DataFrame({self._config.ai_keywords_column_label: [self._config.ai_placeholder_text]})

        return summaries, keywords

    def _build_excel_report(
        self,
        report_date: dt.date,
        top_candidates: pd.DataFrame,
        category_deltas: pd.DataFrame,
        anomalies: pd.DataFrame,
        ai_comment_summaries: pd.DataFrame | None,
        ai_keyword_clusters: pd.DataFrame | None,
    ) -> Path:
        file_path = self._config.output_dir / f"weekly_report_{report_date.isoformat()}.xlsx"
        LOGGER.debug("Writing Excel report to %s", file_path)
        sheets = {
            "Top 100": top_candidates,
            "Category Deltas": category_deltas,
            "Anomalies": anomalies,
        }
        if ai_comment_summaries is not None:
            sheets[self._config.ai_summary_sheet_name] = ai_comment_summaries
        if ai_keyword_clusters is not None:
            sheets[self._config.ai_keywords_sheet_name] = ai_keyword_clusters
        builder = SimpleXLSXBuilder()
        builder.write(file_path, sheets)
        return file_path

    def _build_pdf_report(
        self,
        report_date: dt.date,
        top_candidates: pd.DataFrame,
        category_deltas: pd.DataFrame,
        anomalies: pd.DataFrame,
        ai_comment_summaries: pd.DataFrame | None,
        ai_keyword_clusters: pd.DataFrame | None,
    ) -> Path:
        file_path = self._config.output_dir / f"weekly_report_{report_date.isoformat()}.pdf"
        LOGGER.debug("Writing PDF report to %s", file_path)
        pdf = SimplePDFBuilder()
        pdf.add_table("Top 100 Candidates", top_candidates)
        pdf.add_table("Category Deltas", category_deltas)
        pdf.add_table("Anomalies", anomalies)
        if ai_comment_summaries is not None:
            pdf.add_table(self._config.ai_summary_pdf_title, ai_comment_summaries)
        if ai_keyword_clusters is not None:
            pdf.add_table(self._config.ai_keywords_pdf_title, ai_keyword_clusters)
        pdf.write(file_path)
        return file_path

    def generate_report(self, report_date: dt.date | None = None) -> WeeklyReportArtifacts:
        report_date = report_date or dt.date.today()
        LOGGER.info("Generating weekly report for %s", report_date)
        top_candidates = self._fetch_top_candidates(report_date)
        category_deltas = self._fetch_category_deltas(report_date)
        anomalies = self._fetch_anomalies(report_date)
        ai_comment_summaries: pd.DataFrame | None = None
        ai_keyword_clusters: pd.DataFrame | None = None
        if self._config.ai_enabled:
            try:
                ai_comment_summaries, ai_keyword_clusters = self._prepare_ai_sections(top_candidates)
            except Exception:  # noqa: BLE001
                LOGGER.exception("Failed to prepare AI sections; continuing with legacy report output")
                ai_comment_summaries = None
                ai_keyword_clusters = None
        excel_path = self._build_excel_report(
            report_date,
            top_candidates,
            category_deltas,
            anomalies,
            ai_comment_summaries,
            ai_keyword_clusters,
        )
        pdf_path = self._build_pdf_report(
            report_date,
            top_candidates,
            category_deltas,
            anomalies,
            ai_comment_summaries,
            ai_keyword_clusters,
        )
        artifacts = WeeklyReportArtifacts(
            report_date=report_date,
            excel_path=excel_path,
            pdf_path=pdf_path,
            top_candidates=top_candidates,
            category_deltas=category_deltas,
            anomalies=anomalies,
            ai_comment_summaries=ai_comment_summaries,
            ai_keyword_clusters=ai_keyword_clusters,
        )
        LOGGER.info(
            "Report generation complete for %s (Excel: %s, PDF: %s)",
            report_date,
            excel_path,
            pdf_path,
        )
        scheduled_dt = dt.datetime.combine(report_date, self._config.schedule.run_time)
        if self._config.schedule.timezone is not None:
            scheduled_dt = scheduled_dt.replace(tzinfo=self._config.schedule.timezone)
        self._last_run_at = scheduled_dt
        return artifacts

    def _deliver_to_s3(self, artifacts: WeeklyReportArtifacts, s3_client: ObjectStorageClient | None) -> None:
        if not s3_client or not self._config.s3_bucket:
            LOGGER.debug("Skipping S3 delivery (client or bucket not configured)")
            return
        LOGGER.info("Uploading weekly report artifacts to s3://%s/%s", self._config.s3_bucket, self._config.s3_prefix)
        prefix = self._config.s3_prefix.rstrip("/")
        for path in (artifacts.excel_path, artifacts.pdf_path):
            key = f"{prefix}/{path.name}"
            LOGGER.debug("Uploading %s to %s", path, key)
            s3_client.upload_file(str(path), Bucket=self._config.s3_bucket, Key=key)

    def _deliver_via_email(
        self,
        artifacts: WeeklyReportArtifacts,
        smtp_factory: Callable[[], SMTPClient] | None,
    ) -> None:
        if not self._config.email_recipients or not self._config.email_sender:
            LOGGER.debug("Skipping email delivery (sender or recipients not configured)")
            return
        if smtp_factory is None:
            raise ValueError("smtp_factory must be provided when email delivery is enabled")
        LOGGER.info("Sending weekly report email to %s", ", ".join(self._config.email_recipients))
        message = EmailMessage()
        message["From"] = self._config.email_sender
        message["To"] = ", ".join(self._config.email_recipients)
        message["Subject"] = self._config.email_subject
        message.set_content(
            "\n".join(
                [
                    f"Weekly report for {artifacts.report_date.isoformat()}",
                    "Attachments include the Excel workbook and PDF summary.",
                ]
            )
        )
        attachments = {
            artifacts.excel_path.name: artifacts.excel_path.read_bytes(),
            artifacts.pdf_path.name: artifacts.pdf_path.read_bytes(),
        }
        for filename, payload in attachments.items():
            maintype, subtype = ("application", "octet-stream")
            if filename.endswith(".pdf"):
                maintype, subtype = "application", "pdf"
            elif filename.endswith(".xlsx"):
                maintype, subtype = "application", "vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            message.add_attachment(payload, maintype=maintype, subtype=subtype, filename=filename)
        client = smtp_factory()
        try:
            client.send_message(message)
        finally:
            client.quit()

    def deliver_report(
        self,
        artifacts: WeeklyReportArtifacts,
        *,
        s3_client: ObjectStorageClient | None = None,
        smtp_factory: Callable[[], SMTPClient] | None = None,
    ) -> None:
        retry_policy = self._config.retry_policy
        for attempt in range(retry_policy.max_attempts):
            try:
                self._deliver_to_s3(artifacts, s3_client)
                self._deliver_via_email(artifacts, smtp_factory)
                return
            except Exception:  # noqa: BLE001
                LOGGER.exception("Delivery attempt %s failed", attempt + 1)
                if attempt >= retry_policy.max_attempts - 1:
                    raise
                sleep_with_backoff(attempt, retry_policy)

    def is_due(self, current_time: dt.datetime | None = None) -> bool:
        current_time = current_time or dt.datetime.now(tz=self._config.schedule.timezone)
        scheduled_time = current_time.replace(
            hour=self._config.schedule.run_time.hour,
            minute=self._config.schedule.run_time.minute,
            second=self._config.schedule.run_time.second,
            microsecond=self._config.schedule.run_time.microsecond,
        )
        days_ahead = (self._config.schedule.weekday - scheduled_time.weekday()) % 7
        scheduled_time = scheduled_time + dt.timedelta(days=days_ahead)
        if self._last_run_at is None:
            return current_time >= scheduled_time
        if current_time < scheduled_time:
            return False
        last_run_week = self._last_run_at.isocalendar()[1]
        current_week = scheduled_time.isocalendar()[1]
        return current_week != last_run_week or self._last_run_at < scheduled_time


def schedule_weekly_report(
    generator: WeeklyReportGenerator,
    *,
    now: dt.datetime | None = None,
    s3_client: ObjectStorageClient | None = None,
    smtp_factory: Callable[[], SMTPClient] | None = None,
) -> WeeklyReportArtifacts | None:
    """Run the weekly report if the schedule conditions are met."""

    current_time = now or dt.datetime.now(tz=generator._config.schedule.timezone)  # noqa: SLF001
    if not generator.is_due(current_time):
        LOGGER.debug("Weekly report not due at %s", current_time)
        return None
    artifacts = generator.generate_report(current_time.date())
    generator.deliver_report(artifacts, s3_client=s3_client, smtp_factory=smtp_factory)
    return artifacts


__all__ = [
    "ObjectStorageClient",
    "QueryRunner",
    "ReportConfig",
    "ScheduleConfig",
    "WeeklyReportArtifacts",
    "WeeklyReportGenerator",
    "schedule_weekly_report",
]
