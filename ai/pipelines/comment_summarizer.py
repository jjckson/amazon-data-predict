"""Utilities for summarising customer comments with LLMs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ..clients.base import LLMClient
from ..models import GenerationRequest, GenerationResponse


@dataclass(frozen=True)
class CommentSummary:
    """Container for a single batch summary."""

    batch_index: int
    comments: Sequence[str]
    prompt: str
    response: GenerationResponse

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the summary to a JSON-friendly mapping."""

        return {
            "batch_index": self.batch_index,
            "comments": list(self.comments),
            "prompt": self.prompt,
            "summary": self.response.text.strip(),
            "model": self.response.model,
            "raw": self.response.raw,
        }


PROMPT_TEMPLATE = (
    "You are preparing concise summaries for the customer feedback review.\n"
    "Summarise the key themes and actionable points from the following comments:\n"
    "{comments}\n\n"
    "Format the answer as bullet points grouped by theme."
)


def _batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    """Yield batches of ``items`` respecting the requested ``batch_size``."""

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _build_prompt(comments: Sequence[str]) -> str:
    """Construct the prompt payload for the provided ``comments``."""

    formatted = "\n".join(f"- {comment}" for comment in comments)
    return PROMPT_TEMPLATE.format(comments=formatted)


def _default_output_dir(root: Path, report_date: date) -> Path:
    """Resolve the standard reporting directory for a run."""

    return root / report_date.strftime("%Y%m%d") / "ai"


def generate_comment_summaries(
    comments: Sequence[str],
    *,
    llm_client: LLMClient,
    batch_size: int = 10,
    model: Optional[str] = None,
    window_start: Optional[datetime] = None,
    window_end: Optional[datetime] = None,
    report_date: Optional[date] = None,
    output_root: Path | str = Path("reports/offline_eval"),
    extra_parameters: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Generate summaries for ``comments`` using the provided ``llm_client``.

    The resulting report is persisted under ``reports/offline_eval/{date}/ai`` and
    the path to the generated JSON file is returned.
    """

    # TODO(manual): 设置正式 prompt 模板与安全过滤。

    report_date = report_date or date.today()
    output_root = Path(output_root)
    output_dir = _default_output_dir(output_root, report_date)
    output_dir.mkdir(parents=True, exist_ok=True)

    extra_parameters = extra_parameters or {}
    summaries: List[CommentSummary] = []

    for batch_index, chunk in enumerate(_batched(list(comments), batch_size)):
        if not chunk:
            continue
        prompt = _build_prompt(chunk)
        request = GenerationRequest(
            prompt=prompt,
            model=model,
            extra_parameters=extra_parameters,
        )
        response = llm_client.generate(request)
        summaries.append(
            CommentSummary(
                batch_index=batch_index,
                comments=tuple(chunk),
                prompt=prompt,
                response=response,
            )
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": model or (summaries[0].response.model if summaries else None),
        "window": {
            "start": window_start.isoformat() if window_start else None,
            "end": window_end.isoformat() if window_end else None,
        },
        "batch_size": batch_size,
        "summary_count": len(summaries),
        "summaries": [summary.to_dict() for summary in summaries],
    }

    output_path = output_dir / "comment_summaries.json"
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    return output_path
