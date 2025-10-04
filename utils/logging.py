"""Project-wide logging utilities."""
from __future__ import annotations

import json
import logging
from logging import Logger
from typing import Optional


def configure_logger(name: str, level: str = "INFO", json_output: bool = True) -> Logger:
    """Configure and return a project logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logging_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(logging_level)
    handler = logging.StreamHandler()

    if json_output:
        class JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:  # noqa: D401
                payload = {
                    "level": record.levelname,
                    "name": record.name,
                    "message": record.getMessage(),
                }
                if record.exc_info:
                    payload["exc_info"] = self.formatException(record.exc_info)
                return json.dumps(payload)

        handler.setFormatter(JsonFormatter())
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger(name: Optional[str] = None) -> Logger:
    """Return a configured logger."""
    logger_name = name or "asin_pipeline"
    return configure_logger(logger_name)
