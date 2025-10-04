"""Traffic splitting utilities for deterministic experiment allocation.

The module exposes :class:`TrafficSplitter` which assigns ASINs (or any
identifier) deterministically to experiment variants based on hashing.  The
splitter supports per-category overrides, configurable allocation ratios and
keeps an in-memory audit log that can optionally be persisted using the Python
``logging`` module.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class AuditLogEntry:
    """Represents a single assignment captured for auditing."""

    experiment: str
    asin: str
    category: Optional[str]
    variant: str
    score: float
    allocations: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize the entry as a JSON string."""

        payload = {
            "experiment": self.experiment,
            "asin": self.asin,
            "category": self.category,
            "variant": self.variant,
            "score": self.score,
            "allocations": dict(self.allocations),
            "metadata": dict(self.metadata),
        }
        return json.dumps(payload, sort_keys=True)


class TrafficSplitter:
    """Deterministically split traffic across variants using hashing.

    Parameters
    ----------
    experiment_name:
        Identifier used when generating hashes.  Changing the name will result
        in different assignments even if every other parameter is identical.
    allocations:
        Mapping of variant name to allocation weight (between 0 and 1).  The
        weights are normalised automatically and must sum to ``1``.
    category_allocations:
        Optional overrides for specific categories.  The keys should be the
        category names and the values follow the same format as ``allocations``.
    salt:
        Optional salt mixed into the hash to discourage reverse engineering of
        the allocation logic.
    audit_logger:
        Optional :class:`logging.Logger` instance.  When provided, every
        assignment is logged at ``INFO`` level with the JSON payload of the
        :class:`AuditLogEntry`.
    """

    def __init__(
        self,
        experiment_name: str,
        allocations: Mapping[str, float],
        *,
        category_allocations: Optional[Mapping[str, Mapping[str, float]]] = None,
        salt: str = "",
        audit_logger: Optional[logging.Logger] = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.salt = salt
        self._default_allocations = self._normalise_allocations(allocations)
        self._category_allocations: Dict[str, Dict[str, float]] = {}
        if category_allocations:
            for category, alloc in category_allocations.items():
                self._category_allocations[category] = self._normalise_allocations(alloc)
        self._audit_logger = audit_logger or self._build_default_logger()
        self._audit_log: List[AuditLogEntry] = []

    @staticmethod
    def _build_default_logger() -> logging.Logger:
        logger = logging.getLogger(f"{__name__}.audit")
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
        return logger

    @staticmethod
    def _normalise_allocations(allocations: Mapping[str, float]) -> Dict[str, float]:
        if not allocations:
            raise ValueError("At least one variant allocation must be provided")
        total = float(sum(allocations.values()))
        if total <= 0:
            raise ValueError("Allocation weights must sum to a positive number")
        normalised = {variant: weight / total for variant, weight in allocations.items()}
        if any(weight < 0 for weight in normalised.values()):
            raise ValueError("Allocation weights must be non-negative")
        # Avoid floating point rounding issues by re-normalising cumulative sum.
        discrepancy = abs(sum(normalised.values()) - 1.0)
        if discrepancy > 1e-6:
            raise ValueError("Allocation weights must sum to 1.0")
        return normalised

    def allocations_for(self, category: Optional[str]) -> Dict[str, float]:
        """Return the allocation map for the provided category."""

        return self._category_allocations.get(category or "", self._default_allocations)

    def assign(
        self,
        asin: str,
        *,
        category: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Assign an ASIN to a variant.

        Parameters
        ----------
        asin:
            The identifier to hash.  Any string that uniquely identifies a unit
            of traffic works (e.g. customer id, request id, ...).
        category:
            Optional category for which category-specific allocations should be
            applied.
        metadata:
            Optional metadata to attach to the audit entry (for example
            timestamp, country, or request context).
        """

        metadata = dict(metadata or {})
        active_allocations = self.allocations_for(category)
        variant, score = self._select_variant(asin, category, active_allocations)
        entry = AuditLogEntry(
            experiment=self.experiment_name,
            asin=asin,
            category=category,
            variant=variant,
            score=score,
            allocations=active_allocations,
            metadata=metadata,
        )
        self._audit_log.append(entry)
        if self._audit_logger:
            self._audit_logger.info(entry.to_json())
        return variant

    def _select_variant(
        self,
        asin: str,
        category: Optional[str],
        allocations: Mapping[str, float],
    ) -> Tuple[str, float]:
        score = self._hash_to_unit_interval(asin, category)
        cumulative = 0.0
        last_variant = None
        for variant, weight in allocations.items():
            last_variant = variant
            cumulative += weight
            if score < cumulative:
                return variant, score
        assert last_variant is not None  # pragma: no cover - defensive
        return last_variant, score

    def _hash_to_unit_interval(self, asin: str, category: Optional[str]) -> float:
        key = "|".join(
            part for part in [self.experiment_name, category or "", asin, self.salt] if part
        )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        integer = int(digest, 16)
        max_int = 2**256 - 1
        return integer / max_int

    def get_audit_trail(self) -> List[AuditLogEntry]:
        """Return a copy of the audit log collected so far."""

        return list(self._audit_log)

    def clear_audit_trail(self) -> None:
        """Remove all cached audit records."""

        self._audit_log.clear()
