"""Keyword clustering workflow powered by embeddings."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..clients.base import EmbeddingClient
from ..models import EmbeddingRequest

# TODO(manual): 由合规团队维护词表
_STOPWORDS = {
    "a",
    "an",
    "and",
    "or",
    "the",
    "to",
    "with",
    "for",
}

# TODO(manual): 由合规团队维护词表
_SENSITIVE_WHITELIST = {
    "amazon",
    "prime",
}


def _normalise(keyword: str) -> str:
    return keyword.strip()


def _filter_keywords(keywords: Sequence[str]) -> List[str]:
    """Filter out stop words while respecting the sensitive whitelist."""

    filtered: List[str] = []
    seen = set()
    for keyword in keywords:
        normalised = _normalise(keyword)
        if not normalised:
            continue
        key_lower = normalised.lower()
        if key_lower not in _SENSITIVE_WHITELIST and key_lower in _STOPWORDS:
            continue
        if key_lower in seen:
            continue
        seen.add(key_lower)
        filtered.append(normalised)
    return filtered


def _default_output_dir(root: Path, report_date: date) -> Path:
    return root / report_date.strftime("%Y%m%d") / "ai"


def _initial_centroids(vectors: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    indices = rng.choice(len(vectors), size=k, replace=False)
    return vectors[indices]


def _assign_clusters(vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(vectors[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(distances, axis=1)


def _recompute_centroids(vectors: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    centroids = np.zeros((k, vectors.shape[1]), dtype=vectors.dtype)
    for idx in range(k):
        mask = labels == idx
        if not np.any(mask):
            continue
        centroids[idx] = vectors[mask].mean(axis=0)
    return centroids


def _kmeans(vectors: np.ndarray, k: int, *, max_iter: int = 50, seed: int = 42) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if len(vectors) < k:
        raise ValueError("Number of vectors must be >= k")

    rng = np.random.default_rng(seed)
    centroids = _initial_centroids(vectors, k, rng)

    for _ in range(max_iter):
        labels = _assign_clusters(vectors, centroids)
        new_centroids = _recompute_centroids(vectors, labels, k)
        # Handle empty clusters by reinitialising them randomly
        for idx in range(k):
            if not np.any(labels == idx):
                replacement = rng.choice(len(vectors))
                new_centroids[idx] = vectors[replacement]
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    else:
        labels = _assign_clusters(vectors, centroids)

    return labels


def cluster_keywords(
    keywords: Sequence[str],
    *,
    embedding_client: EmbeddingClient,
    num_clusters: int = 5,
    model: Optional[str] = None,
    window_start: Optional[datetime] = None,
    window_end: Optional[datetime] = None,
    report_date: Optional[date] = None,
    output_root: Path | str = Path("reports/offline_eval"),
    extra_parameters: Optional[Mapping[str, Any]] = None,
    random_seed: int = 42,
) -> Path:
    """Cluster the provided ``keywords`` using embeddings and persist the result."""

    report_date = report_date or date.today()
    output_root = Path(output_root)
    output_dir = _default_output_dir(output_root, report_date)
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered_keywords = _filter_keywords(keywords)
    if not filtered_keywords:
        raise ValueError("No keywords available for clustering after filtering")

    extra_parameters = extra_parameters or {}
    request = EmbeddingRequest(
        inputs=filtered_keywords,
        model=model,
        extra_parameters=extra_parameters,
    )
    response = embedding_client.embed(request)

    vectors = np.asarray(response.embeddings, dtype=float)
    if vectors.ndim != 2:
        raise ValueError("Embeddings must be a 2D array")

    k = min(num_clusters, len(filtered_keywords))
    labels = _kmeans(vectors, k, seed=random_seed)

    clusters: List[Dict[str, Any]] = []
    for cluster_index in range(k):
        members = [
            {
                "keyword": filtered_keywords[item_index],
                "embedding": vectors[item_index].tolist(),
            }
            for item_index, label in enumerate(labels)
            if label == cluster_index
        ]
        if not members:
            continue
        centroid = np.mean([member["embedding"] for member in members], axis=0).tolist()
        clusters.append(
            {
                "cluster_id": cluster_index,
                "keywords": members,
                "centroid": centroid,
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": model or response.model,
        "window": {
            "start": window_start.isoformat() if window_start else None,
            "end": window_end.isoformat() if window_end else None,
        },
        "num_clusters": k,
        "filtered_keyword_count": len(filtered_keywords),
        "clusters": clusters,
    }

    output_path = output_dir / "keyword_clusters.json"
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    return output_path
