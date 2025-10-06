"""Data models for AI enhancement layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Mapping, Optional, Sequence


@dataclass
class GenerationRequest:
    """Parameters for a single LLM generation request."""

    prompt: str
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 1.0
    extra_parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """Represents a single message within a chat-style interaction."""

    role: str
    content: str
    name: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResponse:
    """Result returned from a language model generation."""

    text: str
    model: Optional[str] = None
    usage_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ChatGenerationRequest:
    """Parameters for multi-message chat generations."""

    messages: Sequence[Message]
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 1.0
    extra_parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingRequest:
    """Parameters for generating vector embeddings."""

    inputs: Sequence[str]
    model: Optional[str] = None
    extra_parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResponse:
    """Result returned from an embedding model invocation."""

    embeddings: List[List[float]]
    model: Optional[str] = None
    usage_tokens: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    raw: Mapping[str, Any] = field(default_factory=dict)
