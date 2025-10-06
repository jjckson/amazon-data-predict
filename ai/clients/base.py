"""Abstract client definitions for external AI providers."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import (
    ChatGenerationRequest,
    EmbeddingRequest,
    EmbeddingResponse,
    GenerationRequest,
    GenerationResponse,
)


class LLMClient(ABC):
    """Base interface for interacting with large language model providers."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Invoke a text completion style model."""

        # TODO(manual): 接入真实供应商 API
        raise NotImplementedError

    @abstractmethod
    def chat(self, request: ChatGenerationRequest) -> GenerationResponse:
        """Invoke a chat completion style model."""

        raise NotImplementedError


class EmbeddingClient(ABC):
    """Base interface for generating vector embeddings."""

    @abstractmethod
    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Invoke an embedding model and return vector outputs."""

        # TODO(manual): 接入真实供应商 API
        raise NotImplementedError
