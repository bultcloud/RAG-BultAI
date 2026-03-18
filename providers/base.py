"""Abstract base classes for the RAG provider plugin system."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, AsyncIterator, Optional


class BaseLLMProvider(ABC):
    """Abstract base for LLM backends."""

    @abstractmethod
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.3,
    ) -> str:
        """Single-shot text generation for internal tasks."""
        pass


class BaseRetriever(ABC):
    """Abstract base for retrieval strategies."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        project_id: int,
        user_id: int,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query."""
        pass


class BaseChunker(ABC):
    """Abstract base for text chunking strategies."""

    @abstractmethod
    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata."""
        pass


class BaseReranker(ABC):
    """Abstract base for reranking retrieved chunks."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Rerank chunks by relevance to query."""
        pass


class BaseEmbedder(ABC):
    """Abstract base for embedding providers."""

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass
