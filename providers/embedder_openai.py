"""OpenAI embedding provider implementation."""

from typing import List

from openai import AsyncOpenAI

from providers.base import BaseEmbedder


# Map known models to their output dimensions
_MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedding provider using the AsyncOpenAI client."""

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        dimension: int = None,
    ):
        """Initialize the OpenAI embedder."""
        import os

        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required for embeddings. "
                "Set OPENAI_API_KEY in your environment or pass api_key= to the constructor."
            )

        if not model:
            model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

        if dimension is not None:
            self._dimension = dimension
        else:
            # Auto-detect from model name, fallback to env or 3072
            self._dimension = _MODEL_DIMENSIONS.get(
                model,
                int(os.getenv("EMBEDDING_DIM", "3072")),
            )

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts via OpenAI API."""
        if not texts:
            return []

        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )

        # Sort by index to preserve input order
        embeddings = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in embeddings]

    @property
    def dimension(self) -> int:
        """Return the embedding dimension for this model."""
        return self._dimension
