"""OpenAI LLM provider implementation."""

from typing import List, Dict, AsyncIterator

from openai import AsyncOpenAI

from providers.base import BaseLLMProvider


class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI LLM provider using the AsyncOpenAI client."""

    def __init__(self, api_key: str = None):
        """Initialize the OpenAI provider."""
        if not api_key:
            import os
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY in your environment or pass api_key= to the constructor."
            )
        self._client = AsyncOpenAI(api_key=api_key)

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens from OpenAI."""
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    async def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.3,
    ) -> str:
        """Single-shot generation via OpenAI chat completions."""
        response = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
