from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel, Field


class OpenAIConfig(BaseModel):
    api_key: str = Field(default=None)
    model: str = Field(default=None)
    base_url: str = Field(default=None)

    async def chat(
        self, *, model: str = None, **kwargs
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return await client.chat.completions.create(
            **(kwargs | {"model": model or self.model})
        )
