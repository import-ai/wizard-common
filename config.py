from urllib.parse import parse_qsl

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel, Field, field_validator


def parse_model_name(value: str | None) -> tuple[str | None, dict]:
    """Separate a model ID from explicitly configured thinking parameters."""
    if value is None:
        return None, {}
    model, separator, query = value.partition("?")
    if not model.strip():
        raise ValueError("Model name must not be empty")
    if not separator:
        return model, {}
    pairs = parse_qsl(query, keep_blank_values=True, strict_parsing=True)
    if not pairs or len(dict(pairs)) != len(pairs):
        raise ValueError(
            "Model suffix must contain nonempty, unique thinking parameters"
        )
    parameters = {}
    for key, parameter in pairs:
        if key == "enable_thinking" and parameter in ("true", "false"):
            parameters["extra_body"] = {key: parameter == "true"}
        elif key == "reasoning_effort" and parameter in (
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
            "ultra",
        ):
            parameters[key] = parameter
        else:
            raise ValueError("Unsupported thinking parameter or value in model suffix")
    return model, parameters


class OpenAIConfig(BaseModel):
    api_key: str = Field(default=None)
    model: str = Field(default=None)
    base_url: str = Field(default=None)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value):
        parse_model_name(value)
        return value

    async def chat(
        self, *, model: str = None, **kwargs
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        model_name, parameters = parse_model_name(model or self.model)
        if parameters:
            # Explicit model settings override caller defaults, including extra_body.
            extra_body = dict(kwargs.get("extra_body") or {})
            for key in ("enable_thinking", "reasoning_effort"):
                kwargs.pop(key, None)
                extra_body.pop(key, None)
            extra_body.update(parameters.get("extra_body", {}))
            kwargs["extra_body"] = extra_body
            if "reasoning_effort" in parameters:
                kwargs["reasoning_effort"] = parameters["reasoning_effort"]
        # A streamed response is consumed by the caller after this returns, so the
        # client must stay open here: closing it would close the stream's transport.
        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return await client.chat.completions.create(**(kwargs | {"model": model_name}))
