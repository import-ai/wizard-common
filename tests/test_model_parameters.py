import json
from unittest.mock import patch

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import ValidationError
from wizard_common.config import OpenAIConfig
from wizard_common.grimoire.config import GrimoireOpenAIConfig


@pytest.mark.parametrize(
    "suffix,expected",
    [
        ("", {}),
        ("?reasoning_effort=low", {"reasoning_effort": "low"}),
        ("?reasoning_effort=max", {"reasoning_effort": "max"}),
        ("?enable_thinking=true", {"enable_thinking": True}),
        ("?enable_thinking=false", {"enable_thinking": False}),
        (
            "?enable_thinking=false&reasoning_effort=low",
            {"enable_thinking": False, "reasoning_effort": "low"},
        ),
        (
            "?reasoning_effort=high&enable_thinking=true",
            {"enable_thinking": True, "reasoning_effort": "high"},
        ),
    ],
)
@pytest.mark.asyncio
async def test_model_settings_reach_wire(suffix, expected):
    config = GrimoireOpenAIConfig(
        default=OpenAIConfig(
            model="fallback", api_key="test", base_url="https://model.invalid/v1"
        ),
        vision=OpenAIConfig(model="ZHIPU/GLM-5.3-Flash" + suffix),
    ).get_config("vision")
    captured = []

    def handle(request):
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "ZHIPU/GLM-5.3-Flash",
                "choices": [],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http_client:
        async with AsyncOpenAI(
            api_key=config.api_key, base_url=config.base_url, http_client=http_client
        ) as client:
            with patch("wizard_common.config.AsyncOpenAI", return_value=client):
                await config.chat(
                    messages=[],
                    extra_body={"other": "preserved"},
                )
    assert captured == [
        {
            "model": "ZHIPU/GLM-5.3-Flash",
            "messages": [],
            "other": "preserved",
            **expected,
        }
    ]


@pytest.mark.parametrize(
    "model",
    [
        "",
        "?reasoning_effort=low",
        "model?",
        "model?reasoning_effort=",
        "model?reasoning_effort=typo",
        "model?enable_thinking=1",
        "model?enable_thinking=False",
        "model?unknown=low",
        "model?reasoning_effort=low&enable_thinking=invalid",
        "model?reasoning_effort=low&reasoning_effort=high",
        "model?enable_thinking=false&enable_thinking=false",
        "model?enable_thinking",
    ],
)
def test_invalid_model_fails_at_config_load(model):
    with pytest.raises(ValidationError):
        OpenAIConfig(model=model)


@pytest.mark.asyncio
async def test_explicit_model_override_clears_conflicting_defaults():
    config = OpenAIConfig(model="original?reasoning_effort=high", api_key="test")
    from unittest.mock import AsyncMock

    client = AsyncMock()
    client.__aenter__.return_value = client
    with patch("wizard_common.config.AsyncOpenAI", return_value=client):
        await config.chat(
            model="override?enable_thinking=false",
            messages=[],
            reasoning_effort="high",
            extra_body={"reasoning_effort": "max", "enable_thinking": True},
        )
    client.chat.completions.create.assert_awaited_once_with(
        model="override",
        messages=[],
        extra_body={"enable_thinking": False},
    )


class ClosableTransport(httpx.MockTransport):
    """Serves one SSE body lazily and, like a real connection pool, refuses to
    be read once the client owning it has been closed."""

    closed = False

    async def aclose(self):
        self.closed = True

    async def handle_async_request(self, request):
        chunk = {
            "id": "test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [
                {"index": 0, "delta": {"content": "hi"}, "finish_reason": None}
            ],
        }

        async def body():
            for event in (json.dumps(chunk), "[DONE]"):
                if self.closed:
                    raise httpx.ReadError("transport closed")
                yield f"data: {event}\n\n".encode()

        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body()
        )


@pytest.mark.asyncio
async def test_streamed_completion_stays_readable_after_chat_returns():
    config = OpenAIConfig(
        model="m", api_key="test", base_url="https://model.invalid/v1"
    )
    # Not entered as a context manager on purpose: `chat` owns the client, and
    # the stream is read only after `chat` has returned.
    client = AsyncOpenAI(
        api_key="test",
        base_url="https://model.invalid/v1",
        http_client=httpx.AsyncClient(transport=ClosableTransport(lambda r: None)),
    )
    with patch("wizard_common.config.AsyncOpenAI", return_value=client):
        stream = await config.chat(messages=[], stream=True)

    assert [part.choices[0].delta.content async for part in stream] == ["hi"]
    await client.close()


@pytest.mark.parametrize(
    "suffix", ["", "?enable_thinking=false", "?reasoning_effort=low"]
)
def test_base_agent_uses_configured_model_without_implicit_thinking(suffix):
    from jinja2 import Template
    from pydantic import BaseModel
    from wizard_common.agent.base import BaseAgent

    agent = BaseAgent(
        GrimoireOpenAIConfig(
            default=OpenAIConfig(
                model="default", api_key="test", base_url="https://model.invalid/v1"
            ),
            vision=OpenAIConfig(model="vision" + suffix),
            vision_thinking=OpenAIConfig(model="legacy-thinking"),
        ),
        BaseModel,
        str,
        system_prompt_template=Template("Classify the image"),
        model_size="vision",
    )
    assert agent.openai_config.model == "vision" + suffix
