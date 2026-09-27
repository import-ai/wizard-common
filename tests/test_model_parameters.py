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
        ("", {"enable_thinking": False}),
        ("?reasoning_effort=low", {"reasoning_effort": "low"}),
        ("?reasoning_effort=max", {"reasoning_effort": "max"}),
        ("?enable_thinking=true", {"enable_thinking": True}),
        ("?enable_thinking=false", {"enable_thinking": False}),
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
                    extra_body={"enable_thinking": False, "other": "preserved"},
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
        "model?reasoning_effort=low&enable_thinking=true",
        "model?reasoning_effort=low&reasoning_effort=high",
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


@pytest.mark.parametrize(
    "suffix", ["", "?enable_thinking=false", "?reasoning_effort=low"]
)
def test_explicit_suffix_precedes_legacy_thinking_model(suffix):
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
        enable_thinking=True,
    )
    assert agent.openai_config.model == (
        "vision" + suffix if suffix else "legacy-thinking"
    )
