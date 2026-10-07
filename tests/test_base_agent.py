import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from jinja2 import Template
from pydantic import BaseModel, ConfigDict, ValidationError

from wizard_common.agent.base import BaseAgent
from wizard_common.config import OpenAIConfig
from wizard_common.grimoire.config import GrimoireOpenAIConfig


class Input(BaseModel):
    text: str = "Original content"


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str


def make_agent(output_class=Output):
    return BaseAgent(
        GrimoireOpenAIConfig(
            default=OpenAIConfig(
                model="test", api_key="test", base_url="https://model.invalid/v1"
            )
        ),
        Input,
        output_class,
        system_prompt_template=Template("Generate a title"),
    )


def make_stream(text):
    stream = AsyncMock()
    stream.__aiter__.return_value = [
        SimpleNamespace(choices=[]),
        *[
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=part))]
            )
            for part in (text[:5], text[5:])
        ],
    ]
    return stream


@pytest.mark.asyncio
@pytest.mark.parametrize("failures", [0, 1, 2, 3])
async def test_validation_repair_is_bounded_and_preserves_context(failures):
    invalid = [
        json.dumps({"title": "Test", "priority": f"Attempt {i}"})
        for i in range(failures)
    ]
    responses = invalid + ([] if failures == 3 else ['{"title":"Repaired"}'])
    streams = [make_stream(text) for text in responses]
    requests = []

    async def chat(**kwargs):
        requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return streams[len(requests) - 1]

    with patch.object(OpenAIConfig, "chat", side_effect=chat):
        if failures == 3:
            with pytest.raises(ValidationError) as exc:
                await make_agent().ainvoke({})
            assert exc.value.errors()[0]["loc"] == ("priority",)
            assert exc.value.errors()[0]["input"] == "Attempt 2"
        else:
            assert (await make_agent().ainvoke({})).title == "Repaired"

    assert len(requests) == len(responses)
    for i, request in enumerate(requests):
        assert "extra_body" not in request
        assert "reasoning_effort" not in request
        assert request["messages"][:2] == requests[0]["messages"]
        assert len(request["messages"]) == 2 + 2 * i
        if i:
            assert request["messages"][-2] == {
                "role": "assistant",
                "content": invalid[i - 1],
            }
            feedback = request["messages"][-1]["content"]
            assert '"priority"' in feedback and '"extra_forbidden"' in feedback
            assert '"input"' not in feedback and '"ctx"' not in feedback
            assert "complete corrected JSON" in feedback
    for stream in streams:
        stream.aclose.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("response", ['{"title": []}', "{}"])
async def test_required_fields_and_types_still_fail(response):
    with patch.object(
        OpenAIConfig, "chat", side_effect=lambda **_: make_stream(response)
    ) as chat:
        with pytest.raises(ValidationError):
            await make_agent().ainvoke({})
    assert chat.await_count == 3


@pytest.mark.asyncio
async def test_transport_and_json_errors_are_not_retried():
    with patch.object(
        OpenAIConfig, "chat", side_effect=RuntimeError("transport failed")
    ) as chat:
        with pytest.raises(RuntimeError, match="transport failed"):
            await make_agent().ainvoke({})
    chat.assert_awaited_once()
    stream = make_stream("not JSON")
    with patch.object(OpenAIConfig, "chat", return_value=stream) as chat:
        with pytest.raises(ValueError, match="valid JSON"):
            await make_agent().ainvoke({})
    chat.assert_awaited_once()
    stream.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_streaming_and_text_outputs_do_not_retry():
    for output_class in (Output, str):
        stream = make_stream("not JSON")
        with patch.object(OpenAIConfig, "chat", return_value=stream) as chat:
            agent = make_agent(output_class)
            if output_class is str:
                assert await agent.ainvoke({}) == "not JSON"
            else:
                assert "".join([part async for part in agent.astream({})]) == "not JSON"
        chat.assert_awaited_once()
        stream.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_closing_stream_closes_transport():
    stream = make_stream('{"title":"Test"}')
    with patch.object(OpenAIConfig, "chat", return_value=stream):
        response = make_agent().astream({})
        await anext(response)
        await response.aclose()
    stream.aclose.assert_awaited_once()
