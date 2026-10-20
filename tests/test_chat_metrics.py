import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from common.trace_info import TraceInfo
from openai.types.chat import ChatCompletionChunk
from wizard_common.config import OpenAIConfig
from wizard_common.grimoire.agent import agent as agent_module
from wizard_common.grimoire.agent.agent import Agent
from wizard_common.grimoire.agent.tps_calculator import TPSCalculator
from wizard_common.grimoire.config import GrimoireOpenAIConfig
from wizard_common.grimoire.entity.api import (
    AgentRequest,
    ChatBOSResponse,
    ChatEOSResponse,
    ChatMetricsResponse,
    MessageAttrs,
    MessageDto,
)


class ModelStream:
    def __init__(self, deltas, *, error=None, before_chunk=None, wait_after=None):
        self.deltas = deltas
        self.error = error
        self.close = AsyncMock()
        self.before_chunk = before_chunk
        self.wait_after = wait_after

    async def __aiter__(self):
        for delta in self.deltas:
            if self.before_chunk:
                self.before_chunk()
            yield ChatCompletionChunk.model_validate(
                {
                    "id": "test",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "test",
                    "choices": [{"index": 0, "delta": delta}],
                }
            )
        if self.wait_after:
            await self.wait_after.wait()
        if self.error:
            raise self.error


def make_agent():
    agent = Agent.__new__(Agent)
    agent.openai = GrimoireOpenAIConfig(
        default=OpenAIConfig(
            model="test", api_key="test", base_url="https://model.invalid/v1"
        )
    )
    return agent


def request():
    return AgentRequest(
        query="hi",
        namespace_id="ns",
        conversation_id="conv",
        messages=[
            MessageDto(message={"role": "user", "content": "hi"}, attrs=MessageAttrs())
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deltas, expected_tokens",
    [
        ([{"role": "assistant"}, {"content": "Hello"}, {"content": " world"}], 2),
        ([{"reasoning_content": "Think", "content": "Answer"}, {"content": "!"}], 2),
        (
            [
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "search", "arguments": "{"},
                        }
                    ]
                },
                {"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]},
            ],
            2,
        ),
    ],
)
async def test_chat_emits_metrics_for_generated_model_chunks(deltas, expected_tokens):
    stream = ModelStream(deltas)
    with patch.object(OpenAIConfig, "chat", return_value=stream):
        responses = [response async for response in make_agent().chat(messages=[])]

    metrics = [r for r in responses if isinstance(r, ChatMetricsResponse)]
    assert metrics, "Basic Agent must publish the metrics consumed by the Web footer"
    assert metrics[-1].tokens == expected_tokens
    assert metrics[-1].tps >= 0
    assert isinstance(responses[0], ChatBOSResponse)
    eos_index = next(
        i for i, r in enumerate(responses) if isinstance(r, ChatEOSResponse)
    )
    assert all(not isinstance(r, ChatMetricsResponse) for r in responses[eos_index:])
    stream.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_closing_chat_after_metrics_closes_the_model_stream(monkeypatch):
    now = 0.0

    def advance():
        nonlocal now
        now += 0.6

    monkeypatch.setattr(
        agent_module, "time", SimpleNamespace(time=time.time, monotonic=lambda: now)
    )
    model_stream = ModelStream(
        [{"content": "Hello"}, {"content": " world"}],
        before_chunk=advance,
        wait_after=asyncio.Event(),
    )
    with patch.object(OpenAIConfig, "chat", return_value=model_stream):
        stream = make_agent().chat(messages=[])
        try:
            async with asyncio.timeout(2):
                async for response in stream:
                    if isinstance(response, ChatMetricsResponse):
                        break
            assert isinstance(response, ChatMetricsResponse)
        finally:
            await stream.aclose()
    model_stream.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_metrics_are_throttled_and_share_the_request_total(monkeypatch):
    now = 0.0

    def advance():
        nonlocal now
        now += 0.4

    monkeypatch.setattr(
        agent_module, "time", SimpleNamespace(time=time.time, monotonic=lambda: now)
    )
    calculator = TPSCalculator()
    for _ in range(4):
        await calculator.count()
    model_stream = ModelStream([{"content": "x"}] * 5, before_chunk=advance)
    with patch.object(OpenAIConfig, "chat", return_value=model_stream):
        responses = [
            response
            async for response in make_agent().chat(
                messages=[], tps_calculator=calculator
            )
        ]
    metrics = [r for r in responses if isinstance(r, ChatMetricsResponse)]
    assert [r.tokens for r in metrics] == [7, 9]


@pytest.mark.asyncio
async def test_chat_preserves_transport_errors_and_closes_the_stream():
    model_stream = ModelStream([{"content": "partial"}], error=ValueError("broken"))
    with patch.object(OpenAIConfig, "chat", return_value=model_stream):
        with pytest.raises(ValueError, match="broken"):
            _ = [response async for response in make_agent().chat(messages=[])]
    model_stream.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_astream_accumulates_metrics_across_tool_rounds():
    agent = make_agent()
    agent.max_tool_rounds = 10
    agent.template_parser = SimpleNamespace(render_template=lambda _: "system")
    agent.system_prompt_template = None

    async def execute(messages, **kwargs):
        yield ChatBOSResponse(role="tool")
        yield MessageDto(
            message={"role": "tool", "content": "result", "tool_call_id": "call-1"}
        )
        yield ChatEOSResponse()

    agent.get_tool_executor = lambda *args, **kwargs: SimpleNamespace(
        config={}, tools=[{"type": "function"}], astream=execute
    )
    streams = [
        ModelStream(
            [
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "search", "arguments": "{}"},
                        }
                    ]
                }
            ]
        ),
        ModelStream([{"reasoning_content": "Think"}, {"content": "Answer"}]),
    ]
    with patch.object(OpenAIConfig, "chat", side_effect=streams):
        responses = [r async for r in agent.astream(TraceInfo(), request())]

    role = None
    totals = []
    for response in responses:
        if isinstance(response, ChatBOSResponse):
            role = response.role
        elif isinstance(response, ChatEOSResponse):
            role = None
        elif isinstance(response, ChatMetricsResponse):
            assert role == "assistant"
            totals.append(response.tokens)
    assert totals == [1, 3]
    for stream in streams:
        stream.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_closing_astream_closes_the_active_model_stream():
    agent = make_agent()
    agent.max_tool_rounds = 10
    agent.template_parser = SimpleNamespace(render_template=lambda _: "system")
    agent.system_prompt_template = None
    agent.get_tool_executor = lambda *args, **kwargs: SimpleNamespace(
        config={}, tools=[]
    )
    model_stream = ModelStream([{"content": "Hello"}], wait_after=asyncio.Event())
    with patch.object(OpenAIConfig, "chat", return_value=model_stream):
        stream = agent.astream(TraceInfo(), request())
        assert isinstance(await anext(stream), ChatBOSResponse)
        await stream.aclose()
    model_stream.close.assert_awaited_once()
