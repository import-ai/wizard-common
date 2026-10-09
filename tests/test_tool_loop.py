import asyncio
from importlib.resources import files

import pytest
from common.template_parser import TemplateParser
from common.trace_info import TraceInfo
from wizard_common.grimoire.agent.agent import Agent
from wizard_common.grimoire.agent.tool_loop import canonical_tool_key
from wizard_common.grimoire.entity.api import (
    AgentRequest,
    ChatBOSResponse,
    ChatEOSResponse,
    MessageAttrs,
    MessageDto,
)
from wizard_common.wizard.utils import streaming_response


def tool_call(call_id: str, arguments: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "private_search", "arguments": arguments},
    }


def assistant(content: str = "", tool_calls: list | None = None) -> dict:
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


class ScriptedChat:
    def __init__(self, replies: list[dict]):
        self.replies = replies
        self.tools: list = []

    async def __call__(self, messages, tools=None, **kwargs):
        self.tools.append(tools)
        yield ChatBOSResponse(role="assistant")
        yield MessageDto(message=self.replies.pop(0))
        yield ChatEOSResponse()


class RecordingExecutor:
    def __init__(self):
        self.calls = 0
        self.tools = [{"type": "function", "function": {"name": "private_search"}}]
        self.config = {}

    async def astream(self, messages, trace_info=None):
        self.calls += 1
        tool_call_id = messages[-1].message["tool_calls"][0]["id"]
        yield ChatBOSResponse(role="tool")
        yield MessageDto(
            message={
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": "hit",
            }
        )
        yield ChatEOSResponse()


def make_agent(max_tool_rounds: int = 10) -> Agent:
    agent = Agent.__new__(Agent)
    parser = TemplateParser(
        base_dir=str(files("wizard_common") / "resources" / "prompt_templates")
    )
    agent.template_parser = parser
    agent.system_prompt_template = parser.get_template("ask.j2")
    agent.tool_loop_fallback_template = parser.get_template("tool_loop_fallback.j2")
    agent.max_tool_rounds = max_tool_rounds
    return agent


def request() -> AgentRequest:
    return AgentRequest(
        query="hi",
        namespace_id="ns",
        conversation_id="conv",
        lang="简体中文",
        messages=[
            MessageDto(
                message={"role": "user", "content": "hi"},
                attrs=MessageAttrs(),
            )
        ],
    )


async def collect(agent: Agent):
    return [item async for item in agent.astream(TraceInfo(), request())]


def test_canonical_tool_key_ignores_object_key_order():
    left = canonical_tool_key("private_search", '{"b":1,"a":2}')
    right = canonical_tool_key("private_search", {"a": 2, "b": 1})
    assert left == right


async def test_duplicate_arguments_execute_once_then_drop_tools():
    agent = make_agent()
    executor = RecordingExecutor()
    agent.get_tool_executor = lambda *args, **kwargs: executor
    same = tool_call("1", '{"query":"判决书"}')
    again = tool_call("2", '{"query":"判决书"}')
    agent.chat = ScriptedChat(
        [
            assistant(tool_calls=[same]),
            assistant(tool_calls=[again]),
            assistant("done"),
        ]
    )

    await collect(agent)

    assert executor.calls == 1
    assert agent.chat.tools[0] is executor.tools
    assert agent.chat.tools[1] is executor.tools
    assert agent.chat.tools[2] is None


async def test_reordered_arguments_count_as_duplicate():
    agent = make_agent()
    executor = RecordingExecutor()
    agent.get_tool_executor = lambda *args, **kwargs: executor
    agent.chat = ScriptedChat(
        [
            assistant(tool_calls=[tool_call("1", '{"b":1,"a":2}')]),
            assistant(tool_calls=[tool_call("2", '{"a":2,"b":1}')]),
            assistant("done"),
        ]
    )

    await collect(agent)

    assert executor.calls == 1


async def test_round_limit_skips_the_next_tool_call():
    agent = make_agent(max_tool_rounds=1)
    executor = RecordingExecutor()
    agent.get_tool_executor = lambda *args, **kwargs: executor
    agent.chat = ScriptedChat(
        [
            assistant(tool_calls=[tool_call("1", '{"query":"one"}')]),
            assistant(tool_calls=[tool_call("2", '{"query":"two"}')]),
            assistant("done"),
        ]
    )

    await collect(agent)

    assert executor.calls == 1
    assert agent.chat.tools[-1] is None


async def test_empty_forced_reply_uses_template_not_unknown_error():
    agent = make_agent(max_tool_rounds=1)
    executor = RecordingExecutor()
    agent.get_tool_executor = lambda *args, **kwargs: executor
    agent.chat = ScriptedChat(
        [
            assistant(tool_calls=[tool_call("1", '{"query":"one"}')]),
            assistant(tool_calls=[tool_call("2", '{"query":"two"}')]),
            assistant("", tool_calls=[tool_call("3", '{"query":"three"}')]),
        ]
    )

    chunks = await collect(agent)
    text = " ".join(
        str(getattr(getattr(chunk, "message", None), "content", "") or "")
        for chunk in chunks
    )

    assert "Unknown error" not in text
    assert "这次不能再调用工具了" in text
    assert executor.calls == 1


async def test_cancelling_the_stream_stops_further_model_calls():
    agent = make_agent()
    executor = RecordingExecutor()
    agent.get_tool_executor = lambda *args, **kwargs: executor
    started = asyncio.Event()

    async def chat(messages, tools=None, **kwargs):
        started.set()
        await asyncio.Event().wait()
        yield ChatEOSResponse()

    agent.chat = chat
    stream = agent.astream(TraceInfo(), request())
    task = asyncio.create_task(anext(stream))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await stream.aclose()
    assert executor.calls == 0


async def test_client_disconnect_closes_the_sse_generator():
    import anyio

    closed = anyio.Event()
    first_chunk = anyio.Event()

    async def generate():
        try:
            yield {"response_type": "delta", "message": {"content": "hi"}}
            await anyio.Event().wait()
        finally:
            closed.set()

    async def receive():
        await first_chunk.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            first_chunk.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 123),
        "server": ("test", 80),
    }

    with anyio.fail_after(2):
        await streaming_response(generate())(scope, receive, send)
        await closed.wait()
    assert closed.is_set()
