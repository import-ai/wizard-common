import asyncio
from importlib.resources import files

import pytest
from common.template_parser import TemplateParser
from common.trace_info import TraceInfo

from wizard_common.grimoire.agent.agent import Agent
from wizard_common.grimoire.agent.loop_guard import LoopGuardConfig
from wizard_common.grimoire.agent.tool_executor import UNKNOWN_TOOL_ERROR, ToolExecutor
from wizard_common.grimoire.agent.tool_loop import canonical_tool_key
from wizard_common.grimoire.entity.api import (
    AgentRequest,
    ChatBOSResponse,
    ChatDeltaResponse,
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


def named_tool_call(call_id: str, name: str, arguments: str = "{}") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
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
    agent.loop_guard_config = LoopGuardConfig.for_basic()
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


async def collect_request(agent: Agent, agent_request: AgentRequest):
    return [item async for item in agent.astream(TraceInfo(), agent_request)]


async def empty_search(**kwargs) -> list:
    return []


def search_executor() -> ToolExecutor:
    return ToolExecutor(
        {
            "private_search": {
                "name": "private_search",
                "schema": {
                    "type": "function",
                    "function": {"name": "private_search"},
                },
                "func": empty_search,
            }
        }
    )


class RecordingChat:
    def __init__(self, replies: list[dict]):
        self.replies = replies
        self.requests: list[dict] = []

    async def __call__(self, messages, tools=None, **kwargs):
        self.requests.append({"messages": messages, "tools": tools})
        yield ChatBOSResponse(role="assistant")
        yield MessageDto(message=self.replies.pop(0))
        yield ChatEOSResponse()


def request_with(messages: list[MessageDto]) -> AgentRequest:
    return AgentRequest(
        query="hi",
        namespace_id="ns",
        conversation_id="conv",
        lang="简体中文",
        messages=messages,
    )


def user_message(content: str) -> MessageDto:
    return MessageDto(
        message={"role": "user", "content": content}, attrs=MessageAttrs()
    )


def join_text(chunks) -> str:
    return " ".join(
        str(getattr(getattr(chunk, "message", None), "content", "") or "")
        for chunk in chunks
    )


def test_canonical_tool_key_ignores_object_key_order():
    left = canonical_tool_key("private_search", '{"b":1,"a":2}')
    right = canonical_tool_key("private_search", {"a": 2, "b": 1})
    assert left == right


async def test_repeated_read_executes_again_and_keeps_tools():
    agent = make_agent()
    executor = RecordingExecutor()
    agent.get_tool_executor = lambda *args, **kwargs: executor
    same = tool_call("1", '{"query":"判决书"}')
    again = tool_call("2", '{"query":"判决书"}')
    agent.chat = RecordingChat(
        [
            assistant(tool_calls=[same]),
            assistant(tool_calls=[again]),
            assistant("done"),
        ]
    )

    await collect(agent)

    assert executor.calls == 2
    assert agent.chat.requests[1]["tools"] is executor.tools
    assert agent.chat.requests[2]["tools"] is executor.tools
    second_round_tools = [
        message
        for message in agent.chat.requests[2]["messages"]
        if message.get("role") == "tool"
    ]
    assert any(
        "Note: this is repeat 1" in (message.get("content") or "")
        for message in second_round_tools
    )


async def test_reordered_arguments_share_the_signature_and_execute_again():
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

    assert executor.calls == 2


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
    last_bos = max(
        index
        for index, chunk in enumerate(chunks)
        if isinstance(chunk, ChatBOSResponse) and chunk.role == "assistant"
    )
    eos = next(
        index
        for index in range(last_bos + 1, len(chunks))
        if isinstance(chunks[index], ChatEOSResponse)
    )
    window = chunks[last_bos + 1 : eos]

    assert "Unknown error" not in text
    assert any(
        isinstance(chunk, ChatDeltaResponse)
        and "这次不能再调用工具了" in (chunk.message.content or "")
        for chunk in window
    )
    assert executor.calls == 1


async def test_empty_completed_reply_is_not_replaced_with_the_tool_limit():
    agent = make_agent()
    executor = RecordingExecutor()
    agent.get_tool_executor = lambda *args, **kwargs: executor
    agent.chat = ScriptedChat([assistant("")])

    chunks = await collect(agent)
    text = " ".join(
        str(getattr(getattr(chunk, "message", None), "content", "") or "")
        for chunk in chunks
    )

    assert "这次不能再调用工具了" not in text
    assert executor.calls == 0


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


def history_with_unknown_tool() -> list[MessageDto]:
    return [
        user_message("read the file"),
        MessageDto(
            message={
                "role": "assistant",
                "content": "I read the file earlier.",
                "tool_calls": [named_tool_call("h1", "read_resource")],
            }
        ),
        MessageDto(
            message={
                "role": "tool",
                "tool_call_id": "h1",
                "content": "secret file body",
            }
        ),
        user_message("summarize it"),
    ]


async def test_history_unknown_tool_is_removed_from_the_model_view():
    agent = make_agent()
    executor = search_executor()
    agent.get_tool_executor = lambda *args, **kwargs: executor
    agent.chat = RecordingChat([assistant("final")])

    await collect_request(agent, request_with(history_with_unknown_tool()))

    first = agent.chat.requests[0]["messages"]
    flattened = str(first)
    assert "read_resource" not in flattened
    assert "secret file body" not in flattened
    assert any(
        message.get("content") == "I read the file earlier." for message in first
    )


async def test_available_history_tool_call_still_reaches_the_model():
    agent = make_agent()
    executor = search_executor()
    agent.get_tool_executor = lambda *args, **kwargs: executor
    agent.chat = RecordingChat([assistant("final")])
    history = [
        user_message("search"),
        MessageDto(
            message={
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call("h1", '{"query":"old"}')],
            }
        ),
        MessageDto(
            message={"role": "tool", "tool_call_id": "h1", "content": "old hit"}
        ),
        user_message("again"),
    ]

    await collect_request(agent, request_with(history))

    first = agent.chat.requests[0]["messages"]
    assert any(
        message.get("tool_calls") and message["tool_calls"][0]["id"] == "h1"
        for message in first
    )
    assert any(message.get("tool_call_id") == "h1" for message in first)


async def test_dangling_history_tool_call_is_removed_from_the_model_view():
    agent = make_agent()
    executor = search_executor()
    agent.get_tool_executor = lambda *args, **kwargs: executor
    agent.chat = RecordingChat([assistant("final")])
    history = [
        MessageDto(
            message={
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call("h1", '{"query":"old"}')],
            }
        ),
        user_message("again"),
    ]

    await collect_request(agent, request_with(history))

    first = agent.chat.requests[0]["messages"]
    assert not any(message.get("tool_calls") for message in first)


async def test_current_round_unknown_tool_failure_reaches_the_next_model_call():
    agent = make_agent()
    executor = search_executor()
    agent.get_tool_executor = lambda *args, **kwargs: executor
    agent.chat = RecordingChat(
        [
            assistant(tool_calls=[named_tool_call("u1", "read_resource")]),
            assistant("final answer"),
        ]
    )

    chunks = await collect(agent)

    assert agent.chat.requests[1]["tools"] is executor.tools
    second = agent.chat.requests[1]["messages"]
    assistant_calls = [
        message for message in second if message.get("role") == "assistant"
    ]
    assert assistant_calls[-1]["tool_calls"][0]["function"]["name"] == "read_resource"
    tool_messages = [message for message in second if message.get("role") == "tool"]
    assert tool_messages[-1]["tool_call_id"] == "u1"
    assert tool_messages[-1]["content"] == f"Error: {UNKNOWN_TOOL_ERROR}"
    assert any(
        isinstance(chunk, ChatDeltaResponse)
        and chunk.attrs
        and (chunk.attrs.tool_call or {}).get("status") == "failed"
        for chunk in chunks
    )
    assert agent.chat.replies == []


async def test_repeated_unknown_tool_blocks_then_stops_before_fallback():
    agent = make_agent()
    executor = search_executor()
    agent.get_tool_executor = lambda *args, **kwargs: executor
    agent.chat = RecordingChat(
        [
            assistant(tool_calls=[named_tool_call("u1", "read_resource")]),
            assistant(tool_calls=[named_tool_call("u2", "read_resource")]),
            assistant(tool_calls=[named_tool_call("u3", "read_resource")]),
            assistant(""),
        ]
    )

    chunks = await collect(agent)

    assert len(agent.chat.requests) == 4
    assert agent.chat.requests[1]["tools"] is executor.tools
    assert agent.chat.requests[2]["tools"] is executor.tools
    assert agent.chat.requests[3]["tools"] is None
    stopped = [
        chunk
        for chunk in chunks
        if isinstance(chunk, ChatDeltaResponse)
        and chunk.attrs
        and (chunk.attrs.tool_call or {}).get("status") == "stopped"
    ]
    assert [chunk.message.tool_call_id for chunk in stopped] == ["u2", "u3"]
    assert all(chunk.message.content for chunk in stopped)
    blocked_tools = [
        message
        for message in agent.chat.requests[2]["messages"]
        if message.get("role") == "tool"
    ]
    assert any(
        "was NOT executed" in (message.get("content") or "")
        for message in blocked_tools
    )
    assert "相同参数的调用上一次已经失败" in join_text(chunks)


class MultiRecordingExecutor:
    def __init__(self):
        self.calls = 0
        self.tools = [{"type": "function", "function": {"name": "private_search"}}]
        self.config = {}

    async def astream(self, messages, trace_info=None):
        self.calls += 1
        yield ChatBOSResponse(role="tool")
        for tool_call in messages[-1].message["tool_calls"]:
            yield MessageDto(
                message={
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": "hit",
                }
            )
        yield ChatEOSResponse()


async def test_blocked_duplicate_does_not_drop_the_other_calls_in_the_batch():
    agent = make_agent()
    executor = MultiRecordingExecutor()
    agent.get_tool_executor = lambda *args, **kwargs: executor
    agent.chat = RecordingChat(
        [
            assistant(tool_calls=[tool_call("a1", '{"query":"a"}')]),
            assistant(tool_calls=[tool_call("a2", '{"query":"a"}')]),
            assistant(
                tool_calls=[
                    tool_call("a3", '{"query":"a"}'),
                    tool_call("b1", '{"query":"b"}'),
                ]
            ),
            assistant("done"),
        ]
    )

    chunks = await collect(agent)

    assert executor.calls == 3
    assert agent.chat.requests[3]["tools"] is executor.tools
    stopped = [
        chunk
        for chunk in chunks
        if isinstance(chunk, ChatDeltaResponse)
        and chunk.attrs
        and (chunk.attrs.tool_call or {}).get("status") == "stopped"
    ]
    assert [chunk.message.tool_call_id for chunk in stopped] == ["a3"]
    tool_messages = [
        message
        for message in agent.chat.requests[3]["messages"]
        if message.get("role") == "tool"
    ]
    assert {message.get("tool_call_id") for message in tool_messages} == {
        "a1",
        "a2",
        "a3",
        "b1",
    }
    assert "这次不能再调用工具了" not in join_text(chunks)


async def test_history_filter_does_not_rewrite_request_messages():
    agent = make_agent()
    executor = search_executor()
    agent.get_tool_executor = lambda *args, **kwargs: executor
    agent.chat = RecordingChat([assistant("final")])
    agent_request = request_with(history_with_unknown_tool())
    before = [dto.model_dump() for dto in agent_request.messages]

    await collect_request(agent, agent_request)

    assert [dto.model_dump() for dto in agent_request.messages[: len(before)]] == before
