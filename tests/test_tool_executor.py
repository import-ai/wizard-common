import pytest
from common.trace_info import TraceInfo

from wizard_common.grimoire.agent.tool_executor import (
    INVALID_ARGUMENTS_ERROR,
    TOOL_FAILED_ERROR,
    UNKNOWN_TOOL_ERROR,
    ToolExecutor,
    error_tool_wrapper,
)
from wizard_common.grimoire.entity.api import (
    ChatBOSResponse,
    ChatDeltaResponse,
    ChatEOSResponse,
    MessageDto,
)
from wizard_common.grimoire.entity.chunk import Chunk, ChunkType, ResourceChunkRetrieval


def call(call_id: str, name: str, arguments: str = "{}") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def config(func, name: str = "private_search") -> dict:
    return {
        name: {
            "name": name,
            "schema": {"type": "function", "function": {"name": name}},
            "func": func,
        }
    }


async def search(**kwargs) -> list:
    return [
        ResourceChunkRetrieval(
            chunk=Chunk(
                title="Doc",
                resource_id="resource-1",
                text="matched text",
                chunk_type=ChunkType.snippet,
                parent_id="parent-1",
                start_index=0,
                end_index=12,
            )
        )
    ]


def assistant_message(*tool_calls: dict) -> MessageDto:
    return MessageDto(
        message={
            "role": "assistant",
            "content": "",
            "tool_calls": list(tool_calls),
        }
    )


async def collect(executor: ToolExecutor, *tool_calls: dict) -> list:
    dtos = [assistant_message(*tool_calls)]
    return [chunk async for chunk in executor.astream(dtos, TraceInfo())]


def assert_failed_result(chunks: list, call_id: str, error: str) -> None:
    assert [type(chunk) for chunk in chunks] == [
        ChatBOSResponse,
        ChatDeltaResponse,
        MessageDto,
        ChatEOSResponse,
    ]
    delta, message_dto = chunks[1], chunks[2]
    assert delta.message.tool_call_id == call_id
    assert delta.message.content == f"Error: {error}"
    assert delta.attrs.tool_call == {"status": "failed", "error": error}
    assert message_dto.message == {
        "role": "tool",
        "tool_call_id": call_id,
        "content": f"Error: {error}",
    }
    assert message_dto.attrs.tool_call == {"status": "failed", "error": error}


async def test_unknown_tool_becomes_a_failed_result_and_does_not_raise():
    chunks = await collect(ToolExecutor(config={}), call("c1", "read_resource"))

    assert_failed_result(chunks, "c1", UNKNOWN_TOOL_ERROR)


@pytest.mark.parametrize("arguments", ["{not json", "[1, 2]", "null"])
async def test_invalid_arguments_become_a_failed_result(arguments):
    chunks = await collect(
        ToolExecutor(config=config(search)),
        call("c1", "private_search", arguments),
    )

    assert_failed_result(chunks, "c1", INVALID_ARGUMENTS_ERROR)


async def test_registered_non_search_tool_is_refused():
    async def tool(**kwargs):
        raise AssertionError("must not be invoked")

    chunks = await collect(
        ToolExecutor(config=config(tool, name="edit_resource")),
        call("c1", "edit_resource"),
    )

    assert_failed_result(chunks, "c1", UNKNOWN_TOOL_ERROR)


async def test_registered_tool_failure_reports_then_reraises():
    async def failing(**kwargs):
        raise RuntimeError("internal detail: secret stack")

    executor = ToolExecutor(config=config(failing))
    chunks: list = []
    with pytest.raises(RuntimeError, match="internal detail"):
        async for chunk in executor.astream(
            [assistant_message(call("c1", "private_search", '{"query":"x"}'))],
            TraceInfo(),
        ):
            chunks.append(chunk)

    assert [type(chunk) for chunk in chunks] == [
        ChatBOSResponse,
        ChatDeltaResponse,
        MessageDto,
    ]
    assert chunks[1].message.content == f"Error: {TOOL_FAILED_ERROR}"
    assert chunks[1].attrs.tool_call == {
        "status": "failed",
        "error": TOOL_FAILED_ERROR,
    }
    assert "internal detail" not in str(chunks[1].message.content)
    assert "internal detail" not in str(chunks[1].attrs)
    assert chunks[2].message["content"] == f"Error: {TOOL_FAILED_ERROR}"


async def test_registered_tool_assertion_failure_is_reported_and_reraises():
    async def broken(**kwargs):
        return "not a list of retrievals"

    executor = ToolExecutor(config=config(broken))
    chunks: list = []
    with pytest.raises(AssertionError):
        async for chunk in executor.astream(
            [assistant_message(call("c1", "private_search"))], TraceInfo()
        ):
            chunks.append(chunk)

    assert [type(chunk) for chunk in chunks] == [
        ChatBOSResponse,
        ChatDeltaResponse,
        MessageDto,
    ]
    assert chunks[1].attrs.tool_call["status"] == "failed"


async def test_successful_search_keeps_citations_and_success_status():
    executor = ToolExecutor(config=config(search))
    chunks = await collect(
        executor,
        call("c1", "private_search", '{"query":"matched"}'),
    )

    assert [type(chunk) for chunk in chunks] == [
        ChatBOSResponse,
        ChatDeltaResponse,
        MessageDto,
        ChatEOSResponse,
    ]
    delta, message_dto = chunks[1], chunks[2]
    assert delta.attrs.tool_call == {"status": "success"}
    assert len(delta.attrs.citations) == 1
    assert delta.attrs.citations[0].id.startswith("C1-")
    assert message_dto.attrs.citations[0].id == delta.attrs.citations[0].id
    assert "cite_marker=" in message_dto.message["content"]


async def test_failed_call_does_not_advance_the_citation_counter():
    executor = ToolExecutor(config=config(search))
    chunks = await collect(
        executor,
        call("c1", "read_resource"),
        call("c2", "private_search", '{"query":"matched"}'),
    )

    assert [type(chunk) for chunk in chunks] == [
        ChatBOSResponse,
        ChatDeltaResponse,
        MessageDto,
        ChatEOSResponse,
        ChatBOSResponse,
        ChatDeltaResponse,
        MessageDto,
        ChatEOSResponse,
    ]
    failed_delta, success_delta = chunks[1], chunks[5]
    assert failed_delta.attrs.tool_call["status"] == "failed"
    assert success_delta.attrs.tool_call == {"status": "success"}
    assert success_delta.attrs.citations[0].id.startswith("C1-")


def test_error_tool_wrapper_shape():
    message_dto = error_tool_wrapper("c1", TOOL_FAILED_ERROR)

    assert message_dto.message == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": f"Error: {TOOL_FAILED_ERROR}",
    }
    assert message_dto.attrs.tool_call == {
        "status": "failed",
        "error": TOOL_FAILED_ERROR,
    }
