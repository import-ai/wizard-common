from wizard_common.grimoire.agent.agent import UserQueryPreprocessor
from wizard_common.grimoire.entity.api import MessageAttrs, MessageDto


def call(call_id: str, name: str, arguments: str = "{}") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def assistant(
    content: str = "",
    tool_calls: list | None = None,
    reasoning_content: str | None = None,
) -> MessageDto:
    message: dict = {"role": "assistant", "content": content}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    if tool_calls:
        message["tool_calls"] = tool_calls
    return MessageDto(message=message)


def tool_result(
    call_id: str, content: str = "result", attrs: MessageAttrs | None = None
) -> MessageDto:
    return MessageDto(
        message={"role": "tool", "tool_call_id": call_id, "content": content},
        attrs=attrs,
    )


def user(content: str) -> MessageDto:
    return MessageDto(message={"role": "user", "content": content})


def test_drops_unknown_call_and_result_but_keeps_assistant_text():
    dtos = [
        user("read the file"),
        assistant("I read the file.", [call("c1", "read_resource")]),
        tool_result("c1", "secret file content"),
        assistant("Here is the summary."),
    ]

    view = UserQueryPreprocessor.compatible_history(dtos, {"private_search"})

    assert [dto.message for dto in view] == [
        {"role": "user", "content": "read the file"},
        {"role": "assistant", "content": "I read the file."},
        {"role": "assistant", "content": "Here is the summary."},
    ]


def test_keeps_available_call_and_its_result_untouched():
    dtos = [
        assistant("", [call("c1", "private_search", '{"query":"x"}')]),
        tool_result("c1", "hit"),
    ]

    view = UserQueryPreprocessor.compatible_history(dtos, {"private_search"})

    assert view[0] is dtos[0]
    assert view[1] is dtos[1]
    assert view[0].message["tool_calls"] == [
        call("c1", "private_search", '{"query":"x"}')
    ]


def test_mixed_assistant_message_keeps_only_available_calls_and_results():
    dtos = [
        assistant(
            "working",
            [call("a", "private_search"), call("b", "read_resource")],
        ),
        tool_result("a", "private hit"),
        tool_result("b", "file content"),
    ]

    view = UserQueryPreprocessor.compatible_history(dtos, {"private_search"})

    assert len(view) == 2
    assert view[0].message["content"] == "working"
    assert view[0].message["tool_calls"] == [call("a", "private_search")]
    assert view[1].message == {
        "role": "tool",
        "tool_call_id": "a",
        "content": "private hit",
    }


def test_drops_dangling_available_call_and_keeps_the_text():
    dtos = [
        assistant("partial answer", [call("c1", "private_search")]),
    ]

    view = UserQueryPreprocessor.compatible_history(dtos, {"private_search"})

    assert [dto.message for dto in view] == [
        {"role": "assistant", "content": "partial answer"}
    ]


def test_drops_dangling_call_without_text_or_reasoning():
    dtos = [
        assistant("", [call("c1", "private_search")]),
    ]

    view = UserQueryPreprocessor.compatible_history(dtos, {"private_search"})

    assert view == []


def test_drops_orphan_tool_result():
    dtos = [tool_result("orphan", "nobody asked")]

    view = UserQueryPreprocessor.compatible_history(dtos, {"private_search"})

    assert view == []


def test_keeps_reasoning_when_all_calls_are_filtered():
    dtos = [
        assistant(
            "",
            [call("c1", "read_resource")],
            reasoning_content="thinking about it",
        ),
    ]

    view = UserQueryPreprocessor.compatible_history(dtos, {"private_search"})

    assert [dto.message for dto in view] == [
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "thinking about it",
        }
    ]


def test_merge_search_keeps_only_the_merged_tool_name():
    dtos = [
        assistant("", [call("a", "private_search")]),
        tool_result("a", "private hit"),
        assistant("", [call("b", "search")]),
        tool_result("b", "merged hit"),
        assistant("", [call("c", "web_search")]),
        tool_result("c", "web hit"),
    ]

    view = UserQueryPreprocessor.compatible_history(dtos, {"search"})

    assert [dto.message for dto in view] == [
        assistant("", [call("b", "search")]).message,
        tool_result("b", "merged hit").message,
    ]


def test_empty_tool_set_removes_all_tool_protocol():
    dtos = [
        user("hello"),
        assistant("searching", [call("a", "private_search")]),
        tool_result("a", "hit"),
        assistant("bye"),
    ]

    view = UserQueryPreprocessor.compatible_history(dtos, set())

    assert [dto.message for dto in view] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "searching"},
        {"role": "assistant", "content": "bye"},
    ]


def test_does_not_mutate_input_dtos_or_attrs():
    attrs = MessageAttrs(
        citations=[],
        user_context={"selected_resources": ["doc"]},
        tool_call={"status": "success"},
    )
    dtos = [
        user("question"),
        assistant(
            "answer",
            [call("a", "private_search"), call("b", "read_resource")],
        ),
        tool_result("a", "hit", attrs=attrs),
        tool_result("b", "file"),
    ]
    before = [dto.model_dump() for dto in dtos]

    UserQueryPreprocessor.compatible_history(dtos, {"private_search"})

    assert [dto.model_dump() for dto in dtos] == before
    assert dtos[2].attrs is attrs
    assert attrs.tool_call == {"status": "success"}


def test_filtered_history_keeps_user_context_and_decision_handling():
    context = MessageAttrs(user_context={"selected_resources": ["doc"]})
    decision = MessageAttrs(tool_call={"decisions": [{"type": "approve"}]})
    dtos = [
        MessageDto(message={"role": "user", "content": "question"}, attrs=context),
        assistant("reading", [call("a", "read_resource")]),
        tool_result("a", "file"),
        MessageDto(message={"role": "user", "content": ""}, attrs=decision),
    ]

    view = UserQueryPreprocessor.compatible_history(dtos, set())
    messages = UserQueryPreprocessor.message_dtos_to_openai_messages(view)

    assert messages[0]["role"] == "user"
    assert "<user_context_json>" in messages[0]["content"]
    assert messages[1]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": ""}
