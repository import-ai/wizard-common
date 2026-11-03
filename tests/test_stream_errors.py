import json
import logging
from unittest.mock import MagicMock

import httpx
import openai
import pytest
from common.trace_info import TraceInfo
from opentelemetry import trace
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

from wizard_common.grimoire.entity.api import (
    AgentRequest,
    ChatDeltaResponse,
    MessageDto,
)
from wizard_common.wizard import errors as errors_module
from wizard_common.wizard.errors import (
    StreamErrorInfo,
    build_stream_error_response,
    classify_error,
    render_error_message,
)
from wizard_common.wizard.utils import stream_wrapper

INPUT_MESSAGE_ZH = "本次请求未通过模型服务的安全审核（可能含历史消息），请修改相关内容后重试，或新建对话。"
INPUT_MESSAGE_EN = (
    "This request did not pass the model service's safety review (earlier messages "
    "may be involved). Edit the relevant content and try again, or start a new "
    "conversation."
)
OUTPUT_MESSAGE_ZH = "生成的回复未通过模型服务的安全审核，请新建对话或修改后重试。"
OUTPUT_MESSAGE_EN = (
    "The generated reply did not pass the model service's safety review. Start a new "
    "conversation or edit the relevant content and try again."
)
GENERIC_MESSAGE_ZH = "网络错误，请稍后重试。"
GENERIC_MESSAGE_EN = "Network error. Please try again later."

UPSTREAM_REQUEST_ID = "chatcmpl-decdf808-387c-99b5-a933-cb896438ab9f"
TRACE_ID_HEX = "e39794801c0dd0a7705cb71f49c2464b"
TRACE_ID = int(TRACE_ID_HEX, 16)


def status_error(
    status: int, *, body=None, payload=None, headers=None, message="error"
):
    request = httpx.Request("POST", "https://model.example/v1/chat/completions")
    if payload is None:
        payload = {
            "id": UPSTREAM_REQUEST_ID,
            "error": body if body is not None else {"message": "upstream error"},
        }
    response = httpx.Response(status, request=request, json=payload, headers=headers)
    error_body = payload.get("error") if isinstance(payload, dict) else None
    classes = {
        400: openai.BadRequestError,
        401: openai.AuthenticationError,
        403: openai.PermissionDeniedError,
        404: openai.NotFoundError,
        409: openai.ConflictError,
        422: openai.UnprocessableEntityError,
        429: openai.RateLimitError,
        500: openai.InternalServerError,
        503: openai.InternalServerError,
    }
    return classes.get(status, openai.APIStatusError)(
        message, response=response, body=error_body
    )


def review_error(message="Input text data may contain inappropriate content."):
    return status_error(
        400,
        body={
            "code": "data_inspection_failed",
            "type": "data_inspection_failed",
            "message": message,
        },
    )


def wrapped_error(inner: Exception) -> Exception:
    try:
        raise inner
    except Exception:
        try:
            raise RuntimeError("wrapper")
        except RuntimeError as outer:
            return outer


def caught(error: Exception) -> Exception:
    try:
        raise error
    except Exception as exc:
        return exc


def make_request(**kwargs) -> AgentRequest:
    values = {"query": "hello", "namespace_id": "ns-1", "conversation_id": "conv-1"}
    values.update(kwargs)
    return AgentRequest(**values)


def build(error, **kwargs):
    kwargs.setdefault("trace_info", TraceInfo(request_id="req-1"))
    try:
        raise error
    except Exception as exc:
        return build_stream_error_response(exc, **kwargs)


class FakeStream:
    def __init__(self, deltas=(), error=None, aclose_error=None):
        self.deltas = list(deltas)
        self.error = error
        self.aclose_error = aclose_error
        self.closed = False

    async def __aiter__(self):
        for delta in self.deltas:
            yield delta
        if self.error is not None:
            raise self.error

    async def aclose(self):
        self.closed = True
        if self.aclose_error is not None:
            raise self.aclose_error


def delta(content="hello"):
    return ChatDeltaResponse.model_validate({"message": {"content": content}})


async def collect(stream, request, trace_info):
    return [chunk async for chunk in stream_wrapper(request, stream, trace_info)]


@pytest.mark.parametrize(
    "message",
    [
        "Input text data may contain inappropriate content.",
        "  input text DATA may contain inappropriate content.",
        None,
        "Input text data may contain output content that is inappropriate.",
    ],
)
def test_input_review_classification(message):
    body = {"code": "data_inspection_failed", "type": "data_inspection_failed"}
    if message is not None:
        body["message"] = message
    info = classify_error(status_error(400, body=body))
    assert info.code == "content_review_rejected"
    assert info.retryable is False
    assert info.upstream_status == 400
    assert info.upstream_code == "data_inspection_failed"
    assert info.upstream_request_id == UPSTREAM_REQUEST_ID


@pytest.mark.parametrize(
    "message",
    [
        "Output text data may contain inappropriate content.",
        "  Output text DATA may contain inappropriate content.",
        "\n\toutput text rejected",
    ],
)
def test_output_review_classification(message):
    info = classify_error(review_error(message))
    assert info.code == "output_content_review_rejected"
    assert info.retryable is False
    assert info.upstream_status == 400
    assert info.upstream_code == "data_inspection_failed"


def test_other_bad_request():
    info = classify_error(
        status_error(400, body={"code": "invalid_parameter", "message": "bad"})
    )
    assert info.code == "upstream_bad_request"
    assert info.retryable is False


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (401, "upstream_unauthorized"),
        (403, "upstream_forbidden"),
        (404, "upstream_not_found"),
        (422, "upstream_unprocessable"),
    ],
)
def test_non_retryable_status_codes(status, code):
    info = classify_error(status_error(status))
    assert info.code == code
    assert info.retryable is False
    assert info.upstream_status == status


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (409, "upstream_conflict"),
        (429, "upstream_rate_limited"),
        (500, "upstream_server_error"),
        (503, "upstream_server_error"),
    ],
)
def test_retryable_status_codes(status, code):
    info = classify_error(status_error(status))
    assert info.code == code
    assert info.retryable is True
    assert info.upstream_status == status


def test_timeout_by_exception_and_status():
    request = httpx.Request("POST", "https://model.example/v1/chat/completions")
    timeout = classify_error(openai.APITimeoutError(request=request))
    assert timeout.code == "upstream_timeout"
    assert timeout.retryable is True
    assert timeout.upstream_status is None

    by_status = classify_error(status_error(408))
    assert by_status.code == "upstream_timeout"
    assert by_status.retryable is True
    assert by_status.upstream_status == 408


def test_connection_error():
    request = httpx.Request("POST", "https://model.example/v1/chat/completions")
    info = classify_error(openai.APIConnectionError(request=request))
    assert info.code == "upstream_connection_failed"
    assert info.retryable is True
    assert info.upstream_status is None


def test_streaming_api_error_with_data_inspection():
    request = httpx.Request("POST", "https://model.example/v1/chat/completions")
    error = openai.APIError(
        "stream error",
        request=request,
        body={
            "code": "data_inspection_failed",
            "type": "data_inspection_failed",
            "message": "Output text data may contain inappropriate content.",
        },
    )
    info = classify_error(error)
    assert info.code == "output_content_review_rejected"
    assert info.upstream_status is None
    assert info.upstream_code == "data_inspection_failed"


def test_stateless_api_error():
    request = httpx.Request("POST", "https://model.example/v1/chat/completions")
    info = classify_error(openai.APIError("boom", request=request, body=None))
    assert info.code == "upstream_api_error"
    assert info.retryable is True
    assert info.upstream_status is None


def test_unknown_status_code_is_upstream_api_error():
    info = classify_error(status_error(418))
    assert info.code == "upstream_api_error"
    assert info.retryable is True
    assert info.upstream_status == 418


@pytest.mark.parametrize("error", [ValueError("bad"), RuntimeError("bad")])
def test_non_api_error(error):
    info = classify_error(error)
    assert info.code == "internal_error"
    assert info.retryable is True


def test_wrapped_api_error_is_unwrapped():
    info = classify_error(wrapped_error(review_error()))
    assert info.code == "content_review_rejected"
    assert info.upstream_status == 400
    assert info.upstream_request_id == UPSTREAM_REQUEST_ID


def test_context_api_error_is_unwrapped():
    try:
        raise review_error()
    except Exception:
        try:
            raise RuntimeError("wrapper")
        except RuntimeError as outer:
            error = outer
    info = classify_error(error)
    assert info.code == "content_review_rejected"


def test_unwrap_depth_is_bounded():
    inner = review_error()
    outer = inner
    for _ in range(6):
        outer = wrapped_error(outer)
    info = classify_error(outer)
    assert info.code == "internal_error"


def test_upstream_request_id_fallbacks():
    body = {"code": "data_inspection_failed", "message": "Input text"}
    from_json_request_id = status_error(
        400, body=body, payload={"error": body, "request_id": "json-request-id"}
    )
    assert classify_error(from_json_request_id).upstream_request_id == "json-request-id"

    from_header = status_error(
        400, body=body, payload={"error": body}, headers={"x-request-id": "header-id"}
    )
    assert classify_error(from_header).upstream_request_id == "header-id"

    request = httpx.Request("POST", "https://model.example/v1/chat/completions")
    response = httpx.Response(400, request=request, content=b"not json")
    no_json = openai.BadRequestError("bad", response=response, body=body)
    assert classify_error(no_json).upstream_request_id is None


@pytest.mark.parametrize(
    ("lang", "expected"),
    [
        (None, INPUT_MESSAGE_ZH),
        ("简体中文", INPUT_MESSAGE_ZH),
        ("Klingon", INPUT_MESSAGE_ZH),
        ("English", INPUT_MESSAGE_EN),
    ],
)
def test_language_selection(lang, expected):
    info = StreamErrorInfo(code="content_review_rejected", retryable=False)
    assert render_error_message(info, lang) == expected


def test_review_copy_is_exact():
    input_info = StreamErrorInfo(code="content_review_rejected", retryable=False)
    output_info = StreamErrorInfo(
        code="output_content_review_rejected", retryable=False
    )
    assert render_error_message(input_info, None) == INPUT_MESSAGE_ZH
    assert render_error_message(input_info, "English") == INPUT_MESSAGE_EN
    assert render_error_message(output_info, None) == OUTPUT_MESSAGE_ZH
    assert render_error_message(output_info, "English") == OUTPUT_MESSAGE_EN


@pytest.mark.parametrize(
    "code",
    [
        "upstream_bad_request",
        "upstream_unauthorized",
        "upstream_forbidden",
        "upstream_not_found",
        "upstream_conflict",
        "upstream_unprocessable",
        "upstream_rate_limited",
        "upstream_server_error",
        "upstream_timeout",
        "upstream_connection_failed",
        "upstream_api_error",
        "internal_error",
    ],
)
def test_non_review_codes_use_generic_copy(code):
    info = StreamErrorInfo(code=code, retryable=True)
    message = render_error_message(info, None)
    assert message == GENERIC_MESSAGE_ZH
    assert render_error_message(info, "English") == GENERIC_MESSAGE_EN
    assert message not in ("Inappropriate content", "Unknown error")


def test_visible_message_excludes_ids_and_upstream_details():
    error = review_error()
    context = SpanContext(
        trace_id=TRACE_ID,
        span_id=0x3B0AB02715AD66EC,
        is_remote=False,
        trace_flags=TraceFlags.SAMPLED,
    )
    with trace.use_span(NonRecordingSpan(context), end_on_exit=False):
        response = build(error, lang="简体中文")
    assert response.message == INPUT_MESSAGE_ZH
    for leaked in (
        "req-1",
        TRACE_ID_HEX,
        "chatcmpl-",
        "data_inspection_failed",
        "HTTP 400",
        "Input text data may contain inappropriate content.",
        "trace ID",
        "追踪 ID",
    ):
        assert leaked not in response.message
    assert response.code == "content_review_rejected"
    assert response.retryable is False
    assert response.request_id == "req-1"
    assert response.trace_id == TRACE_ID_HEX
    assert response.upstream_request_id == UPSTREAM_REQUEST_ID
    assert response.upstream_status == 400
    assert response.upstream_code == "data_inspection_failed"


def test_request_id_falls_back_to_trace_id():
    context = SpanContext(
        trace_id=TRACE_ID,
        span_id=0x3B0AB02715AD66EC,
        is_remote=False,
        trace_flags=TraceFlags.SAMPLED,
    )
    with trace.use_span(NonRecordingSpan(context), end_on_exit=False):
        response = build(ValueError("bad"), trace_info=None)
    assert response.request_id == TRACE_ID_HEX
    assert response.trace_id == TRACE_ID_HEX
    assert response.message == GENERIC_MESSAGE_ZH


def test_response_dump_omits_unset_fields():
    response = build(ValueError("bad"), trace_info=None)
    dumped = response.model_dump(exclude_none=True)
    assert dumped["response_type"] == "error"
    assert dumped["message"] == GENERIC_MESSAGE_ZH
    assert dumped["code"] == "internal_error"
    assert dumped["retryable"] is True
    assert "request_id" not in dumped
    assert "trace_id" not in dumped
    assert "upstream_status" not in dumped
    assert "upstream_code" not in dumped
    assert "upstream_request_id" not in dumped


def test_error_response_dump_keeps_structured_fields_separate():
    response = build(review_error())
    dumped = response.model_dump(exclude_none=True)
    assert set(dumped) == {
        "response_type",
        "message",
        "code",
        "retryable",
        "request_id",
        "upstream_request_id",
        "upstream_status",
        "upstream_code",
    }
    assert dumped["message"] == INPUT_MESSAGE_ZH


def test_span_attributes(monkeypatch):
    span = MagicMock()
    span.get_span_context.return_value = SpanContext(
        trace_id=0, span_id=0, is_remote=False
    )
    monkeypatch.setattr(errors_module.trace, "get_current_span", lambda: span)

    response = build(
        review_error(),
        request=make_request(query="SENTINEL-CHAT-TEXT"),
        trace_info=TraceInfo(request_id="req-1"),
    )

    span.record_exception.assert_called_once()
    span.set_status.assert_called_once()
    attributes = dict(
        (call.args[0], call.args[1]) for call in span.set_attribute.call_args_list
    )
    assert attributes["error_code"] == "content_review_rejected"
    assert attributes["error.retryable"] is False
    assert attributes["error.request_id"] == "req-1"
    assert attributes["error.upstream_status"] == 400
    assert attributes["error.upstream_request_id"] == UPSTREAM_REQUEST_ID
    assert "SENTINEL-CHAT-TEXT" not in json.dumps(attributes)
    assert response.trace_id is None


def test_log_payload_is_whitelisted(caplog, monkeypatch):
    span = MagicMock()
    span.get_span_context.return_value = SpanContext(
        trace_id=0, span_id=0, is_remote=False
    )
    monkeypatch.setattr(errors_module.trace, "get_current_span", lambda: span)
    request = make_request(
        query="SENTINEL-CHAT-TEXT",
        share_access_token="SENTINEL-SHARE-TOKEN",
        messages=[
            MessageDto.model_validate(
                {"message": {"role": "user", "content": "SENTINEL-MESSAGE"}}
            )
        ],
        edition="basic",
        level="high",
        lang="English",
    )
    caplog.set_level(logging.DEBUG, logger="wizard")

    build(
        review_error(), lang="English", request=request, trace_info=TraceInfo("req-1")
    )

    records = [
        record
        for record in caplog.records
        if isinstance(record.msg, dict)
        and record.msg.get("event") == "wizard.stream.error"
    ]
    assert len(records) == 1
    record = records[0]
    payload = record.msg
    assert payload["error_code"] == "content_review_rejected"
    assert payload["retryable"] is False
    assert payload["upstream_status"] == 400
    assert payload["upstream_code"] == "data_inspection_failed"
    assert payload["upstream_request_id"] == UPSTREAM_REQUEST_ID
    assert payload["request_id"] == "req-1"
    assert payload["request_type"] == "agent"
    assert payload["namespace_id"] == "ns-1"
    assert payload["conversation_id"] == "conv-1"
    assert payload["edition"] == "basic"
    assert payload["level"] == "high"
    assert payload["lang"] == "English"
    assert payload["message_count"] == 1
    serialized = json.dumps(payload, default=str)
    for sentinel in ("SENTINEL-CHAT-TEXT", "SENTINEL-MESSAGE", "SENTINEL-SHARE-TOKEN"):
        assert sentinel not in serialized
    assert "Authorization" not in serialized
    assert record.exc_info is None


def test_log_payload_records_traceback_for_non_api_errors(caplog, monkeypatch):
    span = MagicMock()
    span.get_span_context.return_value = SpanContext(
        trace_id=0, span_id=0, is_remote=False
    )
    monkeypatch.setattr(errors_module.trace, "get_current_span", lambda: span)
    caplog.set_level(logging.DEBUG, logger="wizard")

    try:
        raise RuntimeError("boom")
    except RuntimeError as error:
        build_stream_error_response(
            error,
            request=make_request(),
            trace_info=TraceInfo("req-1"),
        )

    records = [
        record
        for record in caplog.records
        if isinstance(record.msg, dict)
        and record.msg.get("event") == "wizard.stream.error"
    ]
    assert len(records) == 1
    assert records[0].msg["error_code"] == "internal_error"
    assert records[0].exc_info is not None


def test_helper_survives_logging_and_span_failures(monkeypatch):
    span = MagicMock()
    span.get_span_context.side_effect = RuntimeError("span broken")
    span.record_exception.side_effect = RuntimeError("span broken")
    monkeypatch.setattr(errors_module.trace, "get_current_span", lambda: span)

    trace_info = TraceInfo("req-1")
    monkeypatch.setattr(
        trace_info, "error", MagicMock(side_effect=RuntimeError("log broken"))
    )
    monkeypatch.setattr(
        trace_info, "exception", MagicMock(side_effect=RuntimeError("log broken"))
    )

    response = build(ValueError("bad"), trace_info=trace_info)
    assert response.message == GENERIC_MESSAGE_ZH
    assert response.code == "internal_error"
    assert response.request_id == "req-1"


def test_helper_survives_classifier_failure(monkeypatch):
    def broken(error):
        raise RuntimeError("classifier broken")

    monkeypatch.setattr(errors_module, "classify_error", broken)

    response = build_stream_error_response(ValueError("bad"))
    assert response.message == GENERIC_MESSAGE_ZH
    assert response.code == "internal_error"
    assert response.retryable is True


@pytest.mark.asyncio
async def test_stream_wrapper_normal_stream():
    stream = FakeStream([delta("a"), delta("b")])
    chunks = await collect(stream, make_request(), TraceInfo("req-1"))
    assert [chunk["response_type"] for chunk in chunks] == ["delta", "delta", "done"]
    assert chunks[0]["message"]["content"] == "a"
    assert stream.closed


@pytest.mark.asyncio
async def test_stream_wrapper_error_after_deltas():
    stream = FakeStream([delta("partial")], error=review_error())
    chunks = await collect(stream, make_request(), TraceInfo("req-1"))
    assert [chunk["response_type"] for chunk in chunks] == ["delta", "error", "done"]
    error = chunks[-2]
    assert error["code"] == "content_review_rejected"
    assert error["message"] == INPUT_MESSAGE_ZH
    assert error["retryable"] is False
    assert error["request_id"] == "req-1"
    assert error["upstream_request_id"] == UPSTREAM_REQUEST_ID
    assert error["upstream_status"] == 400
    assert "Inappropriate content" not in error["message"]
    assert stream.closed


@pytest.mark.asyncio
async def test_stream_wrapper_english_lang():
    request = make_request(lang="English")
    stream = FakeStream(error=review_error())
    chunks = await collect(stream, request, TraceInfo("req-1"))
    assert chunks[0]["message"] == INPUT_MESSAGE_EN


@pytest.mark.asyncio
async def test_stream_wrapper_non_api_error():
    stream = FakeStream([], error=ValueError("boom"))
    chunks = await collect(stream, make_request(), TraceInfo("req-1"))
    assert [chunk["response_type"] for chunk in chunks] == ["error", "done"]
    assert chunks[0]["code"] == "internal_error"
    assert chunks[0]["message"] == GENERIC_MESSAGE_ZH
    assert chunks[0]["message"] != "Unknown error"


@pytest.mark.asyncio
async def test_stream_wrapper_aclose_failure_still_ends_stream(caplog):
    caplog.set_level(logging.DEBUG, logger="wizard")
    stream = FakeStream(error=review_error(), aclose_error=RuntimeError("close failed"))
    chunks = await collect(stream, make_request(), TraceInfo("req-1"))
    assert [chunk["response_type"] for chunk in chunks] == ["error", "done"]
    assert chunks[-1] == {"response_type": "done"}
    warnings = [
        record
        for record in caplog.records
        if isinstance(record.msg, dict)
        and record.msg.get("event") == "wizard.stream.aclose_failed"
    ]
    assert len(warnings) == 1
    assert warnings[0].msg["exception_class"] == "RuntimeError"


@pytest.mark.asyncio
async def test_stream_wrapper_debug_log_redacts_share_token(caplog):
    caplog.set_level(logging.DEBUG, logger="wizard")
    request = make_request(share_access_token="SENTINEL-SHARE-TOKEN")
    stream = FakeStream()
    await collect(stream, request, TraceInfo("req-1"))
    debug_requests = [
        record.msg["request"]
        for record in caplog.records
        if isinstance(record.msg, dict) and "request" in record.msg
    ]
    assert debug_requests
    assert all("share_access_token" not in payload for payload in debug_requests)


@pytest.mark.asyncio
async def test_stream_wrapper_error_dump_omits_none_fields():
    stream = FakeStream(error=ValueError("boom"))
    chunks = await collect(stream, make_request(), TraceInfo("req-1"))
    error = chunks[0]
    assert "trace_id" not in error
    assert "upstream_status" not in error
    assert "upstream_code" not in error
    assert "upstream_request_id" not in error
