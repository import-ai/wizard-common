from dataclasses import dataclass
from typing import Any

import openai
from common.logger import get_logger
from common.trace_info import TraceInfo
from opentelemetry import trace
from opentelemetry.trace import StatusCode
from pydantic import BaseModel

from wizard_common.grimoire.entity.api import ChatErrorResponse

CONTENT_REVIEW_MESSAGE_ZH = "本次请求未通过模型服务的安全审核（可能含历史消息），请修改相关内容后重试，或新建对话。"
CONTENT_REVIEW_MESSAGE_EN = (
    "This request did not pass the model service's safety review (earlier messages "
    "may be involved). Edit the relevant content and try again, or start a new "
    "conversation."
)
OUTPUT_CONTENT_REVIEW_MESSAGE_ZH = (
    "生成的回复未通过模型服务的安全审核，请新建对话或修改后重试。"
)
OUTPUT_CONTENT_REVIEW_MESSAGE_EN = (
    "The generated reply did not pass the model service's safety review. Start a new "
    "conversation or edit the relevant content and try again."
)
GENERIC_ERROR_MESSAGE_ZH = "网络错误，请稍后重试。"
GENERIC_ERROR_MESSAGE_EN = "Network error. Please try again later."

_DATA_INSPECTION_FAILED = "data_inspection_failed"
_MAX_CAUSE_DEPTH = 4
_UPSTREAM_MESSAGE_LOG_LIMIT = 500
_ERROR_EVENT = "wizard.stream.error"


@dataclass(frozen=True)
class StreamErrorInfo:
    code: str
    retryable: bool
    upstream_status: int | None = None
    upstream_code: str | None = None
    upstream_type: str | None = None
    upstream_message: str | None = None
    upstream_request_id: str | None = None


def _find_api_error(error: BaseException) -> openai.APIError | None:
    current: BaseException | None = error
    for _ in range(_MAX_CAUSE_DEPTH + 1):
        if isinstance(current, openai.APIError):
            return current
        if current is None:
            return None
        current = current.__cause__ or current.__context__
    return None


def _body(error: openai.APIError) -> dict:
    body = getattr(error, "body", None)
    return body if isinstance(body, dict) else {}


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _upstream_request_id(error: openai.APIError) -> str | None:
    response = getattr(error, "response", None)
    if response is not None:
        try:
            data = response.json()
        except Exception:
            data = None
        if isinstance(data, dict):
            for key in ("id", "request_id"):
                value = _string_or_none(data.get(key))
                if value:
                    return value
    return _string_or_none(getattr(error, "request_id", None))


def classify_error(error: Exception) -> StreamErrorInfo:
    api_error = _find_api_error(error)
    if api_error is None:
        return StreamErrorInfo(code="internal_error", retryable=True)

    status = getattr(api_error, "status_code", None)
    if not isinstance(status, int):
        status = None
    body = _body(api_error)
    upstream_code = _string_or_none(body.get("code")) or _string_or_none(
        getattr(api_error, "code", None)
    )
    upstream_type = _string_or_none(body.get("type")) or _string_or_none(
        getattr(api_error, "type", None)
    )
    upstream_message = _string_or_none(body.get("message")) or _string_or_none(
        getattr(api_error, "message", None)
    )
    request_id = _upstream_request_id(api_error)

    review_code = (upstream_code or "").lower()
    review_type = (upstream_type or "").lower()
    if review_code == _DATA_INSPECTION_FAILED or review_type == _DATA_INSPECTION_FAILED:
        if (upstream_message or "").lstrip().lower().startswith("output"):
            code = "output_content_review_rejected"
        else:
            code = "content_review_rejected"
        return StreamErrorInfo(
            code=code,
            retryable=False,
            upstream_status=status,
            upstream_code=upstream_code,
            upstream_type=upstream_type,
            upstream_message=upstream_message,
            upstream_request_id=request_id,
        )

    if isinstance(api_error, openai.APITimeoutError) or status == 408:
        code, retryable = "upstream_timeout", True
    elif isinstance(api_error, openai.APIConnectionError):
        code, retryable = "upstream_connection_failed", True
    elif status == 400:
        code, retryable = "upstream_bad_request", False
    elif status == 401:
        code, retryable = "upstream_unauthorized", False
    elif status == 403:
        code, retryable = "upstream_forbidden", False
    elif status == 404:
        code, retryable = "upstream_not_found", False
    elif status == 409:
        code, retryable = "upstream_conflict", True
    elif status == 422:
        code, retryable = "upstream_unprocessable", False
    elif status == 429:
        code, retryable = "upstream_rate_limited", True
    elif status is not None and 500 <= status <= 599:
        code, retryable = "upstream_server_error", True
    else:
        code, retryable = "upstream_api_error", True

    return StreamErrorInfo(
        code=code,
        retryable=retryable,
        upstream_status=status,
        upstream_code=upstream_code,
        upstream_type=upstream_type,
        upstream_message=upstream_message,
        upstream_request_id=request_id,
    )


def render_error_message(info: StreamErrorInfo, lang: str | None) -> str:
    english = lang == "English"
    if info.code == "content_review_rejected":
        return CONTENT_REVIEW_MESSAGE_EN if english else CONTENT_REVIEW_MESSAGE_ZH
    if info.code == "output_content_review_rejected":
        return (
            OUTPUT_CONTENT_REVIEW_MESSAGE_EN
            if english
            else OUTPUT_CONTENT_REVIEW_MESSAGE_ZH
        )
    return GENERIC_ERROR_MESSAGE_EN if english else GENERIC_ERROR_MESSAGE_ZH


def current_trace_id() -> str | None:
    try:
        span_context = trace.get_current_span().get_span_context()
        if not span_context.is_valid:
            return None
        trace_id = getattr(span_context, "trace_id", 0)
        if not trace_id:
            return None
        return format(trace_id, "032x")
    except Exception:
        return None


def _request_type(request: BaseModel | None) -> str | None:
    if request is None:
        return None
    if getattr(request, "share_id", None):
        return "share"
    # Ask and write share AgentRequest, so the class name is the only type hint.
    name = type(request).__name__
    return name.removesuffix("Request").lower() or None


def _log_payload(
    error: Exception,
    info: StreamErrorInfo,
    *,
    lang: str | None,
    request: BaseModel | None,
    request_id: str | None,
    trace_id: str | None,
) -> dict:
    messages = getattr(request, "messages", None)
    return {
        "event": _ERROR_EVENT,
        "error_code": info.code,
        "retryable": info.retryable,
        "exception_class": error.__class__.__name__,
        "upstream_status": info.upstream_status,
        "upstream_code": info.upstream_code,
        "upstream_type": info.upstream_type,
        "upstream_message": (
            info.upstream_message[:_UPSTREAM_MESSAGE_LOG_LIMIT]
            if info.upstream_message
            else None
        ),
        "upstream_request_id": info.upstream_request_id,
        "request_id": request_id,
        "trace_id": trace_id,
        "request_type": _request_type(request),
        "namespace_id": getattr(request, "namespace_id", None) if request else None,
        "conversation_id": (
            getattr(request, "conversation_id", None) if request else None
        ),
        "edition": getattr(request, "edition", None) if request else None,
        "level": getattr(request, "level", None) if request else None,
        "lang": lang if lang is not None else getattr(request, "lang", None),
        "message_count": len(messages) if messages else 0,
    }


def _log_error(
    error: Exception,
    info: StreamErrorInfo,
    *,
    lang: str | None,
    request: BaseModel | None,
    request_id: str | None,
    trace_id: str | None,
    trace_info: TraceInfo | None,
) -> None:
    payload = _log_payload(
        error,
        info,
        lang=lang,
        request=request,
        request_id=request_id,
        trace_id=trace_id,
    )
    logger = trace_info if trace_info is not None else get_logger("wizard")
    if isinstance(error, openai.APIError):
        logger.error(payload)
    else:
        logger.exception(payload)


def _set_span_attributes(
    error: Exception,
    info: StreamErrorInfo,
    *,
    message: str,
    request_id: str | None,
    trace_id: str | None,
) -> None:
    span = trace.get_current_span()
    span.record_exception(error)
    span.set_status(StatusCode.ERROR)
    attributes = {
        "error_code": info.code,
        "error.retryable": info.retryable,
        "error.request_id": request_id,
        "error.trace_id": trace_id,
        "error.upstream_status": info.upstream_status,
        "error.upstream_request_id": info.upstream_request_id,
        "error_message": message,
    }
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)


def build_stream_error_response(
    error: Exception,
    *,
    lang: str | None = None,
    request: BaseModel | None = None,
    trace_info: TraceInfo | None = None,
) -> ChatErrorResponse:
    try:
        info = classify_error(error)
    except Exception:
        info = StreamErrorInfo(code="internal_error", retryable=True)
    message = render_error_message(info, lang)
    try:
        trace_id = current_trace_id()
    except Exception:
        trace_id = None
    try:
        request_id = trace_info.request_id if trace_info is not None else None
    except Exception:
        request_id = None
    if not request_id:
        request_id = trace_id

    try:
        _log_error(
            error,
            info,
            lang=lang,
            request=request,
            request_id=request_id,
            trace_id=trace_id,
            trace_info=trace_info,
        )
    except Exception:
        pass
    try:
        _set_span_attributes(
            error,
            info,
            message=message,
            request_id=request_id,
            trace_id=trace_id,
        )
    except Exception:
        pass

    return ChatErrorResponse(
        message=message,
        code=info.code,
        retryable=info.retryable,
        request_id=request_id,
        trace_id=trace_id,
        upstream_request_id=info.upstream_request_id,
        upstream_status=info.upstream_status,
        upstream_code=info.upstream_code,
    )
