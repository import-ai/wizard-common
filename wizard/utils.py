from typing import AsyncIterator

import openai
from opentelemetry import trace
from pydantic import BaseModel
from sse_starlette import EventSourceResponse

from common.trace_info import TraceInfo
from common.utils import json_dumps
from wizard_common.grimoire.base_streamable import BaseStreamable, ChatResponse
from wizard_common.grimoire.entity.api import BaseChatRequest

tracer = trace.get_tracer("wizard-common")


async def stream_wrapper(
    request: BaseModel, stream: AsyncIterator[ChatResponse], trace_info: TraceInfo
) -> AsyncIterator[dict]:
    span = trace.get_current_span()
    trace_info.debug({"request": request.model_dump(exclude_none=True)})
    error: Exception | None = None
    error_message: str | None = ""
    try:
        async for delta in stream:
            yield delta.model_dump(exclude_none=True)
    except openai.APIError as e:
        error, error_message = e, "Inappropriate content"
    except Exception as e:
        error, error_message = e, "Unknown error"
    if error:
        span.record_exception(error)
        span.set_attribute("error_message", error_message)
        trace_info.exception(
            {
                "exception_class": error.__class__.__name__,
                "exception_message": str(error),
                "request": request.model_dump(exclude_none=True),
            }
        )
        yield {"response_type": "error", "message": error_message}
    yield {"response_type": "done"}


async def call_stream(
    s: BaseStreamable, request: BaseChatRequest, trace_info: TraceInfo
) -> AsyncIterator[dict]:
    with tracer.start_as_current_span("wizard.call_stream"):
        stream = s.astream(trace_info.get_child("agent"), request)
        async for delta in stream_wrapper(request, stream, trace_info):  # noqa
            yield delta


async def sse_format(iterator: AsyncIterator[dict]) -> AsyncIterator[str]:
    async for item in iterator:
        yield f"data: {json_dumps(item)}\n\n"


async def sse_dumps(iterator: AsyncIterator[dict]) -> AsyncIterator[str]:
    async for item in iterator:
        yield json_dumps(item)


def streaming_response(iterator: AsyncIterator[dict]) -> EventSourceResponse:
    return EventSourceResponse(sse_dumps(iterator))
