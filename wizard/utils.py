from typing import AsyncIterator

from common.trace_info import TraceInfo
from common.utils import json_dumps
from opentelemetry import trace
from pydantic import BaseModel
from sse_starlette import EventSourceResponse

from wizard_common.grimoire.base_streamable import BaseStreamable, ChatResponse
from wizard_common.grimoire.entity.api import BaseChatRequest, ChatErrorResponse
from wizard_common.wizard.errors import build_stream_error_response

tracer = trace.get_tracer("wizard-common")


async def stream_wrapper(
    request: BaseModel, stream: AsyncIterator[ChatResponse], trace_info: TraceInfo
) -> AsyncIterator[dict]:
    trace_info.debug(
        {
            "request": request.model_dump(
                exclude_none=True, exclude={"share_access_token"}
            )
        }
    )
    error_response: ChatErrorResponse | None = None
    try:
        async for delta in stream:
            yield delta.model_dump(exclude_none=True)
    except Exception as e:
        # Built inside the except block so TraceInfo.exception keeps the traceback.
        error_response = build_stream_error_response(
            e,
            lang=getattr(request, "lang", None),
            request=request,
            trace_info=trace_info,
        )
    finally:
        if aclose := getattr(stream, "aclose", None):
            try:
                await aclose()
            except Exception as close_error:
                trace_info.warning(
                    {
                        "event": "wizard.stream.aclose_failed",
                        "exception_class": close_error.__class__.__name__,
                    }
                )
    if error_response is not None:
        yield error_response.model_dump(exclude_none=True)
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
