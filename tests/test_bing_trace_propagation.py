import httpx
import pytest
from opentelemetry import propagate, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from wizard_common.grimoire.retriever.bing_search import BingSearch


@pytest.mark.asyncio
async def test_bing_request_propagates_active_context_to_server(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test-cross-service-bing")
    constructor = httpx.AsyncClient
    headers = {}

    def handle(request):
        headers.update(request.headers)
        active = trace.get_current_span().get_span_context()
        remote = propagate.extract(dict(request.headers))
        extracted = trace.get_current_span(remote).get_span_context()
        assert extracted.trace_id == active.trace_id
        assert extracted.span_id == active.span_id
        assert extracted.is_remote
        with tracer.start_as_current_span(
            "crawler", context=remote, kind=trace.SpanKind.SERVER
        ):
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Title",
                            "url": "https://page.invalid",
                            "content": "Snippet",
                        }
                    ]
                },
            )

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: constructor(transport=httpx.MockTransport(handle), **kwargs),
    )
    try:
        with tracer.start_as_current_span("wizard") as parent:
            results = await BingSearch("http://crawler.invalid").search_once("query")
        assert len(results) == 1 and "traceparent" in headers
        server = next(s for s in exporter.get_finished_spans() if s.name == "crawler")
        assert server.context.trace_id == parent.get_span_context().trace_id
        assert server.parent.is_remote
    finally:
        provider.shutdown()
