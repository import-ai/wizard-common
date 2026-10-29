import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from wizard_common.grimoire.config import (
    BingSearchConfig,
    DeepSeekSearchConfig,
    SearXNGConfig,
)
from wizard_common.grimoire.retriever import bing_search as bing_module
from wizard_common.grimoire.retriever import web_search as module
from wizard_common.grimoire.retriever.bing_search import BingSearch
from wizard_common.grimoire.retriever.deepseek_search import DeepSeekSearch
from wizard_common.grimoire.retriever.searxng import SearXNGRetrieval
from wizard_common.grimoire.retriever.web_search import WebSearch


def result(content="found"):
    return SearXNGRetrieval(
        result={"title": "Title", "url": "https://example.com", "content": content}
    )


@pytest.fixture
def spans(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(module, "tracer", provider.get_tracer("test-web-search"))
    yield exporter
    provider.shutdown()


def searcher(bing=None, deepseek=None, searxng=None):
    search = WebSearch.__new__(WebSearch)
    search.primary = SimpleNamespace(search=bing or AsyncMock(return_value=[]))
    search.deepseek = SimpleNamespace(search=deepseek or AsyncMock(return_value=[]))
    search.fallback = SimpleNamespace(search=searxng or AsyncMock(return_value=[]))
    return search


def parent_span(spans):
    return next(s for s in spans.get_finished_spans() if s.name == "WebSearch.search")


@pytest.mark.asyncio
async def test_fast_bing_prevents_delayed_requests_and_records_spans(spans):
    search = searcher(bing=AsyncMock(return_value=[result()]))
    assert await search.search("query") == [result()]
    search.deepseek.search.assert_not_awaited()
    search.fallback.search.assert_not_awaited()
    span = parent_span(spans)
    assert span.attributes["web_search.winner"] == "bing"
    assert span.attributes["web_search.bing.started"] is True
    assert span.attributes["web_search.bing.status"] == "success"
    assert span.attributes["web_search.bing.latency_ms"] >= 0
    for name in ("deepseek", "searxng"):
        assert span.attributes[f"web_search.{name}.started"] is False
        assert span.attributes[f"web_search.{name}.status"] == "not_started"
        assert f"web_search.{name}.latency_ms" not in span.attributes
    children = [s for s in spans.get_finished_spans() if s.name == "WebSearch.provider"]
    assert len(children) == 1
    assert children[0].parent.span_id == span.context.span_id
    assert children[0].attributes["search.provider"] == "bing"


@pytest.mark.asyncio
async def test_late_bing_can_still_win_after_deepseek_starts(spans):
    deepseek_started = asyncio.Event()
    deepseek_cancelled = asyncio.Event()

    async def bing(*args, **kwargs):
        await deepseek_started.wait()
        return [result("bing")]

    async def deepseek(*args, **kwargs):
        deepseek_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            deepseek_cancelled.set()

    search = searcher(
        bing=AsyncMock(side_effect=bing), deepseek=AsyncMock(side_effect=deepseek)
    )
    async with asyncio.timeout(3):
        results = await search.search("query")
    assert results == [result("bing")]
    assert deepseek_cancelled.is_set()
    search.fallback.search.assert_not_awaited()
    attrs = parent_span(spans).attributes
    assert attrs["web_search.deepseek.start_delay_ms"] >= 990
    assert attrs["web_search.deepseek.status"] == "cancelled"
    assert attrs["web_search.latency_ms"] >= 990


@pytest.mark.asyncio
async def test_deepseek_wins_and_cancels_slow_bing(spans):
    bing_cancelled = asyncio.Event()

    async def bing(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            bing_cancelled.set()

    search = searcher(
        bing=AsyncMock(side_effect=bing),
        deepseek=AsyncMock(return_value=[result("deepseek")]),
    )
    async with asyncio.timeout(3):
        assert await search.search("query", k=3) == [result("deepseek")]
    assert bing_cancelled.is_set()
    search.fallback.search.assert_not_awaited()
    attrs = parent_span(spans).attributes
    assert attrs["web_search.winner"] == "deepseek"
    assert attrs["web_search.bing.status"] == "cancelled"
    assert attrs["web_search.deepseek.start_delay_ms"] >= 990


@pytest.mark.asyncio
async def test_empty_or_failed_providers_start_the_next_provider_immediately(spans):
    search = searcher(
        bing=AsyncMock(side_effect=httpx.ReadTimeout("slow")),
        deepseek=AsyncMock(return_value=[result(" ")]),
        searxng=AsyncMock(return_value=[result("searxng")]),
    )
    async with asyncio.timeout(0.5):
        assert await search.search("query") == [result("searxng")]
    attrs = parent_span(spans).attributes
    assert attrs["web_search.winner"] == "searxng"
    assert attrs["web_search.bing.status"] == "error"
    assert attrs["web_search.deepseek.status"] == "empty"
    assert attrs["web_search.deepseek.start_delay_ms"] < 500
    assert attrs["web_search.searxng.start_delay_ms"] < 500


@pytest.mark.asyncio
async def test_all_empty_returns_empty_after_all_providers_finish(spans):
    search = searcher()
    async with asyncio.timeout(0.5):
        assert await search.search("query") == []
    attrs = parent_span(spans).attributes
    assert attrs["web_search.winner"] == "none"
    assert attrs["web_search.status"] == "empty"
    assert attrs["web_search.latency_ms"] < 500
    for name in ("bing", "deepseek", "searxng"):
        assert attrs[f"web_search.{name}.started"] is True
        assert attrs[f"web_search.{name}.status"] == "empty"
        assert attrs[f"web_search.{name}.latency_ms"] >= 0


@pytest.mark.asyncio
async def test_early_empty_bing_does_not_remove_searxng_absolute_deadline(spans):
    async def deepseek(*args, **kwargs):
        await asyncio.Event().wait()

    search = searcher(
        deepseek=AsyncMock(side_effect=deepseek),
        searxng=AsyncMock(return_value=[result("searxng")]),
    )
    async with asyncio.timeout(4):
        assert await search.search("query") == [result("searxng")]
    attrs = parent_span(spans).attributes
    assert attrs["web_search.deepseek.start_delay_ms"] < 500
    assert attrs["web_search.searxng.start_delay_ms"] >= 1990
    assert attrs["web_search.deepseek.status"] == "cancelled"


@pytest.mark.asyncio
async def test_empty_deepseek_starts_searxng_while_bing_is_still_running(spans):
    async def bing(*args, **kwargs):
        await asyncio.Event().wait()

    search = searcher(
        bing=AsyncMock(side_effect=bing),
        searxng=AsyncMock(return_value=[result("searxng")]),
    )
    async with asyncio.timeout(1.8):
        assert await search.search("query") == [result("searxng")]
    attrs = parent_span(spans).attributes
    assert attrs["web_search.deepseek.start_delay_ms"] >= 990
    assert 990 <= attrs["web_search.searxng.start_delay_ms"] < 1800
    assert attrs["web_search.bing.status"] == "cancelled"


@pytest.mark.asyncio
async def test_disabled_providers_do_not_delay_remaining_search(spans):
    search = searcher(searxng=AsyncMock(return_value=[result()]))
    search.primary = None
    search.deepseek = None
    async with asyncio.timeout(0.5):
        assert await search.search("query") == [result()]
    attrs = parent_span(spans).attributes
    assert attrs["web_search.bing.status"] == "disabled"
    assert attrs["web_search.deepseek.status"] == "disabled"
    assert attrs["web_search.searxng.start_delay_ms"] < 500


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_cnt", [1, 2])
async def test_bing_does_not_sleep_after_its_last_failed_attempt(
    monkeypatch, retry_cnt
):
    search = BingSearch("https://bing.invalid")
    search.search_once = AsyncMock(side_effect=httpx.ReadTimeout("slow"))
    sleep = AsyncMock()
    monkeypatch.setattr(bing_module, "asyncio", SimpleNamespace(sleep=sleep))
    with pytest.raises(httpx.ReadTimeout):
        await search.search("query", retry_cnt=retry_cnt)
    assert sleep.await_count == retry_cnt - 1


@pytest.mark.asyncio
async def test_searxng_winner_cancels_both_running_competitors(spans):
    cancelled = set()

    async def slow(name, *args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.add(name)

    async def bing(*args, **kwargs):
        return await slow("bing")

    async def deepseek(*args, **kwargs):
        return await slow("deepseek")

    search = searcher(
        bing=AsyncMock(side_effect=bing),
        deepseek=AsyncMock(side_effect=deepseek),
        searxng=AsyncMock(return_value=[result("searxng")]),
    )
    async with asyncio.timeout(4):
        assert await search.search("query") == [result("searxng")]
    assert cancelled == {"bing", "deepseek"}
    attrs = parent_span(spans).attributes
    assert attrs["web_search.winner"] == "searxng"
    for name in ("bing", "deepseek"):
        assert attrs[f"web_search.{name}.status"] == "cancelled"
        assert attrs[f"web_search.{name}.latency_ms"] > 0


@pytest.mark.asyncio
async def test_caller_cancellation_cleans_up_and_does_not_start_delayed_calls(spans):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def bing(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    search = searcher(bing=AsyncMock(side_effect=bing))
    task = asyncio.create_task(search.search("query"))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    search.deepseek.search.assert_not_awaited()
    search.fallback.search.assert_not_awaited()
    attrs = parent_span(spans).attributes
    assert attrs["web_search.status"] == "cancelled"
    assert attrs["web_search.bing.status"] == "cancelled"
    assert attrs["web_search.latency_ms"] >= 0


@pytest.mark.asyncio
async def test_total_latency_is_recorded_when_cancellation_interrupts_cleanup(spans):
    cleanup_started = asyncio.Event()

    async def bing(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await asyncio.Event().wait()

    search = searcher(
        bing=AsyncMock(side_effect=bing),
        deepseek=AsyncMock(return_value=[result()]),
    )
    task = asyncio.create_task(search.search("query"))
    await asyncio.wait_for(cleanup_started.wait(), timeout=3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    attrs = parent_span(spans).attributes
    assert attrs["web_search.status"] == "cancelled"
    assert attrs["web_search.latency_ms"] >= 990


def test_missing_configuration_disables_optional_providers(monkeypatch):
    monkeypatch.delenv("OBW_TASK_SCRAPE_BASE_URL", raising=False)
    search = WebSearch(
        searxng=SearXNGConfig(base_url="https://searxng.invalid"),
        bing_search=BingSearchConfig(enabled=False),
    )
    assert search.primary is None
    assert search.deepseek is None
    enabled = WebSearch(
        searxng=SearXNGConfig(base_url="https://searxng.invalid"),
        bing_search=BingSearchConfig(base_url="https://bing.invalid"),
        deepseek_search=DeepSeekSearchConfig(api_key="test-key"),
    )
    assert enabled.primary is not None
    assert enabled.deepseek is not None


def test_deepseek_key_loads_from_existing_environment_name(monkeypatch):
    from common.config_loader import load_from_env

    from wizard_common.grimoire.config import ToolsConfig

    monkeypatch.setenv("OBW_TOOLS_DEEPSEEK_SEARCH_API_KEY", "test-key")
    loaded = load_from_env(ToolsConfig, "OBW_TOOLS")
    assert loaded["deepseek_search"]["api_key"] == "test-key"


@pytest.mark.asyncio
async def test_deepseek_request_and_result_conversion(monkeypatch):
    captured = []

    def handle(request):
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "A",
                        "url": "https://a.example",
                        "summary": "long",
                        "snippet": "short",
                    },
                    {
                        "title": "B",
                        "url": "https://b.example",
                        "summary": " ",
                        "snippet": "fallback",
                    },
                    {"title": "", "url": "https://bad.example", "summary": "invalid"},
                    {"title": "Bad type", "url": 123, "summary": "invalid"},
                    "invalid",
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(
        "wizard_common.grimoire.retriever.deepseek_search.httpx.AsyncClient",
        lambda **kwargs: client,
    )
    results = await DeepSeekSearch("test-key").search("query", k=2)
    assert client.is_closed
    assert [r.to_citation().snippet for r in results] == ["long", "fallback"]
    assert str(captured[0].url) == "https://api.deepseek.com/v1/tools/search"
    assert captured[0].headers["Authorization"] == "Bearer test-key"
    assert json.loads(captured[0].content) == {
        "query": "query",
        "count": 10,
        "need_content": False,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("payload,status", [({"results": {}}, 200), ({}, 401)])
async def test_deepseek_rejects_invalid_responses_and_http_errors(
    payload, status, monkeypatch
):
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(status, json=payload))
    )
    monkeypatch.setattr(
        "wizard_common.grimoire.retriever.deepseek_search.httpx.AsyncClient",
        lambda **kwargs: client,
    )
    expected = httpx.HTTPStatusError if status != 200 else ValueError
    with pytest.raises(expected):
        await DeepSeekSearch("test-key").search("query")
    assert client.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["", " ", "x" * 129])
async def test_deepseek_rejects_invalid_queries_before_http(query, monkeypatch):
    client = AsyncMock()
    monkeypatch.setattr(
        "wizard_common.grimoire.retriever.deepseek_search.httpx.AsyncClient", client
    )
    with pytest.raises(ValueError):
        await DeepSeekSearch("test-key").search(query)
    client.assert_not_called()
