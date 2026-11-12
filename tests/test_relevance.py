import asyncio
import json
import logging
import time
import traceback
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import httpx
import pytest
from common.config_loader import Loader
from opentelemetry.trace import SpanContext
from pydantic import ValidationError

from wizard_common.grimoire.config import (
    BingSearchConfig,
    RelevanceConfig,
    SearXNGConfig,
    ToolsConfig,
)
from wizard_common.grimoire.entity.chunk import Chunk, ChunkType, ResourceChunkRetrieval
from wizard_common.grimoire.entity.index_record import IndexRecord, IndexRecordType
from wizard_common.grimoire.entity.message import Message, OpenAIMessage
from wizard_common.grimoire.entity.retrieval import Score
from wizard_common.grimoire.entity.tools import PrivateSearchTool
from wizard_common.grimoire.retriever import relevance as relevance_module
from wizard_common.grimoire.retriever.relevance import (
    BING_QUESTION,
    SEARCH_BATCH_CONCURRENCY,
    RelevanceChecker,
    RelevanceError,
    SearchFilterResult,
    _title_matches,
)
from wizard_common.grimoire.retriever.resource_search import ResourceSearch
from wizard_common.grimoire.retriever.searxng import SearXNGRetrieval
from wizard_common.grimoire.retriever.weaviate_vector_db import WeaviateVectorRetriever
from wizard_common.grimoire.retriever.web_search import WebSearch


def config(**overrides):
    return RelevanceConfig(
        **{
            "base_url": "https://judge.invalid/v1",
            "model": "decision-test",
            "bing_enabled": True,
            **overrides,
        }
    )


def answer(score):
    return {
        "type": "choice",
        "choice": "relevant",
        "confidence": 0.99,
        "probabilities": {"off_target": score, "relevant": 1 - score},
    }


def hits():
    return [
        {"title": f"Title {i}", "url": f"https://page.invalid/{i}", "content": "Text"}
        for i in range(5)
    ]


def install_transport(monkeypatch, handler):
    constructor = httpx.AsyncClient
    clients = []

    def create(**kwargs):
        client = constructor(transport=httpx.MockTransport(handler), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", create)
    return clients


def install_span(monkeypatch):
    span = Mock()
    span.is_recording.return_value = True
    monkeypatch.setattr(
        relevance_module.trace, "get_current_span", lambda context=None: span
    )
    return span


def install_evaluate_span(monkeypatch):
    span = MagicMock()
    span.__enter__.return_value = span
    span.__exit__.return_value = False
    tracer = Mock()
    tracer.start_as_current_span.return_value = span
    monkeypatch.setattr(relevance_module, "tracer", tracer)
    return span


def retrieval(resource_id):
    return ResourceChunkRetrieval(
        chunk=Chunk(
            title=resource_id,
            resource_id=resource_id,
            text=f"Private text for {resource_id}",
            chunk_type=ChunkType.snippet,
            parent_id="parent",
        ),
        score=Score(recall=1, rerank=0),
    )


def search_record(resource_id, *, title=None, text=None):
    return IndexRecord(
        type=IndexRecordType.chunk,
        namespace_id="namespace",
        chunk=Chunk(
            title=resource_id if title is None else title,
            resource_id=resource_id,
            text=f"Full text for {resource_id}" if text is None else text,
            chunk_type=ChunkType.doc,
            parent_id="parent",
        ),
    )


def message_record(message_id):
    return IndexRecord(
        type=IndexRecordType.message,
        namespace_id="namespace",
        message=Message(
            conversation_id="conversation",
            message_id=message_id,
            message=OpenAIMessage(role="user", content="hello"),
        ),
    )


def test_default_disabled_requires_no_provider_and_validates_explicit_enable():
    settings = ToolsConfig(searxng=SearXNGConfig(base_url="https://search.invalid"))
    assert (
        not settings.relevance.bing_enabled and not settings.relevance.resource_enabled
    )
    for overrides in [
        {"base_url": None},
        {"base_url": "file:///tmp/model"},
        {"base_url": "https://"},
        {"base_url": "https://judge.invalid/v1?query=bad"},
        {"model": " "},
        {"bing_threshold": -0.1},
        {"timeout_sec": 0},
        {"timeout_sec": float("inf")},
        {"resource_enabled": True},
    ]:
        with pytest.raises(ValidationError):
            config(**overrides)
    assert config(api_key=None).api_key is None


def test_search_defaults_and_validation():
    defaults = RelevanceConfig()
    assert defaults.search_enabled
    assert defaults.search_threshold == 0.5
    assert defaults.search_batch_size == 16
    assert defaults.search_timeout_sec == 3.0
    assert RelevanceConfig(search_enabled=False).search_enabled is False
    assert RelevanceConfig(search_batch_size=16).search_batch_size == 16
    for overrides in [
        {"base_url": "https://judge.invalid/v1"},
        {"model": "decision-test"},
        {"search_batch_size": 0},
        {"search_batch_size": 17},
        {"search_timeout_sec": 0},
        {"search_timeout_sec": float("inf")},
        {"search_threshold": -0.1},
        {"search_threshold": 1.5},
    ]:
        with pytest.raises(ValidationError):
            RelevanceConfig(**overrides)


def test_search_env_config(monkeypatch):
    monkeypatch.setattr("sys.argv", ["wizard"])
    for name, value in {
        "SEARCH_ENABLED": "true",
        "BASE_URL": "http://localhost:9000/v1",
        "MODEL": "local-decision",
        "SEARCH_THRESHOLD": "0.8",
        "SEARCH_BATCH_SIZE": "16",
        "SEARCH_TIMEOUT_SEC": "1.5",
    }.items():
        monkeypatch.setenv("TEST_TOOLS_RELEVANCE_" + name, value)
    settings = Loader(
        ToolsConfig,
        env_prefix="TEST_TOOLS",
        config_dict={"searxng": {"base_url": "https://search.invalid"}},
    ).load()
    assert settings.relevance.search_enabled
    assert settings.relevance.search_threshold == 0.8
    assert settings.relevance.search_batch_size == 16
    assert settings.relevance.search_timeout_sec == 1.5


def test_env_config_and_secret_masking(monkeypatch):
    monkeypatch.setattr("sys.argv", ["wizard"])
    for name, value in {
        "BING_ENABLED": "true",
        "BASE_URL": "http://localhost:9000/v1",
        "MODEL": "local-decision",
        "API_KEY": "test-secret",
    }.items():
        monkeypatch.setenv("TEST_TOOLS_RELEVANCE_" + name, value)
    settings = Loader(
        ToolsConfig,
        env_prefix="TEST_TOOLS",
        config_dict={"searxng": {"base_url": "https://search.invalid"}},
    ).load()
    assert settings.relevance.bing_enabled
    assert settings.relevance.api_key.get_secret_value() == "test-secret"
    assert "test-secret" not in settings.model_dump_json()


@pytest.mark.asyncio
async def test_disabled_checks_make_no_client_or_network_request(monkeypatch):
    def unexpected(**kwargs):
        raise AssertionError("Disabled checks must not construct a client")

    monkeypatch.setattr(httpx, "AsyncClient", unexpected)
    checker = RelevanceChecker(RelevanceConfig(search_enabled=False))
    await checker.check_bing("query", hits())
    resources = [retrieval("keep")]
    assert await checker.filter_resources("query", resources) is resources
    records = [search_record("keep")]
    result = await checker.filter_search_results("query", records)
    assert result.kept == records and result.pending == []
    await checker.aclose()


@pytest.mark.asyncio
async def test_enabled_search_without_endpoint_is_a_network_free_noop(monkeypatch):
    def unexpected(**kwargs):
        raise AssertionError("Unconfigured search checks must not construct a client")

    span = Mock()
    span.get_span_context.return_value = SpanContext(
        trace_id=1, span_id=2, is_remote=False
    )
    monkeypatch.setattr(httpx, "AsyncClient", unexpected)
    monkeypatch.setattr(
        relevance_module.trace, "get_current_span", lambda context=None: span
    )
    checker = RelevanceChecker(RelevanceConfig())
    records = [search_record("keep"), message_record("message")]
    result = await checker.filter_search_results("query", records)
    assert result.kept == records and result.pending == []
    span.set_attributes.assert_called_once_with(
        {"relevance.status": "unconfigured", "relevance.input_count": 2}
    )
    await checker.aclose()


@pytest.mark.asyncio
async def test_titles_only_payload_boundary_and_connection_reuse(monkeypatch):
    requests = []

    def handle(request):
        requests.append(request)
        score = 0.1 if len(requests) == 1 else 0.099
        return httpx.Response(200, json={"answers": {"relationship": answer(score)}})

    clients = install_transport(monkeypatch, handle)
    checker = RelevanceChecker(config(api_key="test-secret"))
    with pytest.raises(RelevanceError, match="bing_off_target"):
        await checker.check_bing("动量守恒定律", hits())
    await checker.check_bing("query", hits())
    assert len(clients) == 1
    assert requests[0].url == "https://judge.invalid/v1/systemone"
    assert requests[0].headers["authorization"] == "Bearer test-secret"
    assert json.loads(requests[0].content) == {
        "model": "decision-test",
        "state": {
            "query": "动量守恒定律",
            "results": [{"title": f"Title {i}"} for i in range(3)],
        },
        "questions": {"relationship": BING_QUESTION},
    }
    await checker.aclose()
    assert clients[0].is_closed and checker._client is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {},
        {"answers": {}},
        {"answers": {"relationship": {"probabilities": {"off_target": 0.1}}}},
        {"answers": {"relationship": answer(-0.1)}},
        {"answers": {"relationship": answer(True)}},
        {
            "answers": {
                "relationship": {
                    "probabilities": {"off_target": 0.1, "relevant": 0.9, "other": 0.1}
                }
            }
        },
        {
            "answers": {
                "relationship": {"probabilities": {"off_target": 0.1, "relevant": 0.1}}
            }
        },
        {"answers": {"relationship": answer(0.01), "unexpected": answer(0.01)}},
    ],
)
async def test_invalid_response_triggers_safe_error(monkeypatch, body):
    install_transport(monkeypatch, lambda request: httpx.Response(200, json=body))
    checker = RelevanceChecker(config())
    try:
        with pytest.raises(RelevanceError, match="^relevance_check_failed$"):
            await checker.check_bing("query", hits())
    finally:
        await checker.aclose()


@pytest.mark.asyncio
async def test_detection_has_a_total_deadline(monkeypatch):
    cancelled = asyncio.Event()

    async def delayed(request):
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.set()

    install_transport(monkeypatch, delayed)
    checker = RelevanceChecker(config(timeout_sec=0.01))
    try:
        with pytest.raises(RelevanceError, match="^relevance_check_failed$"):
            await asyncio.wait_for(checker.check_bing("query", hits()), timeout=0.5)
        assert cancelled.is_set()
    finally:
        await checker.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["rejected", "rate_limit", "timeout", "invalid_json"]
)
async def test_failed_check_advances_fallback_without_retry(monkeypatch, failure):
    counts = {"bing": 0, "judge": 0}

    async def handle(request):
        if request.url.host == "bing.invalid":
            counts["bing"] += 1
            return httpx.Response(200, json={"results": hits()})
        counts["judge"] += 1
        if failure == "rate_limit":
            return httpx.Response(429, text="Private remote diagnostic")
        if failure == "timeout":
            raise httpx.ReadTimeout("Private diagnostic", request=request)
        if failure == "invalid_json":
            return httpx.Response(200, text="not JSON")
        return httpx.Response(200, json={"answers": {"relationship": answer(0.1)}})

    install_transport(monkeypatch, handle)
    checker = RelevanceChecker(config())
    search = WebSearch(
        searxng=SearXNGConfig(base_url="https://fallback.invalid"),
        bing_search=BingSearchConfig(base_url="https://bing.invalid"),
        relevance=checker,
    )
    fallback = SearXNGRetrieval(result=hits()[0])
    search.deepseek = SimpleNamespace(search=AsyncMock(return_value=[fallback]))
    search.fallback = SimpleNamespace(search=AsyncMock(return_value=[]))
    try:
        async with asyncio.timeout(0.5):
            assert await search.search("query", retry_cnt=3) == [fallback]
        assert counts == {"bing": 1, "judge": 1}
        search.deepseek.search.assert_awaited_once()
        search.fallback.search.assert_not_awaited()
    finally:
        await checker.aclose()


@pytest.mark.asyncio
async def test_inflight_check_is_cancelled_when_another_provider_wins(monkeypatch):
    cancelled = asyncio.Event()

    async def handle(request):
        if request.url.host == "bing.invalid":
            return httpx.Response(200, json={"results": hits()})
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    install_transport(monkeypatch, handle)
    checker = RelevanceChecker(config())
    search = WebSearch(
        searxng=SearXNGConfig(base_url="https://fallback.invalid"),
        bing_search=BingSearchConfig(base_url="https://bing.invalid"),
        relevance=checker,
    )
    search.deepseek = SimpleNamespace(
        search=AsyncMock(return_value=[SearXNGRetrieval(result=hits()[0])])
    )
    try:
        async with asyncio.timeout(2):
            assert await search.search("query")
        assert cancelled.is_set()
    finally:
        await checker.aclose()


@pytest.mark.asyncio
async def test_only_visible_resources_are_batched_before_citation_generation(
    monkeypatch,
):
    seen = []
    allowed, denied = retrieval("allowed"), retrieval("denied")
    monkeypatch.setattr(
        WeaviateVectorRetriever, "query", AsyncMock(return_value=[denied, allowed])
    )

    def handle(request):
        body = json.loads(request.content)
        seen.append(body)
        assert len(body["state"]["results"]) == 1
        assert body["state"]["results"][0]["snippet"] == allowed.chunk.text
        assert "denied" not in request.content.decode()
        return httpx.Response(200, json={"answers": {"r0": answer(0.1)}})

    install_transport(monkeypatch, handle)
    checker = RelevanceChecker(config(resource_enabled=True, resource_threshold=0.5))
    search = ResourceSearch.__new__(ResourceSearch)
    search.relevance = checker
    backend = SimpleNamespace(
        filter_visible_resource_ids=AsyncMock(return_value=["allowed"])
    )
    try:
        result = await search.query(
            "query",
            private_search_tool=PrivateSearchTool(
                namespace_id="ns", visible_resources=[]
            ),
            backend_client=backend,
        )
        assert result == [allowed] and result[0] is allowed
        assert len(seen) == 1
    finally:
        await checker.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("unavailable", [False, True])
async def test_resource_filter_drops_irrelevant_or_preserves_results_on_failure(
    monkeypatch, unavailable
):
    resources = [retrieval("one"), retrieval("two")]
    requests = []

    def handle(request):
        requests.append(request)
        if unavailable:
            return httpx.Response(529)
        return httpx.Response(
            200, json={"answers": {"r0": answer(0.5), "r1": answer(0.7)}}
        )

    install_transport(monkeypatch, handle)
    checker = RelevanceChecker(config(resource_enabled=True, resource_threshold=0.5))
    try:
        filtered = await checker.filter_resources("query", resources)
        assert filtered == (resources if unavailable else [])
        assert len(requests) == 1
    finally:
        await checker.aclose()


@pytest.mark.asyncio
async def test_search_sends_full_chunk_text_deduplicates_and_keeps_messages(
    monkeypatch,
):
    requests = []
    duplicate = search_record("one", text="x" * 5000)
    records = [
        duplicate,
        search_record("one", text="duplicate chunk"),
        message_record("message"),
        search_record("two"),
    ]

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body["state"]["query"] == "query"
        assert [item["id"] for item in body["state"]["results"]] == [0, 1]
        assert body["state"]["results"][0]["title"] == "one"
        assert body["state"]["results"][0]["snippet"] == "x" * 5000
        return httpx.Response(
            200, json={"answers": {"s0": answer(0.1), "s1": answer(0.9)}}
        )

    install_transport(monkeypatch, handle)
    checker = RelevanceChecker(config(search_enabled=True, search_threshold=0.5))
    try:
        result = await checker.filter_search_results("query", records)
        assert result.kept == [duplicate, records[2]]
        assert result.pending == []
        assert len(requests) == 1
    finally:
        await checker.aclose()


@pytest.mark.parametrize(
    "query,title,expected",
    [
        ("NEBULA610X", "a", False),
        ("NEBULA610X", "610-11 NEBULA610X-标题编码.md", True),
        ("笔记", "我的笔记.md", True),
        ("笔记", "笔", False),
        ("第二大脑", "打造第二大脑.md", True),
        ("打造第二大脑笔记", "第二大脑", True),
        ("第二 大脑", "第二大脑.md", True),
        (" query ", "QUERY.md", True),
        ("1.5", "15个提升效率的技巧", False),
        ("a-b", "About the project", False),
        ("C#", "C# 教程", True),
        ("AI", "Chair repair manual", False),
        ("AI", "Email etiquette", False),
        ("Go", "Mango recipes", False),
        ("SQL", "PostgreSQL guide", False),
        ("17", "2017 归档", False),
        ("NEBULA610", "NEBULA610X", False),
        ("AI教程", "CHAIR教程", False),
        ("repair a chair", "AI", False),
        ("AI", "AI notes", True),
        ("AI", "AI_notes.md", True),
        ("AI", "基于 AI 的搜索", True),
        ("AI", "AI教程", True),
        ("Go", "Go语言教程", True),
        ("second brain", "SecondBrain.md", True),
        ("HelloWorld", "Hello World.md", True),
        ("how to use AI", "AI", True),
    ],
)
def test_title_matches_normalizes_and_requires_two_characters(query, title, expected):
    assert _title_matches(query, title) is expected


@pytest.mark.asyncio
async def test_search_title_guard_keeps_matches_in_both_directions(monkeypatch):
    forward = search_record("one", title="打造第二大脑.md")
    reverse = search_record("two", title="第二大脑")
    unrelated = search_record("three", title="a.txt", text="Hello World!")

    def handle(request):
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "answers": {
                    f"s{index}": answer(0.99)
                    for index in range(len(body["state"]["results"]))
                }
            },
        )

    install_transport(monkeypatch, handle)
    checker = RelevanceChecker(config(search_enabled=True, search_threshold=0.8))
    try:
        result = await checker.filter_search_results("第二大脑", [forward, unrelated])
        assert result.kept == [forward]
        assert result.pending == []
        result = await checker.filter_search_results("打造第二大脑笔记", [reverse])
        assert result.kept == [reverse]
        assert result.pending == []
    finally:
        await checker.aclose()


@pytest.mark.asyncio
async def test_search_classifies_kept_off_target_and_pending(monkeypatch):
    kept_record = search_record("kept")
    off_record = search_record("off")
    missing_record = search_record("missing")
    message = message_record("message")
    records = [kept_record, message, off_record, missing_record]

    def handle(request):
        body = json.loads(request.content)
        scores = {"kept": 0.1, "off": 0.9}
        answers = {
            f"s{item['id']}": answer(scores[item["title"]])
            for item in body["state"]["results"]
            if item["title"] in scores
        }
        return httpx.Response(200, json={"answers": answers})

    install_transport(monkeypatch, handle)
    checker = RelevanceChecker(config(search_enabled=True, search_threshold=0.5))
    try:
        result = await checker.filter_search_results("query", records)
    finally:
        await checker.aclose()

    assert isinstance(result, SearchFilterResult)
    assert result.kept == [kept_record, message]
    assert result.pending == [missing_record]


@pytest.mark.asyncio
async def test_search_retries_answer_missing_items_once(monkeypatch):
    records = [search_record("one"), search_record("miss")]
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        retry = len(requests) > 1
        answers = {
            f"s{item['id']}": answer(0.1)
            for item in body["state"]["results"]
            if retry or item["title"] != "miss"
        }
        return httpx.Response(200, json={"answers": answers})

    install_transport(monkeypatch, handle)
    span = install_span(monkeypatch)
    checker = RelevanceChecker(config(search_enabled=True, search_threshold=0.5))
    try:
        result = await checker.filter_search_results("query", records)
    finally:
        await checker.aclose()

    assert result.kept == records
    assert result.pending == []
    assert len(requests) == 2
    assert [item["title"] for item in requests[1]["state"]["results"]] == ["miss"]
    events = [call.args[1] for call in span.add_event.call_args_list]
    assert [item["chunk.decision"] for item in events] == ["kept", "kept"]
    assert len({item["chunk.chunk_id"] for item in events}) == 2
    attributes = span.set_attributes.call_args.args[0]
    assert attributes["relevance.status"] == "ok"
    assert attributes["relevance.kept_count"] == 2
    assert attributes["relevance.pending_count"] == 0
    assert attributes["relevance.retried_batches"] == 1
    assert attributes["relevance.retried_error_batches"] == 0
    assert attributes["relevance.retried_answer_batches"] == 1
    assert attributes["relevance.unexpected_batch_errors"] == 0


@pytest.mark.asyncio
async def test_search_still_missing_after_one_retry_is_pending(monkeypatch):
    records = [search_record("one"), search_record("miss")]
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        answers = {
            f"s{item['id']}": answer(0.1)
            for item in body["state"]["results"]
            if item["title"] != "miss"
        }
        return httpx.Response(200, json={"answers": answers})

    install_transport(monkeypatch, handle)
    span = install_span(monkeypatch)
    checker = RelevanceChecker(config(search_enabled=True, search_threshold=0.5))
    try:
        result = await checker.filter_search_results("query", records)
    finally:
        await checker.aclose()

    assert result.kept == [records[0]]
    assert result.pending == [records[1]]
    assert len(requests) == 2
    events = [call.args[1] for call in span.add_event.call_args_list]
    assert [item["chunk.decision"] for item in events] == ["kept", "pending"]
    assert events[1]["chunk.pending_reason"] == "answer_missing"
    assert len(events) == 2


@pytest.mark.asyncio
async def test_search_follow_up_cut_off_by_deadline_is_pending_deadline(monkeypatch):
    records = [search_record("one"), search_record("two")]
    calls = 0

    async def missing_then_hanging(query, items, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            for index in range(len(items)):
                kwargs["on_decision"](index, None)
            return []
        await asyncio.sleep(10)
        return []

    checker = RelevanceChecker(config(search_timeout_sec=10))
    monkeypatch.setattr(checker, "_filter_items", missing_then_hanging)
    span = install_span(monkeypatch)
    before = asyncio.all_tasks()
    try:
        result = await checker.filter_search_results(
            "query", records, deadline=time.monotonic() + 0.4
        )
    finally:
        await checker.aclose()

    assert calls == 2
    assert asyncio.all_tasks() - before == set()
    assert result.kept == []
    assert result.pending == records
    events = [call.args[1] for call in span.add_event.call_args_list]
    assert [item["chunk.decision"] for item in events] == ["pending", "pending"]
    assert {item["chunk.pending_reason"] for item in events} == {"deadline"}
    attributes = span.set_attributes.call_args.args[0]
    assert attributes["relevance.status"] == "partial"
    assert attributes["relevance.succeeded_batches"] == 1
    assert attributes["relevance.failed_batches"] == 0
    assert attributes["relevance.skipped_batches"] == 1
    assert attributes["relevance.retried_answer_batches"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,unexpected",
    [
        (RelevanceError("relevance_check_failed"), 0),
        (RuntimeError("boom"), 1),
    ],
)
async def test_search_follow_up_failure_keeps_answer_missing(
    monkeypatch, caplog, error, unexpected
):
    records = [search_record("one"), search_record("two")]
    calls = 0

    async def missing_then_failing(query, items, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            for index in range(len(items)):
                kwargs["on_decision"](index, None)
            return []
        raise error

    checker = RelevanceChecker(config(search_enabled=True, search_threshold=0.5))
    monkeypatch.setattr(checker, "_filter_items", missing_then_failing)
    span = install_span(monkeypatch)
    caplog.set_level(
        logging.WARNING, logger="wizard_common.grimoire.retriever.relevance"
    )
    try:
        result = await checker.filter_search_results("query", records)
    finally:
        await checker.aclose()

    assert calls == 2
    assert result.kept == []
    assert result.pending == records
    events = [call.args[1] for call in span.add_event.call_args_list]
    assert {item["chunk.decision"] for item in events} == {"pending"}
    assert {item["chunk.pending_reason"] for item in events} == {"answer_missing"}
    assert any(
        record.name == "wizard_common.grimoire.retriever.relevance"
        and record.levelno == logging.WARNING
        for record in caplog.records
    )
    attributes = span.set_attributes.call_args.args[0]
    assert attributes["relevance.status"] == "partial"
    assert attributes["relevance.failed_batches"] == 1
    assert attributes["relevance.skipped_batches"] == 0
    assert attributes["relevance.retried_answer_batches"] == 1
    assert attributes["relevance.unexpected_batch_errors"] == unexpected


@pytest.mark.asyncio
async def test_search_failed_batch_with_title_protected_items_reports_ok(monkeypatch):
    records = [search_record("one", title="query-plan")]

    def handle(request):
        return httpx.Response(500)

    install_transport(monkeypatch, handle)
    span = install_span(monkeypatch)
    checker = RelevanceChecker(config(search_enabled=True, search_threshold=0.5))
    try:
        result = await checker.filter_search_results("query", records)
    finally:
        await checker.aclose()

    assert result.kept == records
    assert result.pending == []
    events = [call.args[1] for call in span.add_event.call_args_list]
    assert [item["chunk.decision"] for item in events] == ["title_protected"]
    attributes = span.set_attributes.call_args.args[0]
    assert attributes["relevance.status"] == "ok"
    assert attributes["relevance.pending_count"] == 0
    assert attributes["relevance.failed_batches"] == 1


@pytest.mark.asyncio
async def test_search_invalid_answer_becomes_pending_after_retry(monkeypatch):
    bad_record = search_record("bad")
    good_record = search_record("good")
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        answers = {}
        for item in body["state"]["results"]:
            if item["title"] == "good":
                answers[f"s{item['id']}"] = answer(0.1)
            else:
                answers[f"s{item['id']}"] = {"probabilities": {"off_target": 0.9}}
        return httpx.Response(200, json={"answers": answers})

    install_transport(monkeypatch, handle)
    checker = RelevanceChecker(config(search_enabled=True, search_threshold=0.5))
    try:
        result = await checker.filter_search_results("query", [bad_record, good_record])
    finally:
        await checker.aclose()

    assert result.kept == [good_record]
    assert result.pending == [bad_record]
    assert len(requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        lambda request: httpx.Response(529),
        lambda request: httpx.Response(200, text="not JSON"),
        lambda request: httpx.Response(200, json={"answers": []}),
        lambda request: httpx.Response(200, json={}),
    ],
)
async def test_search_unparseable_response_moves_every_result_to_pending(
    monkeypatch, response
):
    records = [search_record("one"), search_record("two")]
    requests = []

    def handle(request):
        requests.append(request)
        return response(request)

    install_transport(monkeypatch, handle)
    checker = RelevanceChecker(config(search_enabled=True, search_threshold=0.5))
    try:
        result = await checker.filter_search_results("query", records)
    finally:
        await checker.aclose()

    assert result.kept == []
    assert result.pending == records
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_search_splits_batches(monkeypatch):
    batch_sizes = []
    records = [search_record(f"r{index}") for index in range(5)]

    def handle(request):
        body = json.loads(request.content)
        batch_sizes.append(len(body["state"]["results"]))
        return httpx.Response(
            200,
            json={
                "answers": {
                    f"s{index}": answer(0.1)
                    for index in range(len(body["state"]["results"]))
                }
            },
        )

    install_transport(monkeypatch, handle)
    checker = RelevanceChecker(
        config(search_enabled=True, search_threshold=0.5, search_batch_size=2)
    )
    try:
        result = await checker.filter_search_results("query", records)
        assert result.kept == records
        assert result.pending == []
        assert sorted(batch_sizes) == [1, 2, 2]
    finally:
        await checker.aclose()


@pytest.mark.asyncio
async def test_search_sends_at_most_sixteen_questions_per_batch(monkeypatch):
    records = [search_record(f"r{index}") for index in range(17)]
    batch_sizes = []

    def handle(request):
        body = json.loads(request.content)
        batch_sizes.append(len(body["state"]["results"]))
        return httpx.Response(
            200,
            json={"answers": {key: answer(0.1) for key in body["questions"]}},
        )

    install_transport(monkeypatch, handle)
    checker = RelevanceChecker(config(search_enabled=True, search_threshold=0.5))
    try:
        result = await checker.filter_search_results("query", records)
    finally:
        await checker.aclose()

    assert result.kept == records
    assert result.pending == []
    assert sorted(batch_sizes) == [1, 16]


@pytest.mark.asyncio
async def test_search_runs_at_most_two_batches_concurrently(monkeypatch):
    records = [search_record(f"r{index}") for index in range(6)]
    active = 0
    max_active = 0

    async def handle(request):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={"answers": {key: answer(0.1) for key in body["questions"]}},
        )

    install_transport(monkeypatch, handle)
    checker = RelevanceChecker(
        config(search_enabled=True, search_threshold=0.5, search_batch_size=1)
    )
    try:
        result = await checker.filter_search_results("query", records)
    finally:
        await checker.aclose()

    assert result.kept == records
    assert result.pending == []
    assert max_active == SEARCH_BATCH_CONCURRENCY


@pytest.mark.asyncio
async def test_search_failed_batch_is_retried_once_then_pending(monkeypatch):
    records = [search_record("one"), search_record("two")]
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(500)

    install_transport(monkeypatch, handle)
    span = install_span(monkeypatch)
    checker = RelevanceChecker(config(search_enabled=True, search_threshold=0.5))
    try:
        result = await checker.filter_search_results("query", records)
    finally:
        await checker.aclose()

    assert result.kept == []
    assert result.pending == records
    assert len(requests) == 2
    events = [call.args[1] for call in span.add_event.call_args_list]
    assert [item["chunk.decision"] for item in events] == ["pending", "pending"]
    assert {item["chunk.pending_reason"] for item in events} == {"error"}
    assert len(events) == 2
    attributes = span.set_attributes.call_args.args[0]
    assert attributes["relevance.status"] == "unavailable"
    assert attributes["relevance.retried_batches"] == 1
    assert attributes["relevance.retried_error_batches"] == 1
    assert attributes["relevance.retried_answer_batches"] == 0
    assert attributes["relevance.unexpected_batch_errors"] == 0
    assert attributes["relevance.failed_batches"] == 1


@pytest.mark.asyncio
async def test_search_batch_timeout_is_retried_once_then_pending(monkeypatch):
    records = [search_record("one"), search_record("two")]
    requests = []

    async def handle(request):
        requests.append(request)
        await asyncio.sleep(10)

    install_transport(monkeypatch, handle)
    span = install_span(monkeypatch)
    checker = RelevanceChecker(
        config(
            search_enabled=True,
            search_threshold=0.5,
            search_timeout_sec=1.0,
            timeout_sec=0.2,
        )
    )
    try:
        async with asyncio.timeout(3):
            result = await checker.filter_search_results("query", records)
    finally:
        await checker.aclose()

    assert result.kept == []
    assert result.pending == records
    assert len(requests) == 2
    events = [call.args[1] for call in span.add_event.call_args_list]
    assert [item["chunk.decision"] for item in events] == ["pending", "pending"]
    assert {item["chunk.pending_reason"] for item in events} == {"error"}
    attributes = span.set_attributes.call_args.args[0]
    assert attributes["relevance.status"] == "unavailable"
    assert attributes["relevance.failed_batches"] == 1
    assert attributes["relevance.retried_batches"] == 1
    assert attributes["relevance.retried_error_batches"] == 1
    assert attributes["relevance.retried_answer_batches"] == 0
    assert attributes["relevance.unexpected_batch_errors"] == 0


@pytest.mark.asyncio
async def test_search_logs_and_counts_unexpected_batch_errors(monkeypatch, caplog):
    records = [search_record("one"), search_record("two")]
    attempts = 0

    async def failing_filter(query, items, **kwargs):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("boom")

    checker = RelevanceChecker(config(search_enabled=True, search_threshold=0.5))
    monkeypatch.setattr(checker, "_filter_items", failing_filter)
    span = install_span(monkeypatch)
    caplog.set_level(
        logging.WARNING, logger="wizard_common.grimoire.retriever.relevance"
    )
    try:
        result = await checker.filter_search_results("query", records)
    finally:
        await checker.aclose()

    assert attempts == 2
    assert result.kept == []
    assert result.pending == records
    warnings = [
        record
        for record in caplog.records
        if record.name == "wizard_common.grimoire.retriever.relevance"
    ]
    assert len(warnings) == 2
    assert all(
        "Unexpected relevance batch failure" in item.message for item in warnings
    )
    assert all("RuntimeError" in item.getMessage() for item in warnings)
    assert all("boom" in item.getMessage() for item in warnings)
    attributes = span.set_attributes.call_args.args[0]
    assert attributes["relevance.status"] == "unavailable"
    assert attributes["relevance.unexpected_batch_errors"] == 2
    assert attributes["relevance.retried_error_batches"] == 1


@pytest.mark.asyncio
async def test_search_records_each_chunk_decision_as_a_span_event(monkeypatch):
    kept_record = search_record("kept", text="kept chunk text")
    off_target_record = search_record("off", text="off chunk text")
    protected_record = search_record(
        "protected", title="第二大脑", text="protected chunk text"
    )
    missing_record = search_record("missing", text="missing chunk text")
    records = [kept_record, off_target_record, protected_record, missing_record]

    def handle(request):
        body = json.loads(request.content)
        scores = {"kept": 0.1, "off": 0.9, "第二大脑": 0.9}
        answers = {
            f"s{item['id']}": answer(scores[item["title"]])
            for item in body["state"]["results"]
            if item["title"] in scores
        }
        return httpx.Response(200, json={"answers": answers})

    install_transport(monkeypatch, handle)
    span = install_span(monkeypatch)
    checker = RelevanceChecker(config(search_enabled=True, search_threshold=0.8))
    try:
        result = await checker.filter_search_results("第二大脑", records)
    finally:
        await checker.aclose()

    assert result.kept == [kept_record, protected_record]
    assert result.pending == [missing_record]
    calls = span.add_event.call_args_list
    assert len(calls) == 4
    attributes = []
    for call in calls:
        assert call.args[0] == "relevance.chunk"
        attributes.append(call.args[1])

    assert [item["chunk.decision"] for item in attributes] == [
        "kept",
        "off_target",
        "title_protected",
        "pending",
    ]
    assert attributes[3]["chunk.pending_reason"] == "answer_missing"
    assert [item["chunk.index"] for item in attributes] == [0, 1, 2, 3]
    assert [item["chunk.batch_index"] for item in attributes] == [0, 0, 0, 0]
    assert [item["chunk.resource_id"] for item in attributes] == [
        "kept",
        "off",
        "protected",
        "missing",
    ]
    for record, item in zip(records, attributes):
        assert item["chunk.chunk_id"] == record.chunk.chunk_id
        assert item["chunk.chunk_type"] == "doc"
        assert item["chunk.text_length"] == len(record.chunk.text)
    assert attributes[0]["chunk.title"] == "kept"
    assert attributes[2]["chunk.title"] == "第二大脑"

    assert attributes[0]["chunk.off_target_score"] == 0.1
    assert attributes[0]["chunk.relevant_score"] == pytest.approx(0.9)
    assert attributes[1]["chunk.off_target_score"] == 0.9
    assert attributes[1]["chunk.relevant_score"] == pytest.approx(0.1)
    assert attributes[2]["chunk.off_target_score"] == 0.9
    assert attributes[2]["chunk.relevant_score"] == pytest.approx(0.1)
    assert "chunk.off_target_score" not in attributes[3]
    assert "chunk.relevant_score" not in attributes[3]
    assert all("chunk.fail_open_reason" not in item for item in attributes)
    assert "chunk.pending_reason" not in attributes[0]
    assert "chunk.pending_reason" not in attributes[1]
    assert "chunk.pending_reason" not in attributes[2]

    # The query only reaches the trace as the protected chunk's title; chunk
    # text never does.
    for item in attributes:
        assert "chunk.text" not in item and "query" not in item
        assert "chunk.text_length" in item
    for item in (attributes[0], attributes[1], attributes[3]):
        assert "第二大脑" not in str(item)
    for record in records:
        assert record.chunk.text not in str(attributes)


@pytest.mark.asyncio
async def test_search_title_protection_survives_batch_failure(monkeypatch):
    protected_record = search_record("protected", title="query-plan")
    plain_record = search_record("plain")
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(500)

    install_transport(monkeypatch, handle)
    span = install_span(monkeypatch)
    checker = RelevanceChecker(config(search_enabled=True, search_threshold=0.5))
    try:
        result = await checker.filter_search_results(
            "query", [protected_record, plain_record]
        )
    finally:
        await checker.aclose()

    assert result.kept == [protected_record]
    assert result.pending == [plain_record]
    assert len(requests) == 2
    events = [call.args[1] for call in span.add_event.call_args_list]
    assert [item["chunk.decision"] for item in events] == [
        "title_protected",
        "pending",
    ]
    assert events[1]["chunk.pending_reason"] == "error"
    assert "chunk.pending_reason" not in events[0]
    attributes = span.set_attributes.call_args.args[0]
    assert attributes["relevance.status"] == "unavailable"


@pytest.mark.asyncio
async def test_search_emits_one_final_decision_event_per_chunk(monkeypatch):
    records = [search_record("r0"), search_record("r1"), search_record("r2")]
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        item = body["state"]["results"][0]
        if item["title"] == "r1":
            return httpx.Response(500)
        score = 0.1 if item["title"] == "r0" else 0.9
        return httpx.Response(200, json={"answers": {"s0": answer(score)}})

    install_transport(monkeypatch, handle)
    span = install_span(monkeypatch)
    checker = RelevanceChecker(
        config(search_enabled=True, search_threshold=0.5, search_batch_size=1)
    )
    try:
        result = await checker.filter_search_results("query", records)
    finally:
        await checker.aclose()

    assert result.kept == [records[0]]
    assert result.pending == [records[1]]
    # The failed batch is retried once.
    assert len(requests) == 4
    calls = span.add_event.call_args_list
    assert len(calls) == 3
    events = []
    for call in calls:
        assert call.args[0] == "relevance.chunk"
        events.append(call.args[1])

    assert [item["chunk.decision"] for item in events] == [
        "kept",
        "pending",
        "off_target",
    ]
    assert [item["chunk.index"] for item in events] == [0, 1, 2]
    assert [item["chunk.batch_index"] for item in events] == [0, 1, 2]
    assert [item.get("chunk.pending_reason") for item in events] == [
        None,
        "error",
        None,
    ]
    assert len({item["chunk.chunk_id"] for item in events}) == 3


@pytest.mark.asyncio
async def test_search_preserves_input_order_across_concurrent_batches(monkeypatch):
    records = [search_record(f"r{index}") for index in range(4)]

    async def handle(request):
        body = json.loads(request.content)
        item = body["state"]["results"][0]
        if item["title"] == "r0":
            await asyncio.sleep(0.05)
        return httpx.Response(200, json={"answers": {"s0": answer(0.1)}})

    install_transport(monkeypatch, handle)
    checker = RelevanceChecker(
        config(search_enabled=True, search_threshold=0.5, search_batch_size=1)
    )
    try:
        result = await checker.filter_search_results("query", records)
    finally:
        await checker.aclose()

    assert result.kept == records
    assert result.pending == []


@pytest.mark.asyncio
async def test_search_reports_status_and_counts_for_partial_batches(monkeypatch):
    records = [
        search_record("r0"),
        search_record("r1"),
        search_record("r2"),
        search_record("r3"),
        search_record("query-plan"),
    ]
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        titles = [item["title"] for item in body["state"]["results"]]
        if "r2" in titles or "query-plan" in titles:
            return httpx.Response(500)
        return httpx.Response(
            200, json={"answers": {"s0": answer(0.1), "s1": answer(0.9)}}
        )

    install_transport(monkeypatch, handle)
    span = install_span(monkeypatch)
    checker = RelevanceChecker(
        config(search_enabled=True, search_threshold=0.5, search_batch_size=2)
    )
    try:
        result = await checker.filter_search_results("query", records)
    finally:
        await checker.aclose()

    assert result.kept == [records[0], records[4]]
    assert result.pending == [records[2], records[3]]
    # Three initial batches plus one retry each for the two failed batches.
    assert len(requests) == 5
    events = [call.args[1] for call in span.add_event.call_args_list]
    assert len(events) == 5
    assert [item["chunk.decision"] for item in events] == [
        "kept",
        "off_target",
        "pending",
        "pending",
        "title_protected",
    ]
    assert [item.get("chunk.pending_reason") for item in events] == [
        None,
        None,
        "error",
        "error",
        None,
    ]
    attributes = span.set_attributes.call_args.args[0]
    assert attributes == {
        "relevance.status": "partial",
        "relevance.input_count": 5,
        "relevance.chunk_count": 5,
        "relevance.kept_count": 2,
        "relevance.pending_count": 2,
        "relevance.off_target_count": 1,
        "relevance.threshold": 0.5,
        "relevance.succeeded_batches": 1,
        "relevance.failed_batches": 2,
        "relevance.skipped_batches": 0,
        "relevance.retried_batches": 2,
        "relevance.retried_error_batches": 2,
        "relevance.retried_answer_batches": 0,
        "relevance.unexpected_batch_errors": 0,
        "relevance.batch_concurrency": SEARCH_BATCH_CONCURRENCY,
    }


@pytest.mark.asyncio
async def test_search_records_all_chunk_events_in_overflow_spans(monkeypatch):
    from wizard_common.grimoire.retriever import trace_events

    overflow = MagicMock()
    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value.__enter__.return_value = overflow
    monkeypatch.setattr(trace_events, "tracer", mock_tracer)
    records = [search_record(f"r{index}") for index in range(200)]

    def handle(request):
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={"answers": {key: answer(0.1) for key in body["questions"]}},
        )

    install_transport(monkeypatch, handle)
    span = install_span(monkeypatch)
    checker = RelevanceChecker(
        config(
            search_enabled=True,
            search_threshold=0.8,
            search_timeout_sec=10,
            timeout_sec=10,
        )
    )
    try:
        result = await checker.filter_search_results("query", records)
    finally:
        await checker.aclose()

    assert result.kept == records
    assert result.pending == []
    calls = span.add_event.call_args_list + overflow.add_event.call_args_list
    assert len(span.add_event.call_args_list) == 100
    assert len(overflow.add_event.call_args_list) == 100
    assert [call.args[0] for call in calls] == ["relevance.chunk"] * 200
    assert [call.args[1]["chunk.index"] for call in calls] == list(range(200))
    assert calls[-1].args[1]["chunk.batch_index"] == 12


@pytest.mark.asyncio
async def test_search_skips_chunk_events_when_not_recording(monkeypatch):
    records = [search_record("keep"), search_record("drop")]

    def handle(request):
        return httpx.Response(
            200, json={"answers": {"s0": answer(0.1), "s1": answer(0.9)}}
        )

    install_transport(monkeypatch, handle)
    span = install_span(monkeypatch)
    span.is_recording.return_value = False
    checker = RelevanceChecker(config(search_enabled=True, search_threshold=0.8))
    try:
        result = await checker.filter_search_results("query", records)
    finally:
        await checker.aclose()

    assert result.kept == [records[0]]
    assert result.pending == []
    span.add_event.assert_not_called()


@pytest.mark.asyncio
async def test_search_honors_a_caller_deadline_instead_of_resetting_the_budget(
    monkeypatch,
):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"answers": {"s0": answer(0.1)}})

    install_transport(monkeypatch, handle)
    span = install_span(monkeypatch)
    plain_record = search_record("one")
    protected_record = search_record("two", title="query-plan")
    checker = RelevanceChecker(config(search_timeout_sec=10))
    try:
        result = await checker.filter_search_results(
            "query",
            [plain_record, protected_record],
            deadline=time.monotonic() - 1,
        )
    finally:
        await checker.aclose()

    assert result.kept == [protected_record]
    assert result.pending == [plain_record]
    assert requests == []
    events = [call.args[1] for call in span.add_event.call_args_list]
    assert [item["chunk.decision"] for item in events] == [
        "pending",
        "title_protected",
    ]
    assert events[0]["chunk.pending_reason"] == "deadline"


@pytest.mark.asyncio
async def test_search_cancels_batches_still_in_flight_at_the_shared_deadline(
    monkeypatch,
):
    records = [search_record("one"), search_record("two")]
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def hanging_filter(query, items, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    clients = []
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: clients.append(kwargs))
    span = install_span(monkeypatch)
    checker = RelevanceChecker(config(search_timeout_sec=10))
    monkeypatch.setattr(checker, "_filter_items", hanging_filter)
    before = asyncio.all_tasks()
    try:
        result = await checker.filter_search_results(
            "query", records, deadline=time.monotonic() + 0.5
        )
    finally:
        await checker.aclose()

    assert started.is_set()
    assert cancelled.is_set()
    assert asyncio.all_tasks() - before == set()
    assert clients == []
    assert result.kept == []
    assert result.pending == records
    events = [call.args[1] for call in span.add_event.call_args_list]
    assert [item["chunk.decision"] for item in events] == ["pending", "pending"]
    assert {item["chunk.pending_reason"] for item in events} == {"deadline"}
    attributes = span.set_attributes.call_args.args[0]
    assert attributes["relevance.status"] == "unavailable"
    assert attributes["relevance.succeeded_batches"] == 0
    assert attributes["relevance.failed_batches"] == 0
    assert attributes["relevance.skipped_batches"] == 1
    assert attributes["relevance.retried_batches"] == 0


@pytest.mark.asyncio
async def test_search_offsets_chunk_indices_across_pages(monkeypatch):
    def handle(request):
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={"answers": {key: answer(0.1) for key in body["questions"]}},
        )

    install_transport(monkeypatch, handle)
    span = install_span(monkeypatch)
    records = [search_record("one"), search_record("two")]
    checker = RelevanceChecker(config(search_batch_size=1))
    try:
        result = await checker.filter_search_results("query", records, index_offset=5)
    finally:
        await checker.aclose()

    assert result.kept == records
    assert result.pending == []
    chunk_events = [
        call.args[1]
        for call in span.add_event.call_args_list
        if call.args[0] == "relevance.chunk"
    ]
    assert [attributes["chunk.index"] for attributes in chunk_events] == [5, 6]
    assert [attributes["chunk.batch_index"] for attributes in chunk_events] == [0, 1]


@pytest.mark.asyncio
async def test_evaluate_records_and_chains_http_status_error(monkeypatch):
    install_transport(monkeypatch, lambda request: httpx.Response(500))
    span = install_evaluate_span(monkeypatch)
    checker = RelevanceChecker(config())
    try:
        with pytest.raises(
            RelevanceError, match="^relevance_check_failed$"
        ) as exc_info:
            await checker.check_bing("query", hits())
    finally:
        await checker.aclose()

    cause = exc_info.value.__cause__
    assert isinstance(cause, httpx.HTTPStatusError)
    assert "500" in str(cause)
    span.set_attribute.assert_any_call("error.type", "HTTPStatusError")
    span.set_attribute.assert_any_call("error.message", "500 Internal Server Error")


@pytest.mark.asyncio
async def test_evaluate_sanitizes_invalid_json_body(monkeypatch):
    install_transport(
        monkeypatch,
        lambda request: httpx.Response(200, text="Private remote diagnostic"),
    )
    span = install_evaluate_span(monkeypatch)
    checker = RelevanceChecker(config())
    try:
        with pytest.raises(
            RelevanceError, match="^relevance_check_failed$"
        ) as exc_info:
            await checker.check_bing("query", hits())
    finally:
        await checker.aclose()

    cause = exc_info.value.__cause__
    assert isinstance(cause, ValueError)
    assert str(cause) == "invalid JSON response"
    assert "Private remote diagnostic" not in str(cause)
    traceback_text = "".join(
        traceback.format_exception(
            type(exc_info.value), exc_info.value, exc_info.value.__traceback__
        )
    )
    assert "Private remote diagnostic" not in traceback_text
    span.set_attribute.assert_any_call("error.type", "JSONDecodeError")
    span.set_attribute.assert_any_call("error.message", "invalid JSON response")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query,title",
    [("AI", "Chair repair manual"), ("AI", "Email etiquette"), ("Go", "Mango recipes")],
)
async def test_search_english_substrings_do_not_override_model_rejection(
    monkeypatch, query, title
):
    install_transport(
        monkeypatch,
        lambda request: httpx.Response(200, json={"answers": {"s0": answer(0.99)}}),
    )
    checker = RelevanceChecker(config(search_enabled=True, search_threshold=0.5))
    try:
        result = await checker.filter_search_results(
            query, [search_record("unrelated", title=title)]
        )
        assert result.kept == []
        assert result.pending == []
    finally:
        await checker.aclose()
