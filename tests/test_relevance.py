import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from common.config_loader import Loader
from pydantic import ValidationError

from wizard_common.grimoire.config import (
    BingSearchConfig,
    RelevanceConfig,
    SearXNGConfig,
    ToolsConfig,
)
from wizard_common.grimoire.entity.chunk import Chunk, ChunkType, ResourceChunkRetrieval
from wizard_common.grimoire.entity.retrieval import Score
from wizard_common.grimoire.entity.tools import PrivateSearchTool
from wizard_common.grimoire.retriever.relevance import (
    BING_QUESTION,
    RelevanceChecker,
    RelevanceError,
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
    checker = RelevanceChecker(RelevanceConfig())
    await checker.check_bing("query", hits())
    resources = [retrieval("keep")]
    assert await checker.filter_resources("query", resources) is resources
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
