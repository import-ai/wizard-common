import asyncio
import time
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError
from test_weaviate_hybrid_fusion import (
    CONDITION,
    FakeMetadata,
    FakeObject,
    QueryResponse,
    make_db,
)
from weaviate.collections.filters import _FilterToREST

from wizard_common.grimoire.config import SearchDiagnosticsConfig
from wizard_common.grimoire.retriever import weaviate_vector_db as vector_module
from wizard_common.grimoire.retriever.search_diagnostics import (
    distance_metric,
    record_candidates,
)


def attributes(span):
    result = {}
    for call in span.set_attributes.call_args_list:
        result.update(call.args[0])
    for call in span.set_attribute.call_args_list:
        result[call.args[0]] = call.args[1]
    return result


def setup_probe():
    first = FakeObject(
        uuid.UUID(int=1),
        {"chunk_resource_id": "r1", "chunk_id": "c1"},
        FakeMetadata(score=0.01, distance=0.25),
    )
    second = FakeObject(
        uuid.UUID(int=2),
        {"chunk_resource_id": "r2", "chunk_id": "c2"},
        FakeMetadata(score=0.02, distance=0.4),
    )
    collection = SimpleNamespace(
        query=SimpleNamespace(
            near_vector=AsyncMock(return_value=QueryResponse([first, second])),
            bm25=AsyncMock(return_value=QueryResponse([first])),
        )
    )
    span = MagicMock()
    args = dict(
        collection=collection,
        span=span,
        config=SearchDiagnosticsConfig(enabled=True),
        query="NEBULA610X",
        vector=[1.0, 0.0],
        filters=CONDITION.to_weaviate_filters(),
        word_properties=["chunk_title", "chunk_text"],
        selected=[
            (str(o.uuid), {"properties": o.properties, "score": 0.03})
            for o in [first, second]
        ],
        hybrid_objects=[first, second],
        gse_objects=[second],
        gse_enabled=True,
        gse_failed=False,
        metric="cosine",
        offset=100,
    )
    return args, collection, span


@pytest.mark.asyncio
async def test_probe_maps_raw_scores_by_uuid_and_preserves_missing_states():
    args, collection, span = setup_probe()
    await record_candidates(**args)
    events = [call.args[1] for call in span.add_event.call_args_list]
    assert events[0]["candidate.rank"] == 101
    assert events[0]["candidate.vector_distance"] == 0.25
    assert events[0]["candidate.word_bm25_score"] == 0.01
    assert events[0]["candidate.hybrid_score"] == 0.01
    assert events[0]["candidate.final_score"] == 0.03
    assert events[0]["candidate.gse_bm25_status"] == "not_in_recall"
    assert events[1]["candidate.gse_bm25_score"] == 0.02
    assert events[1]["candidate.word_bm25_status"] == "not_matched"
    assert "candidate.word_bm25_score" not in events[1]
    assert attributes(span)["search.diagnostics.status"] == "ok"
    for method in [collection.query.near_vector, collection.query.bm25]:
        kwargs = method.call_args.kwargs
        assert kwargs["limit"] == 2
        assert kwargs["return_properties"] is False
        wire = str(_FilterToREST.convert(kwargs["filters"]))
        assert "namespace_id" in wire and "space" in wire
        assert all(uuid in wire for uuid, _ in args["selected"])
    assert collection.query.near_vector.call_args.kwargs["near_vector"] == [1.0, 0.0]
    assert all(
        isinstance(value, (str, int, float, bool))
        for event in events
        for value in event.values()
    )
    assert not any("text" in key or "url" in key for event in events for key in event)


@pytest.mark.asyncio
async def test_probes_run_concurrently_and_timeout_without_discarding_other_leg():
    args, collection, span = setup_probe()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def wait_forever(**kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    original = collection.query.bm25.return_value

    async def word(**kwargs):
        await started.wait()
        return original

    collection.query.near_vector.side_effect = wait_forever
    collection.query.bm25.side_effect = word
    args["config"] = SearchDiagnosticsConfig(enabled=True, timeout_sec=0.01)
    await asyncio.wait_for(record_candidates(**args), timeout=1)
    assert cancelled.is_set()
    event = span.add_event.call_args_list[0].args[1]
    assert event["candidate.vector_status"] == "timeout"
    assert event["candidate.word_bm25_score"] == 0.01
    assert attributes(span)["search.diagnostics.status"] == "partial"


@pytest.mark.asyncio
async def test_expired_request_budget_starts_no_queries():
    args, collection, span = setup_probe()
    args["deadline"] = time.monotonic() - 1
    await record_candidates(**args)
    collection.query.near_vector.assert_not_called()
    collection.query.bm25.assert_not_called()
    assert span.add_event.call_args.args[1]["candidate.vector_status"] == "deadline"


@pytest.mark.asyncio
async def test_probe_error_does_not_leak_exception_body_or_invent_score():
    args, collection, span = setup_probe()
    collection.query.bm25.side_effect = RuntimeError("secret-request-body")
    await record_candidates(**args)
    event = span.add_event.call_args_list[0].args[1]
    assert event["candidate.word_bm25_status"] == "error"
    assert "candidate.word_bm25_score" not in event
    assert "secret-request-body" not in str(span.mock_calls)


@pytest.mark.asyncio
async def test_cap_filters_only_recorded_slice_and_reports_truncation():
    args, collection, span = setup_probe()
    args["config"] = SearchDiagnosticsConfig(enabled=True, max_candidates=1)
    await record_candidates(**args)
    assert span.add_event.call_count == 1
    assert collection.query.near_vector.call_args.kwargs["limit"] == 1
    assert attributes(span)["search.diagnostics.truncated_count"] == 1


@pytest.mark.asyncio
async def test_cancellation_propagates_and_stops_both_queries():
    args, collection, _ = setup_probe()
    entered = 0
    ready = asyncio.Event()
    finished = []

    async def wait(**kwargs):
        nonlocal entered
        entered += 1
        if entered == 2:
            ready.set()
        try:
            await asyncio.Event().wait()
        finally:
            finished.append(True)

    collection.query.near_vector.side_effect = wait
    collection.query.bm25.side_effect = wait
    task = asyncio.create_task(record_candidates(**args))
    await asyncio.wait_for(ready.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(finished) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "enabled,recording", [(False, True), (True, False), (True, True)]
)
async def test_integration_keeps_ranking_and_embedding_count(enabled, recording):
    obj = FakeObject(uuid.UUID(int=1), {"chunk_title": "NEBULA"}, FakeMetadata(0.02))
    db, collection = make_db(hybrid_objects=[obj])
    db.config = SimpleNamespace(
        search_diagnostics=SearchDiagnosticsConfig(enabled=enabled)
    )
    db._distance_metric = "cosine"
    collection.query.near_vector.return_value = QueryResponse([])
    span = MagicMock()
    span.is_recording.return_value = recording
    with patch.object(vector_module.trace, "get_current_span", return_value=span):
        hits = await db._hybrid_query("space", "NEBULA", CONDITION, limit=20)
    assert hits == [(obj.properties, 0.02)]
    db._embed.assert_awaited_once()
    collection.query.hybrid.assert_awaited_once()
    if enabled and recording:
        collection.query.near_vector.assert_awaited_once()
        collection.query.bm25.assert_awaited_once()
    else:
        collection.query.near_vector.assert_not_called()
        collection.query.bm25.assert_not_called()


def test_config_bounds_and_named_vector_metric():
    assert SearchDiagnosticsConfig().enabled
    for kwargs in [{"max_candidates": 101}, {"timeout_sec": 0}, {"timeout_sec": 2}]:
        with pytest.raises(ValidationError):
            SearchDiagnosticsConfig(**kwargs)
    config = SimpleNamespace(
        vector_config={
            "default": SimpleNamespace(
                vector_index_config=SimpleNamespace(distance_metric="cosine")
            )
        }
    )
    assert distance_metric(config) == "cosine"
    assert distance_metric(SimpleNamespace()) == "unknown"


@pytest.mark.asyncio
async def test_invalid_metadata_is_missing_not_no_match_or_zero():
    args, collection, span = setup_probe()
    obj = FakeObject(uuid.UUID(int=1), {}, FakeMetadata(score=float("nan")))
    collection.query.bm25.return_value = QueryResponse([obj])
    collection.query.near_vector.return_value = QueryResponse([obj])
    await record_candidates(**args)
    event = span.add_event.call_args_list[0].args[1]
    assert event["candidate.word_bm25_status"] == "missing"
    assert event["candidate.vector_status"] == "missing"
    assert "candidate.word_bm25_score" not in event
    assert "candidate.vector_distance" not in event
    assert attributes(span)["search.diagnostics.status"] == "partial"


@pytest.mark.asyncio
async def test_empty_slice_does_not_probe():
    args, collection, span = setup_probe()
    args["selected"] = []
    await record_candidates(**args)
    assert attributes(span)["search.diagnostics.status"] == "empty"
    span.add_event.assert_not_called()
    collection.query.bm25.assert_not_called()
    collection.query.near_vector.assert_not_called()


@pytest.mark.asyncio
async def test_empty_query_preserves_fetch_path_without_diagnostics():
    obj = FakeObject(uuid.UUID(int=1), {}, FakeMetadata(0.01))
    db, collection = make_db()
    collection.query.fetch_objects.return_value = QueryResponse([obj])
    db.config = SimpleNamespace(
        search_diagnostics=SearchDiagnosticsConfig(enabled=True)
    )
    hits = await db._hybrid_query("space", "", CONDITION, limit=20)
    assert hits == [(obj.properties, 0.01)]
    collection.query.near_vector.assert_not_called()
    collection.query.bm25.assert_not_called()
    db._embed.assert_not_called()


@pytest.mark.asyncio
async def test_diagnostics_internal_failure_cannot_replace_search_results():
    obj = FakeObject(uuid.UUID(int=1), {"chunk_title": "NEBULA"}, FakeMetadata(0.02))
    db, collection = make_db(hybrid_objects=[obj])
    db.config = SimpleNamespace(
        search_diagnostics=SearchDiagnosticsConfig(enabled=True)
    )
    db._distance_metric = "cosine"
    span = MagicMock()
    with (
        patch.object(vector_module.trace, "get_current_span", return_value=span),
        patch.object(vector_module, "record_candidates", side_effect=RuntimeError),
    ):
        hits = await db._hybrid_query("space", "NEBULA", CONDITION, limit=20)
    assert hits == [(obj.properties, 0.02)]
    assert attributes(span)["search.diagnostics.status"] == "error"
