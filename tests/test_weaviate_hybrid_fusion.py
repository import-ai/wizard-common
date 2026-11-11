import asyncio
import uuid
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest
from weaviate.collections.filters import _FilterToREST
from weaviate.exceptions import WeaviateQueryError

from wizard_common.grimoire.config import SearchThresholdsConfig, VectorConfig
from wizard_common.grimoire.entity.tools import Condition
from wizard_common.grimoire.retriever.search_retrieval import ALPHA, RRF_K
from wizard_common.grimoire.retriever.weaviate_vector_db import WeaviateVectorDB


@dataclass
class FakeMetadata:
    score: float | None = None
    distance: float | None = None


@dataclass
class FakeObject:
    uuid: uuid.UUID
    properties: dict
    metadata: FakeMetadata | None = None


@dataclass
class QueryResponse:
    objects: list


CONDITION = Condition(namespace_id="space")


def obj(index, *, distance=None, score=None):
    return FakeObject(
        uuid.UUID(int=index),
        {"chunk_title": f"title-{index}"},
        FakeMetadata(score=score, distance=distance),
    )


def make_db(vector_objects=(), word_objects=(), gse_objects=(), thresholds=None):
    db = object.__new__(WeaviateVectorDB)
    db.config = VectorConfig(
        embedding={}, search_thresholds=thresholds or SearchThresholdsConfig()
    )
    db._distance_metric = "cosine"
    collection = AsyncMock()

    async def vector(**kwargs):
        distance = kwargs.get("distance")
        objects = [
            o
            for o in vector_objects
            if distance is None
            or (o.metadata.distance is not None and o.metadata.distance <= distance)
        ]
        return QueryResponse(objects[: kwargs["limit"]])

    async def bm25(**kwargs):
        objects = (
            gse_objects
            if kwargs["query_properties"][0].endswith("_gse")
            else word_objects
        )
        return QueryResponse(list(objects)[: kwargs["limit"]])

    collection.query.near_vector.side_effect = vector
    collection.query.bm25.side_effect = bm25
    db._get_shard = AsyncMock(return_value=collection)
    db._embed = AsyncMock(return_value=[[1.0, 0.0]])
    return db, collection


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["NEBULA610X", "2024", "!!!"])
async def test_ascii_query_runs_two_native_legs_once_without_diagnostic_probes(query):
    target = obj(1, distance=0.4)
    db, collection = make_db(vector_objects=[target])
    window = await db._retrieve_window("space", query, CONDITION, limit=20)
    assert [c.uuid for c in window.candidates] == [str(target.uuid)]
    collection.query.hybrid.assert_not_called()
    collection.query.near_vector.assert_awaited_once()
    collection.query.bm25.assert_awaited_once()
    db._embed.assert_awaited_once_with(query)
    assert collection.query.near_vector.call_args.kwargs["distance"] == 0.5
    assert collection.query.near_vector.call_args.kwargs["limit"] == 21


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["标题编码", "NEBULA610X 标题编码"])
async def test_cjk_leg_receives_only_cjk_and_has_independent_threshold(query):
    db, collection = make_db(gse_objects=[obj(1, score=0.1)])
    window = await db._retrieve_window("space", query, CONDITION, limit=20)
    calls = collection.query.bm25.call_args_list
    assert calls[0].kwargs["query"] == query
    assert calls[1].kwargs["query"] == "标题编码"
    assert window.candidates[0].passed_legs == ["gse"]
    assert window.candidates[0].gse_bm25_score == 0.1


@pytest.mark.asyncio
async def test_union_includes_vector_only_keyword_only_and_cjk_only():
    db, _ = make_db(
        vector_objects=[obj(1, distance=0.5)],
        word_objects=[obj(2, score=1)],
        gse_objects=[obj(3, score=0.1)],
    )
    window = await db._retrieve_window("space", "query 中文", CONDITION, limit=20)
    assert {c.uuid: c.passed_legs for c in window.candidates} == {
        str(uuid.UUID(int=1)): ["vector"],
        str(uuid.UUID(int=2)): ["word"],
        str(uuid.UUID(int=3)): ["gse"],
    }


@pytest.mark.asyncio
async def test_threshold_edges_and_failed_branch_do_not_contribute():
    db, _ = make_db(
        vector_objects=[obj(1, distance=0.5), obj(2, distance=0.50001)],
        word_objects=[obj(3, score=1.0), obj(4, score=0.99999)],
    )
    window = await db._retrieve_window("space", "query", CONDITION, limit=20)
    assert {c.uuid for c in window.candidates} == {
        str(uuid.UUID(int=1)),
        str(uuid.UUID(int=3)),
    }
    assert all(c.final_score == ALPHA / RRF_K for c in window.candidates)


@pytest.mark.asyncio
async def test_same_uuid_fuses_only_qualifying_legs_and_preserves_raw_low_score():
    vector = obj(1, distance=0.4)
    word = obj(1, score=0.2)
    db, _ = make_db(vector_objects=[vector], word_objects=[word])
    window = await db._retrieve_window("space", "query", CONDITION, limit=20)
    c = window.candidates[0]
    assert c.passed_legs == ["vector"]
    assert c.returned_legs == ["vector", "word"]
    assert c.word_bm25_score == 0.2
    assert c.final_score == ALPHA / RRF_K


@pytest.mark.asyncio
async def test_matching_two_legs_outranks_one_and_ties_use_uuid():
    db, _ = make_db(
        vector_objects=[obj(3, distance=0.3), obj(2, distance=0.3)],
        word_objects=[obj(1, score=6), obj(2, score=6)],
    )
    window = await db._retrieve_window("space", "query", CONDITION, limit=20)
    assert [c.uuid for c in window.candidates] == [
        str(uuid.UUID(int=i)) for i in [2, 1, 3]
    ]


@pytest.mark.asyncio
async def test_disabling_thresholds_and_custom_gse_minimum():
    thresholds = SearchThresholdsConfig(
        max_distance=None, word_min_bm25_score=None, gse_min_bm25_score=2
    )
    db, _ = make_db(
        vector_objects=[obj(1, distance=0.9)],
        word_objects=[obj(2, score=0.01)],
        gse_objects=[obj(3, score=1)],
        thresholds=thresholds,
    )
    window = await db._retrieve_window("space", "query 中文", CONDITION, limit=20)
    assert {c.uuid for c in window.candidates} == {
        str(uuid.UUID(int=i)) for i in [1, 2]
    }


@pytest.mark.asyncio
async def test_weak_keyword_tail_exhausts_branch_without_fabricating_more():
    objects = [obj(1, score=6)] + [obj(i, score=0.2) for i in range(2, 150)]
    db, _ = make_db(word_objects=objects)
    window = await db._retrieve_window("space", "query", CONDITION, limit=100)
    assert len(window.candidates) == 1
    assert not window.has_more


@pytest.mark.asyncio
async def test_lookahead_and_last_page_are_explicit():
    db, collection = make_db(vector_objects=[obj(i, distance=0.4) for i in range(1, 4)])
    first = await db._retrieve_window("space", "query", CONDITION, limit=2)
    second = await db._retrieve_window("space", "query", CONDITION, limit=2, offset=2)
    assert len(first.candidates) == 2 and first.has_more
    assert len(second.candidates) == 1 and not second.has_more
    assert collection.query.near_vector.call_args.kwargs["limit"] == 5


@pytest.mark.asyncio
async def test_all_legs_receive_identical_scope_filters():
    db, collection = make_db()
    condition = Condition(
        namespace_id="space",
        user_id="user",
        resource_ids=["r1"],
        conversation_ids=["current"],
        exclude_conversation_ids=["old"],
    )
    await db._retrieve_window("space", "query 中文", condition, limit=3)
    expected = _FilterToREST.convert(condition.to_weaviate_filters())
    for call in [
        collection.query.near_vector.call_args,
        *collection.query.bm25.call_args_list,
    ]:
        assert _FilterToREST.convert(call.kwargs["filters"]) == expected


@pytest.mark.asyncio
async def test_one_failed_leg_cannot_qualify_but_other_leg_survives():
    db, collection = make_db(word_objects=[obj(1, score=2)])
    db._embed.side_effect = RuntimeError("embedding failed")
    window = await db._retrieve_window("space", "query", CONDITION, limit=3)
    assert window.candidates[0].passed_legs == ["word"]
    assert window.branch_status == {"gse": "disabled", "vector": "error", "word": "ok"}
    collection.query.near_vector.assert_not_called()


@pytest.mark.asyncio
async def test_all_failed_legs_raise_and_missing_tenant_is_empty():
    db, collection = make_db()
    db._embed.side_effect = RuntimeError("embedding failed")
    collection.query.bm25.side_effect = RuntimeError("word failed")
    with pytest.raises(RuntimeError, match="embedding failed"):
        await db._retrieve_window("space", "query", CONDITION, limit=3)
    error = WeaviateQueryError("tenant not found", "grpc")
    db._embed.side_effect = error
    collection.query.bm25.side_effect = error
    assert (
        await db._retrieve_window("space", "query", CONDITION, limit=3)
    ).candidates == []


@pytest.mark.asyncio
async def test_cancellation_stops_all_legs():
    db, collection = make_db()
    entered = 0
    started = asyncio.Event()
    finished = []

    async def wait(*args, **kwargs):
        nonlocal entered
        entered += 1
        if entered == 2:
            started.set()
        try:
            await asyncio.Event().wait()
        finally:
            finished.append(True)

    db._embed.side_effect = wait
    collection.query.bm25.side_effect = wait
    task = asyncio.create_task(db._retrieve_window("space", "query", CONDITION, 3))
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(finished) == 2


@pytest.mark.asyncio
async def test_keyword_and_embedding_run_concurrently():
    db, collection = make_db()
    ready = asyncio.Event()

    async def embedding(query):
        await ready.wait()
        return [[1.0]]

    async def bm25(**kwargs):
        ready.set()
        return QueryResponse([])

    db._embed.side_effect = embedding
    collection.query.bm25.side_effect = bm25
    await asyncio.wait_for(db._retrieve_window("space", "query", CONDITION, 3), 1)


@pytest.mark.asyncio
async def test_empty_query_uses_fetch_without_embedding_or_keyword_queries():
    db, collection = make_db()
    collection.query.fetch_objects.return_value = QueryResponse([obj(1)])
    hits = await db._hybrid_query("space", "", CONDITION, 3)
    assert hits == [({"chunk_title": "title-1"}, 0.0)]
    db._embed.assert_not_called()
    collection.query.near_vector.assert_not_called()
    collection.query.bm25.assert_not_called()
