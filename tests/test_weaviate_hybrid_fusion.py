import math
import uuid
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import weaviate.classes as wvc
from weaviate.collections.filters import _FilterToREST
from weaviate.exceptions import WeaviateQueryError

import wizard_common.grimoire.retriever.weaviate_vector_db as vector_db_module
from wizard_common.grimoire.entity.tools import Condition
from wizard_common.grimoire.retriever.weaviate_vector_db import (
    ALPHA,
    GSE_LEG_WEIGHT,
    GSE_QUERY_PROPERTIES,
    RRF_K,
    WORD_QUERY_PROPERTIES,
    WeaviateVectorDB,
)


@dataclass
class FakeMetadata:
    score: float | None = None
    certainty: float | None = None


@dataclass
class FakeObject:
    uuid: uuid.UUID
    properties: dict
    metadata: FakeMetadata | None = None


@dataclass
class QueryResponse:
    objects: list


CONDITION = Condition(namespace_id="space")


def make_db(hybrid_objects=(), bm25_objects=(), bm25_error=None):
    db = object.__new__(WeaviateVectorDB)
    collection = AsyncMock()
    collection.query.hybrid.return_value = QueryResponse(list(hybrid_objects))
    if bm25_error is not None:
        collection.query.bm25.side_effect = bm25_error
    else:
        collection.query.bm25.return_value = QueryResponse(list(bm25_objects))
    db._get_shard = AsyncMock(return_value=collection)
    db._embed = AsyncMock(return_value=[[1.0, 0.0]])
    return db, collection


def score_by_title(hits):
    return {props["chunk_title"]: score for props, score in hits}


@pytest.mark.asyncio
async def test_ascii_query_only_runs_word_hybrid_leg():
    obj = FakeObject(uuid.uuid4(), {"chunk_title": "NEBULA610X"}, FakeMetadata(0.02))
    db, collection = make_db(hybrid_objects=[obj])

    hits = await db._hybrid_query("space", "NEBULA610X", CONDITION, limit=20)

    collection.query.bm25.assert_not_called()
    arguments = collection.query.hybrid.call_args.kwargs
    assert arguments["query"] == "NEBULA610X"
    assert arguments["query_properties"] == WORD_QUERY_PROPERTIES
    assert all("_gse" not in prop for prop in arguments["query_properties"])
    assert arguments["alpha"] == ALPHA
    assert arguments["fusion_type"] == wvc.query.HybridFusion.RANKED
    assert arguments["limit"] == 20
    assert hits == [(obj.properties, 0.02)]


@pytest.mark.asyncio
async def test_cjk_query_runs_gse_bm25_leg():
    obj = FakeObject(uuid.uuid4(), {"chunk_title": "标题编码"}, FakeMetadata(0.03))
    db, collection = make_db(bm25_objects=[obj])

    hits = await db._hybrid_query("space", "标题编码", CONDITION, limit=5)

    arguments = collection.query.bm25.call_args.kwargs
    assert arguments["query"] == "标题编码"
    assert arguments["query_properties"] == GSE_QUERY_PROPERTIES
    assert arguments["limit"] == 5
    assert arguments["return_metadata"].score is True
    assert collection.query.hybrid.call_args.kwargs["query"] == "标题编码"
    assert hits == [(obj.properties, GSE_LEG_WEIGHT / RRF_K)]


@pytest.mark.asyncio
async def test_mixed_query_splits_channels():
    target = FakeObject(uuid.uuid4(), {"chunk_title": "610-11 NEBULA610X-标题编码.md"})
    db, collection = make_db(hybrid_objects=[target], bm25_objects=[target])

    await db._hybrid_query("space", "NEBULA610X 标题编码", CONDITION, limit=3)

    assert collection.query.hybrid.call_args.kwargs["query"] == "NEBULA610X 标题编码"
    assert collection.query.bm25.call_args.kwargs["query"] == "标题编码"


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["!!!", "NEBULA610X", "2024"])
async def test_stripped_empty_query_skips_gse_leg(query):
    db, collection = make_db()

    await db._hybrid_query("space", query, CONDITION, limit=3)

    collection.query.bm25.assert_not_called()
    assert collection.query.hybrid.call_args.kwargs["query"] == query


@pytest.mark.asyncio
async def test_fusion_adds_scores_for_same_uuid():
    shared = uuid.uuid4()
    other = uuid.uuid4()
    db, collection = make_db(
        hybrid_objects=[
            FakeObject(shared, {"chunk_title": "shared"}, FakeMetadata(0.01)),
            FakeObject(other, {"chunk_title": "other"}, FakeMetadata(0.02)),
        ],
        bm25_objects=[
            FakeObject(shared, {"chunk_title": "shared"}, FakeMetadata(9.0)),
            FakeObject(other, {"chunk_title": "other"}, FakeMetadata(1.0)),
        ],
    )

    hits = await db._hybrid_query("space", "标题", CONDITION, limit=10)

    scores = score_by_title(hits)
    assert math.isclose(scores["shared"], 0.01 + GSE_LEG_WEIGHT / RRF_K)
    assert math.isclose(scores["other"], 0.02 + GSE_LEG_WEIGHT / (RRF_K + 1))
    assert [props["chunk_title"] for props, _ in hits] == ["other", "shared"]


@pytest.mark.asyncio
async def test_fusion_orders_by_score_then_uuid_string():
    first = uuid.UUID(int=1)
    second = uuid.UUID(int=2)
    db, collection = make_db(
        hybrid_objects=[
            FakeObject(second, {"chunk_title": "second"}, FakeMetadata(0.02)),
            FakeObject(first, {"chunk_title": "first"}, FakeMetadata(0.02)),
        ]
    )

    hits = await db._hybrid_query("space", "NEBULA610X", CONDITION, limit=10)

    assert [props["chunk_title"] for props, _ in hits] == ["first", "second"]


@pytest.mark.asyncio
async def test_digit_noise_cannot_outrank_exact_title():
    target = FakeObject(
        uuid.UUID(int=1),
        {"chunk_title": "610-11 NEBULA610X-标题编码.md", "chunk_title_gse": "标题编码"},
        FakeMetadata(GSE_LEG_WEIGHT / RRF_K),
    )
    noise = FakeObject(
        uuid.UUID(int=2),
        {"chunk_title": "杭州10个区", "chunk_title_gse": "杭州 10 5 8 6334"},
        None,
    )
    db, collection = make_db(
        hybrid_objects=[target, noise],
        bm25_objects=[
            # Raw BM25 magnitude is large but the GSE leg only contributes by rank.
            FakeObject(noise.uuid, noise.properties, FakeMetadata(50.0)),
            FakeObject(target.uuid, target.properties, FakeMetadata(0.1)),
        ],
    )

    hits = await db._hybrid_query("space", "NEBULA610X 标题编码", CONDITION, limit=3)

    assert hits[0][0]["chunk_title"] == "610-11 NEBULA610X-标题编码.md"
    assert hits[1][0]["chunk_title"] == "杭州10个区"


@pytest.mark.asyncio
async def test_gse_leg_failure_falls_back_to_hybrid_results(caplog):
    obj = FakeObject(uuid.uuid4(), {"chunk_title": "目标"}, FakeMetadata(0.04))
    db, collection = make_db(
        hybrid_objects=[obj],
        bm25_error=WeaviateQueryError("bm25 exploded", "grpc"),
    )
    fake_span = MagicMock()

    with patch.object(vector_db_module, "trace") as fake_trace:
        fake_trace.get_current_span.return_value = fake_span
        hits = await db._hybrid_query("space", "标题", CONDITION, limit=5)

    assert hits == [(obj.properties, 0.04)]
    attributes = {}
    for call in fake_span.set_attributes.call_args_list:
        attributes.update(call.args[0])
    for call in fake_span.set_attribute.call_args_list:
        attributes[call.args[0]] = call.args[1]
    assert "bm25 exploded" in attributes["weaviate.hybrid.gse_error"]
    assert attributes["weaviate.hybrid.gse_leg"] is True
    assert attributes["weaviate.hybrid.gse_object_count"] == 0
    assert attributes["weaviate.hybrid.fused_object_count"] == 1
    assert "bm25 exploded" in caplog.text


@pytest.mark.asyncio
async def test_tenant_not_found_from_both_legs_returns_empty():
    tenant_error = WeaviateQueryError("tenant not found", "grpc")
    db, collection = make_db(bm25_error=tenant_error)
    collection.query.hybrid.side_effect = tenant_error

    assert await db._hybrid_query("space", "标题编码", CONDITION, limit=5) == []


@pytest.mark.asyncio
async def test_tenant_not_found_from_hybrid_leg_returns_empty():
    tenant_error = WeaviateQueryError("tenant not found", "grpc")
    db, collection = make_db()
    collection.query.hybrid.side_effect = tenant_error

    assert await db._hybrid_query("space", "NEBULA610X", CONDITION, limit=5) == []


@pytest.mark.asyncio
async def test_message_path_uses_split_message_properties():
    db, collection = make_db()

    await db._hybrid_query("space", "标题编码", CONDITION, limit=3)

    assert (
        "message_content"
        in collection.query.hybrid.call_args.kwargs["query_properties"]
    )
    assert (
        "message_content_gse"
        in collection.query.bm25.call_args.kwargs["query_properties"]
    )
    assert (
        "message_content"
        not in collection.query.bm25.call_args.kwargs["query_properties"]
    )
    assert (
        "message_content_gse"
        not in collection.query.hybrid.call_args.kwargs["query_properties"]
    )


@pytest.mark.asyncio
async def test_offset_slices_fused_ranking_after_fusion():
    ids = [uuid.UUID(int=index) for index in range(1, 7)]
    db, collection = make_db(
        hybrid_objects=[
            FakeObject(ids[0], {"chunk_title": "h0"}, FakeMetadata(0.50)),
            FakeObject(ids[1], {"chunk_title": "h1"}, FakeMetadata(0.40)),
            FakeObject(ids[2], {"chunk_title": "hb"}, FakeMetadata(0.305)),
            FakeObject(ids[3], {"chunk_title": "ha"}, FakeMetadata(0.30)),
            FakeObject(ids[4], {"chunk_title": "h4"}, FakeMetadata(0.20)),
            FakeObject(ids[5], {"chunk_title": "h5"}, FakeMetadata(0.10)),
        ],
        bm25_objects=[
            FakeObject(ids[3], {"chunk_title": "ha"}, FakeMetadata(9.0)),
        ],
    )

    hits = await db._hybrid_query("space", "标题", CONDITION, limit=4, offset=2)

    assert collection.query.hybrid.call_args.kwargs["limit"] == 6
    assert collection.query.bm25.call_args.kwargs["limit"] == 6
    assert [props["chunk_title"] for props, _ in hits] == ["ha", "hb", "h4", "h5"]


@pytest.mark.asyncio
async def test_both_legs_receive_the_same_filters():
    db, collection = make_db()
    condition = Condition(
        namespace_id="space", user_id="user", resource_ids=["r1", "r2"]
    )

    await db._hybrid_query("space", "标题编码", condition, limit=3)

    hybrid_filters = collection.query.hybrid.call_args.kwargs["filters"]
    bm25_filters = collection.query.bm25.call_args.kwargs["filters"]
    assert hybrid_filters is bm25_filters
    assert _FilterToREST.convert(hybrid_filters) == _FilterToREST.convert(bm25_filters)


@pytest.mark.asyncio
async def test_non_tenant_hybrid_failure_reraises():
    noise = FakeObject(uuid.uuid4(), {"chunk_title": "噪声"}, FakeMetadata(1.0))
    db, collection = make_db(bm25_objects=[noise])
    collection.query.hybrid.side_effect = WeaviateQueryError("hybrid exploded", "grpc")

    with pytest.raises(WeaviateQueryError, match="hybrid exploded"):
        await db._hybrid_query("space", "标题", CONDITION, limit=5)


@pytest.mark.asyncio
async def test_embedding_failure_reraises_even_when_gse_succeeds():
    noise = FakeObject(uuid.uuid4(), {"chunk_title": "噪声"}, FakeMetadata(1.0))
    db, collection = make_db(bm25_objects=[noise])
    db._embed = AsyncMock(side_effect=RuntimeError("embedding exploded"))

    with pytest.raises(RuntimeError, match="embedding exploded"):
        await db._hybrid_query("space", "标题", CONDITION, limit=5)

    collection.query.bm25.assert_awaited()
    collection.query.hybrid.assert_not_awaited()


@pytest.mark.asyncio
async def test_embedding_tenant_not_found_returns_empty():
    db, collection = make_db()
    db._embed = AsyncMock(side_effect=WeaviateQueryError("tenant not found", "grpc"))

    assert await db._hybrid_query("space", "标题", CONDITION, limit=5) == []
    collection.query.hybrid.assert_not_awaited()
