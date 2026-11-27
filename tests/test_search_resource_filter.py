from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest
from weaviate.collections.filters import _FilterToREST

from wizard_common.grimoire.entity.index_record import IndexRecordType
from wizard_common.grimoire.retriever.weaviate_vector_db import WeaviateVectorDB


@dataclass
class SearchResponse:
    objects: list


def leaves(node):
    return (
        [leaf for child in node["operands"] for leaf in leaves(child)]
        if "operands" in node
        else [node]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("resource_ids", [None, [], ["r1", "r2"]])
async def test_search_filters_by_resource_ids(resource_ids):
    db = object.__new__(WeaviateVectorDB)
    collection = AsyncMock()
    collection.query.near_vector.return_value = SearchResponse([])
    collection.query.bm25.return_value = SearchResponse([])
    db._get_shard = AsyncMock(return_value=collection)
    db._embed = AsyncMock(return_value=[[1.0]])

    records = await db.search(
        query="query",
        namespace_id="space",
        user_id="user",
        record_type=IndexRecordType.chunk,
        offset=0,
        limit=20,
        resource_ids=resource_ids,
    )

    assert records == []
    calls = [
        collection.query.near_vector.call_args,
        *collection.query.bm25.call_args_list,
    ]
    assert len(calls) == 2
    wires = [_FilterToREST.convert(call.kwargs["filters"]) for call in calls]
    assert all(wire == wires[0] for wire in wires)
    wire = wires[0]
    resource_filters = [
        leaf for leaf in leaves(wire) if leaf.get("path") == ["chunk_resource_id"]
    ]
    assert resource_filters == (
        [
            {
                "path": ["chunk_resource_id"],
                "operator": "ContainsAny",
                "valueTextArray": resource_ids,
            }
        ]
        if resource_ids
        else []
    )
