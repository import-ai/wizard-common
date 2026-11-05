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
    collection.query.hybrid.return_value = SearchResponse([])
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
    wire = _FilterToREST.convert(collection.query.hybrid.call_args.kwargs["filters"])
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
