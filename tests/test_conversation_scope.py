from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest
from weaviate.collections.filters import _FilterToREST

from wizard_common.grimoire.entity.tools import Condition
from wizard_common.grimoire.retriever.weaviate_vector_db import WeaviateVectorDB


@dataclass
class SearchResponse:
    objects: list


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "included,excluded",
    [(None, None), ([], []), (["current"], None), (["a", "b"], ["b"]), (["a"], ["a"])],
)
async def test_scope_is_sent_to_vector_query_before_limit(included, excluded):
    db = object.__new__(WeaviateVectorDB)
    collection = AsyncMock()
    collection.query.hybrid.return_value = SearchResponse([])
    db._get_shard = AsyncMock(return_value=collection)
    db._embed = AsyncMock(return_value=[[1.0]])
    condition = Condition(
        namespace_id="space",
        user_id="user",
        conversation_ids=included,
        exclude_conversation_ids=excluded,
    )
    assert await db._hybrid_query("space", "query", condition, limit=1) == []
    arguments = collection.query.hybrid.call_args.kwargs
    assert arguments["limit"] == 1
    wire = _FilterToREST.convert(arguments["filters"])

    def leaves(node):
        return (
            [leaf for child in node["operands"] for leaf in leaves(child)]
            if "operands" in node
            else [node]
        )

    scope = [leaf for leaf in leaves(wire) if leaf.get("path") == ["conversation_id"]]
    assert scope == (
        [
            {
                "path": ["conversation_id"],
                "operator": "ContainsAny",
                "valueTextArray": included,
            }
        ]
        if included
        else []
    ) + [
        {"path": ["conversation_id"], "operator": "NotEqual", "valueText": value}
        for value in excluded or []
    ]
