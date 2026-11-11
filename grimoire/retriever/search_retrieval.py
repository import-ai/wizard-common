import asyncio
import math
from collections.abc import Awaitable, Callable
from typing import Any

import weaviate.classes as wvc
from opentelemetry import trace
from pydantic import BaseModel, Field

from wizard_common.grimoire.config import SearchThresholdsConfig

ALPHA = 0.5
RRF_K = 60
GSE_LEG_WEIGHT = 1 - ALPHA
tracer = trace.get_tracer(__name__)


class RetrievedCandidate(BaseModel):
    uuid: str
    properties: dict[str, Any]
    final_score: float = 0
    word_vector_score: float = 0
    vector_distance: float | None = None
    word_bm25_score: float | None = None
    gse_bm25_score: float | None = None
    passed_legs: list[str] = Field(default_factory=list)
    returned_legs: list[str] = Field(default_factory=list)


class RetrievalWindow(BaseModel):
    candidates: list[RetrievedCandidate] = Field(default_factory=list)
    has_more: bool = False
    branch_status: dict[str, str] = Field(default_factory=dict)


def _value(obj: Any, field: str) -> float | None:
    value = getattr(obj.metadata, field, None)
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    ):
        return float(value)
    return None


async def retrieve_window(
    *,
    collection: Any,
    embed: Callable[[str], Awaitable[list[list[float]]]],
    query: str,
    gse_query: str,
    filters: Any,
    thresholds: SearchThresholdsConfig,
    word_properties: list[str],
    gse_properties: list[str],
    offset: int,
    limit: int,
    span: Any,
) -> RetrievalWindow:
    depth = offset + limit + 1
    gse_enabled = any(character.isalnum() for character in gse_query)
    maximum = thresholds.max_distance
    span.set_attributes(
        {
            "search.retrieval.depth": depth,
            "search.retrieval.offset": offset,
            "search.retrieval.limit": limit,
            "search.retrieval.fusion": "ranked",
            "search.thresholds.max_distance": maximum
            if maximum is not None
            else "disabled",
            "search.thresholds.word_min_bm25_score": (
                thresholds.word_min_bm25_score
                if thresholds.word_min_bm25_score is not None
                else "disabled"
            ),
            "search.thresholds.gse_min_bm25_score": (
                thresholds.gse_min_bm25_score
                if thresholds.gse_min_bm25_score is not None
                else "disabled"
            ),
            "weaviate.hybrid.query": query,
            "weaviate.hybrid.gse_query": gse_query,
            "weaviate.hybrid.gse_leg": gse_enabled,
        }
    )

    async def vector_query():
        vector = (await embed(query))[0]
        return await collection.query.near_vector(
            near_vector=vector,
            distance=maximum,
            filters=filters,
            limit=depth,
            return_metadata=wvc.query.MetadataQuery(distance=True),
        )

    async def run(name: str, operation: Awaitable):
        with tracer.start_as_current_span(f"WeaviateVectorDB.{name}") as child:
            try:
                return await operation
            except Exception as error:
                child.set_attribute("search.branch.error_type", type(error).__name__)
                return error

    operations = [
        ("vector", vector_query()),
        (
            "word",
            collection.query.bm25(
                query=query,
                query_properties=word_properties,
                filters=filters,
                limit=depth,
                return_metadata=wvc.query.MetadataQuery(score=True),
            ),
        ),
    ]
    if gse_enabled:
        operations.append(
            (
                "gse",
                collection.query.bm25(
                    query=gse_query,
                    query_properties=gse_properties,
                    filters=filters,
                    limit=depth,
                    return_metadata=wvc.query.MetadataQuery(score=True),
                ),
            )
        )
    results = await asyncio.gather(
        *(run(name, operation) for name, operation in operations)
    )
    failures = [result for result in results if isinstance(result, Exception)]
    if len(failures) == len(results):
        if all("tenant not found" in str(error).lower() for error in failures):
            span.set_attribute("search.retrieval.status", "tenant_not_found")
            return RetrievalWindow()
        raise failures[0]

    fused: dict[str, RetrievedCandidate] = {}
    raw_metrics: dict[str, dict[str, float | None]] = {}
    branch_status = {"gse": "disabled"}
    branch_more = False
    for (name, _), result in zip(operations, results):
        prefix = f"search.retrieval.{name}"
        if isinstance(result, Exception):
            branch_status[name] = "error"
            span.set_attribute(f"{prefix}.status", "error")
            span.set_attribute(f"{prefix}.error_type", type(result).__name__)
            continue
        objects = result.objects
        field = "distance" if name == "vector" else "score"
        minimum = (
            thresholds.word_min_bm25_score
            if name == "word"
            else thresholds.gse_min_bm25_score
        )
        measured = [(obj, _value(obj, field)) for obj in objects]
        branch_status[name] = "ok"
        raw_metrics[name] = {str(obj.uuid): value for obj, value in measured}
        accepted = [
            (obj, value)
            for obj, value in measured
            if value is not None
            and (
                (maximum is None or value <= maximum)
                if name == "vector"
                else minimum is None or value >= minimum
            )
        ]
        accepted.sort(
            key=lambda item: (
                item[1] if name == "vector" else -item[1],
                str(item[0].uuid),
            )
        )
        # Keyword scores are descending: once the tail is below the minimum,
        # deeper objects cannot qualify through this branch.
        tail = measured[-1][1] if measured else None
        more = len(objects) >= depth and (
            (maximum is None or (tail is not None and tail <= maximum))
            if name == "vector"
            else minimum is None or (tail is not None and tail >= minimum)
        )
        branch_more |= more
        span.set_attributes(
            {
                f"{prefix}.status": "ok",
                f"{prefix}.returned_count": len(objects),
                f"{prefix}.accepted_count": len(accepted),
                f"{prefix}.rejected_count": len(objects) - len(accepted),
                f"{prefix}.has_more": more,
            }
        )
        if name == "vector":
            span.set_attribute(
                f"{prefix}.distance_filtered_on_server", maximum is not None
            )
        for rank, (obj, value) in enumerate(accepted):
            uuid = str(obj.uuid)
            candidate = fused.setdefault(
                uuid,
                RetrievedCandidate(
                    uuid=uuid,
                    properties=obj.properties or {},
                ),
            )
            candidate.passed_legs.append(name)
            if name == "vector":
                candidate.vector_distance = value
                weight = ALPHA
            elif name == "word":
                candidate.word_bm25_score = value
                weight = 1 - ALPHA
            else:
                candidate.gse_bm25_score = value
                weight = GSE_LEG_WEIGHT
            candidate.final_score += weight / (rank + RRF_K)
            if name != "gse":
                candidate.word_vector_score += weight / (rank + RRF_K)
    for uuid, candidate in fused.items():
        for name, values in raw_metrics.items():
            if uuid in values:
                candidate.returned_legs.append(name)
                field = {
                    "vector": "vector_distance",
                    "word": "word_bm25_score",
                    "gse": "gse_bm25_score",
                }[name]
                setattr(candidate, field, values[uuid])
    ordered = sorted(
        fused.values(), key=lambda candidate: (-candidate.final_score, candidate.uuid)
    )
    span.set_attributes(
        {
            "search.retrieval.status": "partial" if failures else "ok",
            "search.retrieval.fused_count": len(ordered),
            "weaviate.hybrid.fused_object_count": len(ordered),
        }
    )
    return RetrievalWindow(
        candidates=ordered[offset : offset + limit],
        has_more=branch_more or len(ordered) > offset + limit,
        branch_status=branch_status,
    )
