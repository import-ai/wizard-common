import asyncio
import math
import time
from typing import Any

import weaviate.classes as wvc

from wizard_common.grimoire.config import SearchDiagnosticsConfig


def distance_metric(config: Any) -> str:
    vector_config = getattr(config, "vector_config", None)
    if isinstance(vector_config, dict):
        if len(vector_config) != 1:
            return "unknown"
        config = next(iter(vector_config.values()))
    index = getattr(config, "vector_index_config", None)
    metric = getattr(index, "distance_metric", None)
    return str(getattr(metric, "value", metric)) if metric is not None else "unknown"


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


async def record_candidates(
    *,
    collection: Any,
    span: Any,
    config: SearchDiagnosticsConfig,
    query: str,
    vector: list[float],
    filters: Any,
    word_properties: list[str],
    selected: list[tuple[str, dict]],
    hybrid_objects: list,
    gse_objects: list,
    gse_enabled: bool,
    gse_failed: bool,
    metric: str,
    offset: int,
    deadline: float | None = None,
) -> None:
    started = time.monotonic()
    candidates = selected[: config.max_candidates]
    span.set_attributes(
        {
            "search.diagnostics.candidate_count": len(selected),
            "search.diagnostics.recorded_count": len(candidates),
            "search.diagnostics.truncated_count": len(selected) - len(candidates),
            "search.diagnostics.distance_metric": metric,
        }
    )
    if not candidates:
        span.set_attribute("search.diagnostics.status", "empty")
        return
    budget = config.timeout_sec
    if deadline is not None:
        budget = min(budget, deadline - started)
    candidate_filter = wvc.query.Filter.by_id().contains_any(
        [uuid for uuid, _ in candidates]
    )
    if filters is not None:
        candidate_filter = filters & candidate_filter

    async def probe(kind: str) -> tuple[str, dict[str, float], set[str]]:
        if budget <= 0:
            return "deadline", {}, set()
        try:
            async with asyncio.timeout(budget):
                if kind == "vector":
                    result = await collection.query.near_vector(
                        near_vector=vector,
                        filters=candidate_filter,
                        limit=len(candidates),
                        return_properties=False,
                        return_metadata=wvc.query.MetadataQuery(distance=True),
                    )
                    field = "distance"
                else:
                    result = await collection.query.bm25(
                        query=query,
                        query_properties=word_properties,
                        filters=candidate_filter,
                        limit=len(candidates),
                        return_properties=False,
                        return_metadata=wvc.query.MetadataQuery(score=True),
                    )
                    field = "score"
                values = {
                    str(obj.uuid): value
                    for obj in result.objects
                    if (value := _number(getattr(obj.metadata, field, None)))
                    is not None
                }
                return "ok", values, {str(obj.uuid) for obj in result.objects}
        except TimeoutError:
            return "timeout", {}, set()
        except Exception:
            # Exception payloads can contain endpoints or request content.
            return "error", {}, set()

    (
        (vector_status, distances, _),
        (word_status, word_scores, word_returned),
    ) = await asyncio.gather(probe("vector"), probe("word"))
    hybrid = {str(obj.uuid): obj for obj in hybrid_objects}
    gse = {str(obj.uuid): obj for obj in gse_objects}
    for rank, (uuid, entry) in enumerate(candidates, start=offset + 1):
        properties = entry["properties"]
        attributes: dict[str, Any] = {
            "candidate.uuid": uuid,
            "candidate.rank": rank,
            "candidate.final_score": entry["score"],
            "candidate.hybrid_returned": uuid in hybrid,
            "candidate.gse_returned": uuid in gse,
            "candidate.distance_metric": metric,
            "candidate.vector_status": (
                "ok"
                if uuid in distances
                else "missing"
                if vector_status == "ok"
                else vector_status
            ),
            "candidate.word_bm25_status": (
                "ok"
                if uuid in word_scores
                else "missing"
                if uuid in word_returned
                else "not_matched"
                if word_status == "ok"
                else word_status
            ),
            "candidate.gse_bm25_status": (
                "ok"
                if uuid in gse
                and _number(getattr(gse[uuid].metadata, "score", None)) is not None
                else "missing"
                if uuid in gse
                else "error"
                if gse_failed
                else "not_in_recall"
                if gse_enabled
                else "disabled"
            ),
        }
        for source, target in (
            ("chunk_id", "chunk.chunk_id"),
            ("chunk_resource_id", "chunk.resource_id"),
            ("message_id", "message.id"),
        ):
            if properties.get(source):
                attributes[target] = properties[source]
        if uuid in distances:
            attributes["candidate.vector_distance"] = distances[uuid]
        if uuid in word_scores:
            attributes["candidate.word_bm25_score"] = word_scores[uuid]
        if uuid in hybrid:
            score = _number(getattr(hybrid[uuid].metadata, "score", None))
            if score is not None:
                attributes["candidate.hybrid_score"] = score
        if uuid in gse:
            score = _number(getattr(gse[uuid].metadata, "score", None))
            if score is not None:
                attributes["candidate.gse_bm25_score"] = score
        span.add_event("search.candidate", attributes)
    span.set_attributes(
        {
            "search.diagnostics.status": (
                "ok"
                if (
                    vector_status == word_status == "ok"
                    and len(distances) == len(candidates)
                    and word_returned <= word_scores.keys()
                    and not gse_failed
                )
                else "partial"
            ),
            "search.diagnostics.vector_status": vector_status,
            "search.diagnostics.word_bm25_status": word_status,
            "search.diagnostics.duration_ms": (time.monotonic() - started) * 1000,
        }
    )
