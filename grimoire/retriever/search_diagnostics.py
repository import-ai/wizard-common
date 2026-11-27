from typing import Any

from wizard_common.grimoire.config import SearchDiagnosticsConfig
from wizard_common.grimoire.retriever.search_retrieval import RetrievalWindow
from wizard_common.grimoire.retriever.trace_events import event_spans


def distance_metric(config: Any) -> str:
    vector_config = getattr(config, "vector_config", None)
    if isinstance(vector_config, dict):
        if len(vector_config) != 1:
            return "unknown"
        config = next(iter(vector_config.values()))
    index = getattr(config, "vector_index_config", None)
    metric = getattr(index, "distance_metric", None)
    return str(getattr(metric, "value", metric)) if metric is not None else "unknown"


def record_candidates(
    *,
    span: Any,
    config: SearchDiagnosticsConfig,
    window: RetrievalWindow,
    metric: str,
    offset: int,
) -> None:
    candidates = (
        window.candidates[: config.max_candidates]
        if config.max_candidates
        else window.candidates
    )
    span.set_attributes(
        {
            "search.diagnostics.candidate_count": len(window.candidates),
            "search.diagnostics.recorded_count": len(candidates),
            "search.diagnostics.truncated_count": len(window.candidates)
            - len(candidates),
            "search.diagnostics.distance_metric": metric,
            "search.diagnostics.extra_query_count": 0,
            "search.diagnostics.status": (
                "partial" if "error" in window.branch_status.values() else "ok"
            ),
        }
    )
    for rank, (candidate, event_span) in enumerate(
        zip(candidates, event_spans(span, len(candidates), "search.candidates")),
        start=offset + 1,
    ):
        attributes: dict[str, Any] = {
            "candidate.uuid": candidate.uuid,
            "candidate.rank": rank,
            "candidate.final_score": candidate.final_score,
            "candidate.word_vector_rrf_score": candidate.word_vector_score,
            "candidate.distance_metric": metric,
            "candidate.passed_legs": tuple(candidate.passed_legs),
        }
        for name, field in (
            ("vector", "vector_distance"),
            ("word", "word_bm25_score"),
            ("gse", "gse_bm25_score"),
        ):
            value = getattr(candidate, field)
            branch = window.branch_status.get(name, "disabled")
            status = (
                branch
                if branch != "ok"
                else "not_in_recall"
                if name not in candidate.returned_legs
                else "missing"
                if value is None
                else "ok"
                if name in candidate.passed_legs
                else "below_threshold"
            )
            status_name = "vector" if name == "vector" else name + "_bm25"
            attributes[f"candidate.{status_name}_status"] = status
            if value is not None:
                attributes[f"candidate.{field}"] = value
        for source, target in (
            ("chunk_id", "chunk.chunk_id"),
            ("chunk_resource_id", "chunk.resource_id"),
            ("message_id", "message.id"),
        ):
            if candidate.properties.get(source):
                attributes[target] = candidate.properties[source]
        event_span.add_event("search.candidate", attributes)
