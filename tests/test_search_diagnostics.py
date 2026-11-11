import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError
from test_weaviate_hybrid_fusion import CONDITION, make_db, obj

from wizard_common.grimoire.config import (
    SearchDiagnosticsConfig,
    SearchThresholdsConfig,
)
from wizard_common.grimoire.retriever import weaviate_vector_db as vector_module
from wizard_common.grimoire.retriever.search_diagnostics import (
    distance_metric,
    record_candidates,
)
from wizard_common.grimoire.retriever.search_retrieval import (
    RetrievalWindow,
    RetrievedCandidate,
)


def attributes(span):
    result = {}
    for call in span.set_attributes.call_args_list:
        result.update(call.args[0])
    for call in span.set_attribute.call_args_list:
        result[call.args[0]] = call.args[1]
    return result


def candidate(index=1):
    return RetrievedCandidate(
        uuid=str(uuid.UUID(int=index)),
        properties={
            "chunk_resource_id": "r1",
            "chunk_id": "c1",
            "chunk_text": "private text",
        },
        final_score=0.03,
        word_vector_score=0.03,
        vector_distance=0.25,
        word_bm25_score=0.2,
        passed_legs=["vector"],
        returned_legs=["vector", "word"],
    )


def test_events_distinguish_raw_metric_from_qualifying_branch_without_payloads():
    span = MagicMock()
    window = RetrievalWindow(
        candidates=[candidate()],
        branch_status={"vector": "ok", "word": "ok", "gse": "disabled"},
    )
    record_candidates(
        span=span,
        config=SearchDiagnosticsConfig(),
        window=window,
        metric="cosine",
        offset=100,
    )
    e = span.add_event.call_args.args[1]
    assert e["candidate.rank"] == 101
    assert e["candidate.vector_distance"] == 0.25
    assert e["candidate.word_bm25_score"] == 0.2
    assert e["candidate.word_bm25_status"] == "below_threshold"
    assert e["candidate.gse_bm25_status"] == "disabled"
    assert "candidate.gse_bm25_score" not in e
    assert e["candidate.passed_legs"] == ("vector",)
    assert e["candidate.final_score"] == 0.03
    assert e["candidate.word_vector_rrf_score"] == 0.03
    assert "candidate.hybrid_score" not in e
    assert "private text" not in str(span.mock_calls)
    assert attributes(span)["search.diagnostics.extra_query_count"] == 0


def test_cap_reports_truncation_without_changing_result_window():
    span = MagicMock()
    window = RetrievalWindow(candidates=[candidate(i) for i in range(1, 102)])
    record_candidates(
        span=span,
        config=SearchDiagnosticsConfig(),
        window=window,
        metric="cosine",
        offset=0,
    )
    assert span.add_event.call_count == 100
    assert len(window.candidates) == 101
    assert attributes(span)["search.diagnostics.truncated_count"] == 1


def test_error_missing_and_not_in_recall_are_distinct():
    c = candidate()
    c.vector_distance = None
    c.word_bm25_score = None
    window = RetrievalWindow(
        candidates=[c], branch_status={"vector": "ok", "word": "error", "gse": "ok"}
    )
    span = MagicMock()
    record_candidates(
        span=span,
        config=SearchDiagnosticsConfig(),
        window=window,
        metric="cosine",
        offset=0,
    )
    e = span.add_event.call_args.args[1]
    assert e["candidate.vector_status"] == "missing"
    assert e["candidate.word_bm25_status"] == "error"
    assert e["candidate.gse_bm25_status"] == "not_in_recall"
    assert "candidate.vector_distance" not in e
    assert "candidate.word_bm25_score" not in e
    assert attributes(span)["search.diagnostics.status"] == "partial"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "enabled,recording", [(False, True), (True, False), (True, True)]
)
async def test_trace_toggle_never_changes_results_or_adds_queries(enabled, recording):
    db, collection = make_db(
        vector_objects=[obj(1, distance=0.4)], word_objects=[obj(1, score=2)]
    )
    db.config.search_diagnostics.enabled = enabled
    span = MagicMock()
    span.is_recording.return_value = recording
    with patch.object(vector_module.trace, "get_current_span", return_value=span):
        hits = await db._hybrid_query("space", "query", CONDITION, 20)
    assert len(hits) == 1
    db._embed.assert_awaited_once()
    collection.query.near_vector.assert_awaited_once()
    collection.query.bm25.assert_awaited_once()
    assert span.add_event.call_count == (1 if enabled and recording else 0)


@pytest.mark.asyncio
async def test_diagnostic_failure_cannot_replace_search_results():
    db, _ = make_db(vector_objects=[obj(1, distance=0.4)])
    with (
        patch.object(vector_module, "record_candidates", side_effect=RuntimeError),
        patch.object(vector_module.trace, "get_current_span", return_value=MagicMock()),
    ):
        hits = await db._hybrid_query("space", "query", CONDITION, 20)
    assert len(hits) == 1


def test_config_defaults_bounds_and_named_metric():
    assert SearchDiagnosticsConfig().enabled
    c = SearchThresholdsConfig()
    assert (
        c.max_distance == 0.5
        and c.word_min_bm25_score == 1
        and c.gse_min_bm25_score is None
    )
    for kwargs in [
        {"max_distance": 2.1},
        {"max_distance": float("nan")},
        {"word_min_bm25_score": -1},
        {"gse_min_bm25_score": float("inf")},
    ]:
        with pytest.raises(ValidationError):
            SearchThresholdsConfig(**kwargs)
    for cap in [0, 101]:
        with pytest.raises(ValidationError):
            SearchDiagnosticsConfig(max_candidates=cap)
    config = SimpleNamespace(
        vector_config={
            "default": SimpleNamespace(
                vector_index_config=SimpleNamespace(distance_metric="cosine")
            )
        }
    )
    assert distance_metric(config) == "cosine"
    assert distance_metric(SimpleNamespace()) == "unknown"
