# Search candidate diagnostics

Diagnostics are enabled by default. They only format metadata from formal retrieval;
they add no queries, embeddings, or result fields. A recording span is required.

```dotenv
OBW_VECTOR_SEARCH_DIAGNOSTICS_ENABLED=true
OBW_VECTOR_SEARCH_DIAGNOSTICS_MAX_CANDIDATES=0
```

A zero candidate limit (the default) records every fused candidate in the retrieval window. A positive limit explicitly caps diagnostics without changing retrieval. The first 100 events stay on the retrieval span; overflow is stored in `search.candidates` child spans, in batches of 100, to avoid the default SDK event limit. Inspect the full trace, not only the parent span, when collecting scores.

The previous diagnostic timeout setting is obsolete: there are no diagnostic RPCs
and no separate five-second diagnostic phase. Existing request deadlines still apply.
Disabling diagnostics does not disable retrieval thresholds.

## Measurements and events

The existing `WeaviateVectorDB._hybrid_query` span name is retained for continuity,
but retrieval now uses independent vector, WORD and optional GSE branches. See
[Search thresholds](search-thresholds.md).

Each `search.candidate` event records UUID, post-fusion rank, chunk/resource or
message join IDs, `candidate.passed_legs`, raw vector distance, raw WORD/GSE BM25
scores, `candidate.final_score` and `candidate.word_vector_rrf_score`.
RRF scores are not native hybrid scores, similarities or probabilities.
Raw metrics are available only when that branch returned the candidate. A missing
metric does not mean zero or prove there was no keyword match outside top-K.
A value below a branch threshold is still reported if another branch retained it.

Per-metric statuses distinguish `ok`, `below_threshold`, `missing`, `not_in_recall`,
`disabled` and `error`. Unavailable numeric attributes are omitted.
Distance metric names come from the already-loaded collection schema; unknown
metrics are labeled unknown, not assumed to be cosine. No extra schema read occurs.

Branch attributes report returned/accepted/rejected counts, status and lookahead.
Vector distance filtering happens on the server, so its rejected count does not
include objects discarded inside Weaviate. Overall status is partial on branch failure.
Diagnostics report candidate/recorded/truncated counts and `extra_query_count=0`.
Events are split into batches of at most 100 per span; this does not truncate
retrieval. Custom SDK limits below 100 or exporter failures may still drop events.
Events contain no titles, snippets, URLs, vectors, error payloads or credentials.
