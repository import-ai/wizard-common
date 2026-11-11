# Search candidate diagnostics

Search diagnostics are enabled by default and provide tracing only. They do not change candidate selection,
scores, ordering, or result DTOs. They require a recording search span.

## Configuration

Both Wizard applications load these settings through the shared VectorConfig:

```dotenv
OBW_VECTOR_SEARCH_DIAGNOSTICS_ENABLED=true
OBW_VECTOR_SEARCH_DIAGNOSTICS_TIMEOUT_SEC=0.2
OBW_VECTOR_SEARCH_DIAGNOSTICS_MAX_CANDIDATES=100
```

Timeout must be greater than zero and at most one second. The candidate cap is
1–100. Set OBW_VECTOR_SEARCH_DIAGNOSTICS_ENABLED=false to disable diagnostic probes.
No cluster configuration is changed automatically.

With diagnostics disabled, or without a recording span, no diagnostic queries or
embedding calls are added. Empty-query fetches are not probed.

## Measurements

For the current post-fusion offset/limit slice, the helper probes at most 100 UUIDs.
It reuses the existing query embedding and issues two concurrent, candidate-scoped
queries: near_vector with distance metadata and BM25 over the original WORD fields.
Both retain the original filters and namespace tenant. GSE scores are reused from
the existing GSE response, with no additional GSE query.

The probe timeout is also bounded by the remaining caller deadline when supplied
by the Wizard search paginator. Exhausted budgets start no queries; cancellations
propagate and cancel both probes. Probe failures never replace search results.
Enabling diagnostics can consume search budget and reduce how much later work
fits in that budget; do not compare its end-to-end latency with normal searches
without separating search.diagnostics.duration_ms.

The original BM25 score is measured in a candidate-filtered query using the same
query text and WORD properties, not extracted from hybrid explain_score. It is
not a global BM25 rank. A successful diagnostic match does not prove that the
object participated in the original hybrid keyword top-K. Corpus updates between
retrieval and probes can produce missing metadata.

## Trace events

The existing WeaviateVectorDB._hybrid_query span receives search.candidate events:

- candidate.uuid and candidate.rank: object UUID and one-based post-fusion rank
- chunk.chunk_id / chunk.resource_id: join keys for relevance.chunk events
- message.id: join key for message records, when present
- candidate.vector_distance and candidate.distance_metric
- candidate.word_bm25_score and candidate.gse_bm25_score: raw keyword scores
- candidate.hybrid_score: Weaviate hybrid score, before the application GSE fusion
- candidate.final_score: application-fused score
- candidate.hybrid_returned / candidate.gse_returned: presence in original responses
- candidate.vector_status / candidate.word_bm25_status / candidate.gse_bm25_status

Distance metrics come from the already-loaded collection schema. Unknown metrics
are labeled unknown, never assumed to be cosine. No extra schema query is added.
Never interpret hybrid/final scores as probabilities or raw vector similarities.

Statuses distinguish ok, missing metadata, not_matched (successful WORD probe only),
not_in_recall (GSE top-K absence), disabled, error, timeout, and deadline.
Unavailable numeric attributes are omitted, not replaced by zero or null.
GSE absence from top-K does not prove that the text has no matching Chinese terms.

Span attributes include diagnostic status, candidate/recorded/truncated counts,
per-probe status, metric, and duration_ms. Only the first max_candidates selected
objects are probed/emitted; the search result slice itself is never truncated.
The 100-event ceiling leaves room within OpenTelemetry's default 128-event limit.
Custom lower exporter/SDK limits may still drop events and should be monitored.

The new events contain IDs and numeric/status metadata, not titles, snippets,
vectors, URLs, exception payloads, or credentials. Existing trace attributes are
unchanged. Internal search metadata is not added to the public response schema.
