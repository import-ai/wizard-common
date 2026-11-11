# Search retrieval thresholds

Nonempty searches execute two independent branches concurrently: vector retrieval
(including one query embedding) and WORD BM25. Chinese or mixed queries also run
GSE BM25 over the non-ASCII-alphanumeric portion. Every branch retains identical
tenant, ACL, record type and resource filters.

## Initial defaults

- `vector.search_thresholds.max_distance = 0.50`: inclusive native vector distance
  ceiling, also checked in the application. Calibrated for the current cosine index;
  do not reuse this value for another metric or embedding without evaluation.
- `vector.search_thresholds.word_min_bm25_score = 1.0`: inclusive original WORD
  BM25 floor, applied in the application. This is corpus/query dependent, not a probability.
- `vector.search_thresholds.gse_min_bm25_score = None`: no GSE floor; actual
  keyword matches remain eligible. GSE tokenization can still admit weak matches.

Each field accepts `None` in typed configuration to disable its gate independently.
Environment variable examples are in both Wizard example.env files; omitting them
uses these defaults. Do not use an empty string to represent None.

A candidate survives if ANY successful branch passes its own gate, never only if
both vector and WORD pass. Only passing branches contribute to RRF (k=60, weight
0.5 per branch). Equal fused scores are ordered by UUID. Raw branch scores remain
available for diagnostics even when another branch is what retained the candidate.
Empty-query listing keeps its existing fetch behavior without embeddings.

## Pagination and failures

Each branch requests offset + limit + 1 objects from its beginning. Vector distance
is filtered inside Weaviate; keyword floors are applied before fusion. The returned
internal search page carries explicit has_more based on lookahead and remaining
eligible branch tails. Wizard advances by the actual raw page length, not a fixed
window size, and never treats a short filtered page alone as exhaustion. Its existing
scan/time budgets and resource deduplication, ACL and relevance filter remain.
Cache keys include threshold values and the split-retrieval version.

This remains bounded top-K fusion, not a snapshot cursor: increasing recall depth
or concurrent corpus updates may affect ordering across windows. Public pagination
contracts are unchanged. Result counts are not padded with weak matches.

One failed branch permits remaining branches to serve results and records partial
status. Failed branches cannot contribute scores. If all branches fail, the error
propagates; a missing tenant remains an empty result. Cancellation propagates.

## Calibration

The initial cosine ceiling retains the known target (0.4442) while excluding the
closest observed vector-only noise (0.5033) in the NEBULA610X sample. Its WORD score
was 6.6754. This is a starting configuration, not multilingual quality validation.
Evaluate Chinese, English, mixed text, prefixes, short terms and no-match queries
before tightening defaults. Inspect retained/rejected counts, target recall,
relevance-filter acceptance and end-to-end latency; do not tune against RRF scores.
