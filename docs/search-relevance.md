# Optional search relevance checks

Wizard and Wizard Pro share this implementation. Bing and resource checks are disabled by default. Search checks are enabled by default with a `0.5` threshold, but a deployment without a relevance endpoint and model starts normally and performs no relevance calls, so existing deployments need no new model service or credentials. The backend's search and picker consume this endpoint in 20-row pages, so each record they judge within the shared budget has a decision; records whose judgment could not complete are returned as pending rather than as kept results. The open API's chunk search returns filtered results on its own page size. The crawler service continues to return its original search response format.

## Configuration

Both applications read the same `OBW_TOOLS_RELEVANCE_*` environment variables. These settings do not use the Pro chat-model prefix.

| Variable suffix | Default | Meaning |
| --- | --- | --- |
| `BING_ENABLED` | `false` | Check a Bing result set before accepting it |
| `RESOURCE_ENABLED` | `false` | Filter visible resource snippets before citations |
| `SEARCH_ENABLED` | `true` | Single switch for relevance filtering at the wizard search endpoint |
| `BASE_URL` | unset | HTTP service base URL, excluding `/systemone` |
| `API_KEY` | unset | Optional Bearer credential; not required by unauthenticated local services |
| `MODEL` | unset | Decision model supported by the configured service |
| `TIMEOUT_SEC` | `2` | HTTP timeout in seconds |
| `BING_THRESHOLD` | `0.1` | Reject when the off-target score is greater than or equal to this value |
| `RESOURCE_THRESHOLD` | unset | Separately calibrated threshold, required when resource checks are enabled |
| `SEARCH_THRESHOLD` | `0.5` | Reject when the off-target score is greater than or equal to this value |
| `SEARCH_BATCH_SIZE` | `16` | Results judged per System One request, between 1 and 16. Larger values are rejected because the decision API answers more than 16 questions per request with HTTP 400 |
| `SEARCH_TIMEOUT_SEC` | `3` | Total relevance budget for one search response |

Bing and resource checks can remain disabled with endpoint and model fields unset. Explicitly enabling either requires an HTTP(S) base URL and a model, and resource checks also require an explicit threshold. Search checks are enabled by default: without both base URL and model, search relevance is a no-op that returns every record as kept and records an `unconfigured` status. Supplying only one of base URL and model fails validation, as does enabling a check with invalid values, and `SEARCH_BATCH_SIZE` outside 1–16 is rejected.

For a compatible decision service:

```dotenv
OBW_TOOLS_RELEVANCE_BASE_URL="http://127.0.0.1:9000/compatible-mode/v1"
OBW_TOOLS_RELEVANCE_MODEL="decision-model-preview"
OBW_TOOLS_RELEVANCE_API_KEY=""
OBW_TOOLS_RELEVANCE_BING_ENABLED=true
OBW_TOOLS_RELEVANCE_BING_THRESHOLD=0.1
OBW_TOOLS_RELEVANCE_RESOURCE_ENABLED=false
```

Configure a service and credentials suitable for the deployment. No provider endpoint, credential or model is hard-coded. Use environment/Secret injection for credentials; never commit actual values. Defaults do not enable or provision a local model.

## HTTP protocol

The first implementation supports `POST <BASE_URL>/systemone` with `state`, `questions` and `model`. It uses the existing HTTP client dependency; no provider SDK is needed. A generic `/chat/completions` endpoint is not compatible with this payload and is not implemented.

The response must contain exactly the requested answer keys, each with a numeric binary distribution:

```json
{
  "answers": {
    "relationship": {
      "probabilities": {"off_target": 0.1, "relevant": 0.9}
    }
  }
}
```

Scores must be finite numbers in `[0, 1]` and sum to approximately one. Decisions use `probabilities.off_target`, not `choice` or `confidence`. For the strict Bing and resource checks, missing/partial answers, invalid distributions and malformed responses count as detector failures. Search evaluates each answer independently: an invalid or missing answer marks only that result, and is retried once before it becomes pending. Connections are reused and closed at application shutdown. Cancellation from a winning search provider propagates to the pending Bing detection request. Separately, the internal chunk-search filter cancels its in-flight batches when its shared budget expires or its caller is cancelled.

## Bing behavior

The classifier receives the query and the first three usable results as ordered `{"title": "..."}` objects. URLs, snippets, ranks, annotations and reference results are excluded. The English binary rule matches the title-only benchmark.

A score at or above `BING_THRESHOLD` raises `bing_off_target`. Timeout, HTTP errors (including rate limits) and invalid responses raise `relevance_check_failed`. These errors bypass Bing retries and allow the existing staggered search to start its next provider immediately. Bing cannot win the search race until its check passes. Other providers retain their current behavior.

The 0.1 threshold was measured on the saved Bing title-only corpus; it is not a universal threshold for arbitrary models or resource chunks. Benchmark data remains separate from runtime configuration.

## Resource behavior

Resource checks run only after backend visibility filtering. Only authorized candidate snippets, up to the requested result limit, are sent. Each includes a local numeric ID, title and snippet text; actual resource IDs and other metadata are not sent. All candidate questions are batched into one HTTP request.

A separately defined rule judges each snippet. Scores below `RESOURCE_THRESHOLD` preserve the original retrieval objects, order and citation identity. Scores at or above the threshold remove the snippet. An all-rejected set returns an empty result. A detector failure preserves the already-authorized retrievals and records an unavailable status; visibility failures still return no results and never reach the detector.

Resource checks remain off until their model/rule/threshold are evaluated on resource-specific examples. Sending private snippets to a configured remote endpoint is an explicit deployment choice; a local compatible endpoint is also supported.

## Search behavior

Search checks run inside the wizard's internal search endpoint whenever `SEARCH_ENABLED` is on, which is the default; there is no separate backend flag or request field. A wizard without a relevance base URL or model skips the check, returns every record as kept and records an `unconfigured` status. Only chunk records are judged: distinct records that share a `resource_id` are deduplicated, keeping the highest-ranked chunk, and every other record is returned untouched as kept. The same `IndexRecord` object passed twice is not deduplicated.

Each request carries the query plus up to `SEARCH_BATCH_SIZE` results with their title and complete chunk text, in rank order. Up to two batches run concurrently, and results are merged in input order rather than in batch completion order. Title protection is applied before scores: the query and title are case-folded and compared after removing whitespace while keeping punctuation, and the shorter side must have at least two characters. A result is protected when the query appears in the title, or when the title appears in the query. Protected results are always kept, including when their batch fails or the budget expires. Other results are kept when the off-target score is below `SEARCH_THRESHOLD` and dropped when it reaches the threshold.

One `SEARCH_TIMEOUT_SEC` budget is shared by all batches of a search response; each batch uses the smaller of the remaining budget and `TIMEOUT_SEC` as its HTTP timeout. When the budget expires, batches still in flight are cancelled. A failed or timed-out batch is retried once within the remaining budget, and results whose answer is missing or invalid are also retried once in a follow-up batch within the same budget. The relevance filter returns kept and pending records separately: off-target records are dropped, pending records are not returned as kept, and kept contains only title-protected or below-threshold results. The HTTP search response itself exposes only `records`, `total` and `has_more`; pending records stay in the paginator cache for a later refill. Every pending record carries a reason: `deadline` when the shared budget expired, `error` when the detector failed after its retry, or `answer_missing` when no usable answer arrived after the follow-up. A follow-up cut off by the shared deadline is reported as `deadline` and counts in `skipped_batches`; a follow-up that errors keeps `answer_missing` for its items and counts in `failed_batches`. Trace attributes record input, chunk, kept, pending and off-target counts, the threshold, batch counts (`succeeded_batches`, `failed_batches`, `skipped_batches`), retry counts, unexpected batch errors and the concurrency limit, plus a status of `ok` (no pending), `partial` (some pending) or `unavailable` (pending and no successful batch). This differs from the strict Bing/resource answers: those still fail the whole response.

The backend's search and picker request filtered pages from this endpoint, currently 20 records per page, so each returned kept record has been judged and pending records stay cached for a later refill. The page size and request budget will change in a later stage of this rollout. The open API's chunk search asks for its own page size, up to 100 records, and also receives filtered results. Message search sends message records through the same endpoint, but the check only judges chunk records, so messages are returned untouched and make no relevance call. `SEARCH_ENABLED` in the wizard is the single switch for all consumers. Sending titles and full chunk text, including text the requesting user cannot access, to a remote endpoint is an explicit deployment and privacy decision; the backend still applies permissions to the final response.

## Observability and rollout

Bing requests propagate the active W3C trace context to the crawler. Its `Bing.*` and `BingBrowser.*` spans separate queueing, browser reuse/launch, context/page creation, navigation, result readiness, HTML retrieval, parsing and cleanup. Navigation attributes include available document DNS, connection, first-byte and download timings without recording URLs, HTML or cookies.

`RelevanceChecker` spans record question count, HTTP status, latency, Bing off-target score/threshold/rejection, and resource input/kept counts. Failed evaluations also record `error.type` and a sanitized `error.message` on the evaluate span and preserve the underlying exception as the `RelevanceError` cause. Remote error bodies, authorization headers and snippet contents are still excluded; JSON parse errors are reported as `invalid JSON response` instead of the body. Do not enable raw configuration debug logging in deployments that carry credentials.

`RelevanceChecker.search` additionally emits one final `relevance.chunk` event per judged chunk, at most 127 per span; retries do not emit intermediate events. Each event carries `chunk.index` (position in the deduplicated chunk list), `chunk.batch_index` (0-based batch ordinal), `chunk.chunk_id`, `chunk.resource_id`, `chunk.title` when set, `chunk.chunk_type`, `chunk.line_range` when set, `chunk.text_length` when text is not None, and `chunk.decision` (`kept`, `off_target`, `title_protected` or `pending`). Decisions with a score also record `chunk.off_target_score` and `chunk.relevant_score`; pending events record `chunk.pending_reason` (`deadline`, `error` or `answer_missing`) and no score. When more than 127 events would be emitted, a single `relevance.chunks_truncated` event reports `relevance.omitted_count`, keeping the span within the OpenTelemetry default of 128 events. Unexpected non-`RelevanceError` batch exceptions produce a warning log with the exception type and message, without payload contents, and increment `relevance.unexpected_batch_errors`; their items stay pending as `error`. During the answer-missing follow-up, detector errors keep the reason `answer_missing` and count in `failed_batches`, while unexpected exceptions are logged and counted as above; a follow-up cut off by the shared deadline switches the reason to `deadline`. Chunk text and the query are never recorded. Titles and resource IDs are user metadata and now appear in traces; treat trace access accordingly.

Deploy the shared change with both Wizard applications pinned to the same full commit. Bing and resource checks stay disabled until configured: enable Bing in the intended deployment after configuring the decision endpoint, and evaluate resource behavior separately before enabling it. Search checks are on by default and activate automatically once a deployment supplies the relevance base URL and model; without them, wizard search returns unfiltered results. Set `SEARCH_ENABLED=false` in the wizard to restore the previous unfiltered behavior for every consumer; there is no backend switch to change. Disabling this switch restores that tool's previous behavior without changing the crawler.
