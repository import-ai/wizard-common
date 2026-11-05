# Optional search relevance checks

Wizard and Wizard Pro share this implementation. Bing and resource checks are disabled by default. Search checks are enabled by default with a `0.6` threshold, but a deployment without a relevance endpoint and model starts normally and performs no relevance calls, so existing deployments need no new model service or credentials. The backend's search and picker consume this endpoint in 20-row pages, so each record they consider has been judged; the open API's chunk search returns filtered results on its own page size. The crawler service continues to return its original search response format.

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
| `SEARCH_THRESHOLD` | `0.6` | Reject when the off-target score is greater than or equal to this value |
| `SEARCH_BATCH_SIZE` | `10` | Results judged per request, between 1 and 100. The System One decision API accepts at most 16 questions per request, so keep this at 16 or below |
| `SEARCH_TIMEOUT_SEC` | `1` | Total relevance budget for one search response |

Bing and resource checks can remain disabled with endpoint and model fields unset. Explicitly enabling either requires an HTTP(S) base URL and a model, and resource checks also require an explicit threshold. Search checks are enabled by default: without both base URL and model, search relevance is a no-op that returns every result and records an `unconfigured` status. Supplying only one of base URL and model fails validation, as does enabling a check with invalid values.

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

Scores must be finite numbers in `[0, 1]` and sum to approximately one. Decisions use `probabilities.off_target`, not `choice` or `confidence`. Missing/partial answers, invalid distributions and malformed responses count as detector failures. Connections are reused and closed at application shutdown. Cancellation from a winning search provider propagates to the pending detection request.

## Bing behavior

The classifier receives the query and the first three usable results as ordered `{"title": "..."}` objects. URLs, snippets, ranks, annotations and reference results are excluded. The English binary rule matches the title-only benchmark.

A score at or above `BING_THRESHOLD` raises `bing_off_target`. Timeout, HTTP errors (including rate limits) and invalid responses raise `relevance_check_failed`. These errors bypass Bing retries and allow the existing staggered search to start its next provider immediately. Bing cannot win the search race until its check passes. Other providers retain their current behavior.

The 0.1 threshold was measured on the saved Bing title-only corpus; it is not a universal threshold for arbitrary models or resource chunks. Benchmark data remains separate from runtime configuration.

## Resource behavior

Resource checks run only after backend visibility filtering. Only authorized candidate snippets, up to the requested result limit, are sent. Each includes a local numeric ID, title and snippet text; actual resource IDs and other metadata are not sent. All candidate questions are batched into one HTTP request.

A separately defined rule judges each snippet. Scores below `RESOURCE_THRESHOLD` preserve the original retrieval objects, order and citation identity. Scores at or above the threshold remove the snippet. An all-rejected set returns an empty result. A detector failure preserves the already-authorized retrievals and records an unavailable status; visibility failures still return no results and never reach the detector.

Resource checks remain off until their model/rule/threshold are evaluated on resource-specific examples. Sending private snippets to a configured remote endpoint is an explicit deployment choice; a local compatible endpoint is also supported.

## Search behavior

Search checks run inside the wizard's internal search endpoint whenever `SEARCH_ENABLED` is on, which is the default; there is no separate backend flag or request field. A wizard without a relevance base URL or model skips the check, returns all records and records `unconfigured` status. Only chunk records are considered: they are deduplicated by `resource_id`, keeping the highest-ranked chunk, and every other record is returned untouched. `raw_count` reports the number of chunk records before deduplication and filtering.

Each request carries the query plus up to `SEARCH_BATCH_SIZE` results with their title and complete chunk text, in rank order. Results whose title contains the query, or whose title is contained in the query, are always kept. Other results are dropped when the off-target score reaches `SEARCH_THRESHOLD`.

The whole response has a `SEARCH_TIMEOUT_SEC` budget shared by all batches; each batch uses the remaining budget as its HTTP timeout. A batch that times out or fails keeps its results and later batches keep theirs, so relevance filtering never removes results because the detector is unavailable. Individual missing or invalid answers keep only the affected result. Trace attributes record the input, chunk and kept counts, the threshold, and batch outcomes (`ok`, `partial` or `unavailable`). This differs from the strict Bing/resource answers: those still fail the whole response.

The backend's search and picker always request raw pages of 20 records from this endpoint, so each record they consider has been judged. The open API's chunk search asks for its own page size, up to 100 records, and also receives filtered results. Message search sends message records through the same endpoint, but the check only judges chunk records, so messages are returned untouched and make no relevance call. `SEARCH_ENABLED` in the wizard is the single switch for all consumers. Sending titles and full chunk text, including text the requesting user cannot access, to a remote endpoint is an explicit deployment and privacy decision; the backend still applies permissions to the final response.

## Observability and rollout

Bing requests propagate the active W3C trace context to the crawler. Its `Bing.*` and `BingBrowser.*` spans separate queueing, browser reuse/launch, context/page creation, navigation, result readiness, HTML retrieval, parsing and cleanup. Navigation attributes include available document DNS, connection, first-byte and download timings without recording URLs, HTML or cookies.

`RelevanceChecker` spans record question count, HTTP status, latency, Bing off-target score/threshold/rejection, and resource input/kept counts. Failed evaluations also record `error.type` and a sanitized `error.message` on the evaluate span and preserve the underlying exception as the `RelevanceError` cause. Remote error bodies, authorization headers and snippet contents are still excluded; JSON parse errors are reported as `invalid JSON response` instead of the body. Do not enable raw configuration debug logging in deployments that carry credentials.

`RelevanceChecker.search` additionally emits one `relevance.chunk` event per judged chunk, at most 127 per span. Each event carries `chunk.index` (position in the deduplicated chunk list), `chunk.batch_index` (0-based batch ordinal), `chunk.chunk_id`, `chunk.resource_id`, `chunk.title` when set, `chunk.chunk_type`, `chunk.line_range` when set, `chunk.text_length` when text is not None, and `chunk.decision` (`kept`, `off_target`, `title_protected`, `answer_missing`, or `fail_open`). Decisions with a score also record `chunk.off_target_score` and `chunk.relevant_score`; fail-open events record `chunk.fail_open_reason` (`error` or `deadline`) and no score. When more than 127 events would be emitted, a single `relevance.chunks_truncated` event reports `relevance.omitted_count`, keeping the span within the OpenTelemetry default of 128 events. Chunk text and the query are never recorded. Titles and resource IDs are user metadata and now appear in traces; treat trace access accordingly.

Deploy the shared change with both Wizard applications pinned to the same full commit. Bing and resource checks stay disabled until configured: enable Bing in the intended deployment after configuring the decision endpoint, and evaluate resource behavior separately before enabling it. Search checks are on by default and activate automatically once a deployment supplies the relevance base URL and model; without them, wizard search returns unfiltered results. Set `SEARCH_ENABLED=false` in the wizard to restore the previous unfiltered behavior for every consumer; there is no backend switch to change. Disabling this switch restores that tool's previous behavior without changing the crawler.
