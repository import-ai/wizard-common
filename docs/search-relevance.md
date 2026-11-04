# Optional search relevance checks

Wizard and Wizard Pro share this implementation. Checks are disabled by default: existing deployments need no new model service, credentials or network requests. The crawler service continues to return its original search response format.

## Configuration

Both applications read the same `OBW_TOOLS_RELEVANCE_*` environment variables. These settings do not use the Pro chat-model prefix.

| Variable suffix | Default | Meaning |
| --- | --- | --- |
| `BING_ENABLED` | `false` | Check a Bing result set before accepting it |
| `RESOURCE_ENABLED` | `false` | Filter visible resource snippets before citations |
| `SEARCH_ENABLED` | `false` | Filter global resource search and picker results by relevance |
| `BASE_URL` | unset | HTTP service base URL, excluding `/systemone` |
| `API_KEY` | unset | Optional Bearer credential; not required by unauthenticated local services |
| `MODEL` | unset | Decision model supported by the configured service |
| `TIMEOUT_SEC` | `2` | HTTP timeout in seconds |
| `BING_THRESHOLD` | `0.1` | Reject when the off-target score is greater than or equal to this value |
| `RESOURCE_THRESHOLD` | unset | Separately calibrated threshold, required when resource checks are enabled |
| `SEARCH_THRESHOLD` | unset | Separately calibrated threshold, required when search checks are enabled |
| `SEARCH_BATCH_SIZE` | `20` | Results judged per request, between 1 and 100 |
| `SEARCH_TIMEOUT_SEC` | `1` | Total relevance budget for one search response |

With all checks disabled, endpoint and model fields can remain unset. Explicitly enabling a check requires an HTTP(S) base URL and a model. Resource and search checks also require an explicit threshold. Invalid configuration fails validation rather than silently disabling an enabled check.

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

Search checks run inside the wizard's internal search endpoint when the caller sets `filter_relevance` and `SEARCH_ENABLED` is on. Only chunk records are considered: they are deduplicated by `resource_id`, keeping the highest-ranked chunk, and every other record is returned untouched. `raw_count` reports the number of chunk records before deduplication and filtering.

Each request carries the query plus up to `SEARCH_BATCH_SIZE` results with their title and complete chunk text, in rank order. Results whose title contains the query, or whose title is contained in the query, are always kept. Other results are dropped when the off-target score reaches `SEARCH_THRESHOLD`.

The whole response has a `SEARCH_TIMEOUT_SEC` budget shared by all batches; each batch uses the remaining budget as its HTTP timeout. A batch that times out or fails keeps its results and later batches keep theirs, so relevance filtering never removes results because the detector is unavailable. Individual missing or invalid answers keep only the affected result. Trace attributes record the input, chunk and kept counts, the threshold, and batch outcomes (`ok`, `partial` or `unavailable`). This differs from the strict Bing/resource answers: those still fail the whole response.

The backend calls this check only when its own `OBB_SEARCH_RELEVANCE_ENABLED` flag is on, and returns the same response shape either way. Both the paginated search and the GET picker ask for raw pages of 20 records while the check is active, so each record the backend considers has been judged. Sending titles and full chunk text, including text the requesting user cannot access, to a remote endpoint is an explicit deployment and privacy decision; the backend still applies permissions to the final response.

## Observability and rollout

Bing requests propagate the active W3C trace context to the crawler. Its `Bing.*` and `BingBrowser.*` spans separate queueing, browser reuse/launch, context/page creation, navigation, result readiness, HTML retrieval, parsing and cleanup. Navigation attributes include available document DNS, connection, first-byte and download timings without recording URLs, HTML or cookies.

`RelevanceChecker` spans record question count, HTTP status, latency, Bing off-target score/threshold/rejection, and resource input/kept counts. Remote error bodies, authorization headers and snippet contents are not added to these spans. Do not enable raw configuration debug logging in deployments that carry credentials.

Deploy the shared change with both Wizard applications pinned to the same full commit. Keep the optional flags disabled for existing open-source configurations. Enable Bing checks in the intended deployment after configuring the decision endpoint; evaluate resource behavior separately before enabling it. Search checks additionally need the backend's `OBB_SEARCH_RELEVANCE_ENABLED`; either switch being off restores the previous search behavior. Disabling a flag restores that tool's previous behavior without changing the crawler.
