from collections.abc import Iterator

from opentelemetry import trace

tracer = trace.get_tracer(__name__)
EVENTS_PER_SPAN = 100


def event_spans(span: trace.Span, count: int, name: str) -> Iterator[trace.Span]:
    """Keep overflow events in bounded child spans instead of dropping them."""
    for start in range(0, count, EVENTS_PER_SPAN):
        size = min(EVENTS_PER_SPAN, count - start)
        if start == 0:
            yield from (span for _ in range(size))
        else:
            with tracer.start_as_current_span(
                name,
                context=trace.set_span_in_context(span),
                attributes={"events.batch_offset": start, "events.batch_count": size},
            ) as child:
                yield from (child for _ in range(size))
