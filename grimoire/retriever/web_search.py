import asyncio
import os
import time
from functools import partial

from common.trace_info import TraceInfo
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from wizard_common.grimoire.config import (
    BingSearchConfig,
    DeepSeekSearchConfig,
    SearXNGConfig,
)
from wizard_common.grimoire.entity.tools import BaseTool
from wizard_common.grimoire.retriever.base import BaseRetriever, SearchFunction
from wizard_common.grimoire.retriever.bing_search import BingSearch
from wizard_common.grimoire.retriever.deepseek_search import DeepSeekSearch
from wizard_common.grimoire.retriever.searxng import SearXNG, SearXNGRetrieval

tracer = trace.get_tracer(__name__)


class WebSearch(BaseRetriever):
    """Race Bing, DeepSeek and SearXNG with staggered starts."""

    def __init__(
        self,
        searxng: SearXNGConfig,
        bing_search: BingSearchConfig,
        deepseek_search: DeepSeekSearchConfig | None = None,
    ):
        self.fallback = SearXNG(
            base_url=searxng.base_url,
            engines=searxng.engines,
        )

        bing_base_url = (
            bing_search.base_url or os.getenv("OBW_TASK_SCRAPE_BASE_URL") or ""
        ).rstrip("/")
        self.primary: BingSearch | None = None
        if bing_search.enabled and bing_base_url:
            self.primary = BingSearch(
                base_url=bing_base_url,
                mkt=bing_search.mkt,
                setlang=bing_search.setlang,
                cc=bing_search.cc,
                page_start=bing_search.page_start,
                page_end=bing_search.page_end,
                timeout_sec=bing_search.timeout_sec,
            )
        self.deepseek = (
            DeepSeekSearch(deepseek_search.api_key)
            if deepseek_search and deepseek_search.api_key
            else None
        )

    async def search(
        self,
        query: str,
        *,
        k: int | None = None,
        trace_info: TraceInfo | None = None,
        **kwargs,
    ) -> list[SearXNGRetrieval]:
        with tracer.start_as_current_span("WebSearch.search") as span:
            started_at = time.monotonic()
            providers = (
                ("bing", 0, self.primary),
                ("deepseek", 1, self.deepseek),
                ("searxng", 2, self.fallback),
            )
            span.set_attribute("web_search.winner", "none")
            span.set_attribute("web_search.status", "empty")
            span.set_attribute("web_search.result_count", 0)
            for name, _, provider in providers:
                span.set_attributes(
                    {
                        f"web_search.{name}.started": False,
                        f"web_search.{name}.status": "not_started"
                        if provider
                        else "disabled",
                    }
                )

            async def run_provider(name, delay, provider):
                await asyncio.sleep(max(0, started_at + delay - time.monotonic()))
                call_started_at = time.monotonic()
                prefix = f"web_search.{name}"
                span.set_attributes(
                    {
                        f"{prefix}.started": True,
                        f"{prefix}.start_delay_ms": (call_started_at - started_at)
                        * 1000,
                    }
                )
                with tracer.start_as_current_span(
                    "WebSearch.provider",
                    attributes={"search.provider": name},
                    record_exception=False,
                    set_status_on_exception=False,
                ) as provider_span:
                    status = "cancelled"
                    results = []
                    try:
                        results = await provider.search(
                            query, k=k, trace_info=trace_info, **kwargs
                        )
                        results = [
                            r for r in results if BingSearch._is_usable_result(r.result)
                        ]
                        status = "success" if results else "empty"
                        return name, results
                    except Exception as e:
                        status = "error"
                        results = []
                        provider_span.set_attribute("error.type", type(e).__name__)
                        provider_span.set_status(StatusCode.ERROR)
                        return name, []
                    finally:
                        latency_ms = (time.monotonic() - call_started_at) * 1000
                        provider_span.set_attributes(
                            {
                                "search.status": status,
                                "search.latency_ms": latency_ms,
                                "search.result_count": len(results),
                            }
                        )
                        span.set_attributes(
                            {
                                f"{prefix}.status": status,
                                f"{prefix}.latency_ms": latency_ms,
                            }
                        )

            tasks = [
                asyncio.create_task(run_provider(name, delay, provider))
                for name, delay, provider in providers
                if provider is not None
            ]
            try:
                for task in asyncio.as_completed(tasks):
                    name, results = await task
                    if results:
                        span.set_attributes(
                            {
                                "web_search.winner": name,
                                "web_search.status": "success",
                                "web_search.result_count": len(results),
                            }
                        )
                        return results
                return []
            except asyncio.CancelledError:
                span.set_attribute("web_search.status", "cancelled")
                raise
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                try:
                    await asyncio.gather(*tasks, return_exceptions=True)
                except asyncio.CancelledError:
                    span.set_attribute("web_search.status", "cancelled")
                    raise
                finally:
                    span.set_attribute(
                        "web_search.latency_ms", (time.monotonic() - started_at) * 1000
                    )

    def get_function(self, tool: BaseTool, **kwargs) -> SearchFunction:
        return partial(self.search, **kwargs)

    @classmethod
    def get_schema(cls) -> dict:
        return cls.generate_schema(
            "web_search",
            'Search the web for public information. Return in <cite id=""></cite> format.',
            display_name={"zh": "网络搜索", "en": "Web Search"},
        )
