import os
from functools import partial

from common.exception import CommonException
from common.trace_info import TraceInfo

from wizard_common.grimoire.config import BingSearchConfig, SearXNGConfig
from wizard_common.grimoire.entity.tools import BaseTool
from wizard_common.grimoire.retriever.base import BaseRetriever, SearchFunction
from wizard_common.grimoire.retriever.bing_search import BingSearch
from wizard_common.grimoire.retriever.searxng import SearXNG, SearXNGRetrieval


class WebSearch(BaseRetriever):
    """Bing primary + SearXNG fallback, exposed as web_search."""

    def __init__(self, searxng: SearXNGConfig, bing_search: BingSearchConfig):
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

    async def search(
        self,
        query: str,
        *,
        k: int | None = None,
        trace_info: TraceInfo | None = None,
        **kwargs,
    ) -> list[SearXNGRetrieval]:
        if self.primary is not None:
            try:
                results = await self.primary.search(
                    query, k=k, trace_info=trace_info, **kwargs
                )
                if results:
                    return results
                if trace_info:
                    trace_info.warning(
                        {
                            "message": "Bing returned no results, fallback to SearXNG",
                            "query": query,
                        }
                    )
            except Exception as e:
                if trace_info:
                    trace_info.warning(
                        {
                            "message": "Bing search failed, fallback to SearXNG",
                            "query": query,
                            "error": CommonException.parse_exception(e),
                        }
                    )

        return await self.fallback.search(query, k=k, trace_info=trace_info, **kwargs)

    def get_function(self, tool: BaseTool, **kwargs) -> SearchFunction:
        return partial(self.search, **kwargs)

    @classmethod
    def get_schema(cls) -> dict:
        return cls.generate_schema(
            "web_search",
            'Search the web for public information. Return in <cite id=""></cite> format.',
            display_name={"zh": "网络搜索", "en": "Web Search"},
        )
