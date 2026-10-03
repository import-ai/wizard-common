import asyncio
from functools import partial
from typing import Any

import httpx
from common.exception import CommonException
from common.trace_info import TraceInfo
from opentelemetry import trace

from wizard_common.grimoire.entity.tools import BaseTool
from wizard_common.grimoire.retriever.base import BaseRetriever, SearchFunction
from wizard_common.grimoire.retriever.searxng import SearXNGRetrieval

tracer = trace.get_tracer(__name__)


class BingSearch(BaseRetriever):
    """Client for playwright-server Bing search API."""

    def __init__(
        self,
        base_url: str,
        *,
        mkt: str = "zh-CN",
        setlang: str = "zh-Hans",
        cc: str = "CN",
        page_start: int = 1,
        page_end: int | None = None,
        timeout_sec: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.mkt = mkt
        self.setlang = setlang
        self.cc = cc
        self.page_start = page_start
        self.page_end = page_end
        self.timeout_sec = timeout_sec

    def _build_params(self, query: str) -> dict[str, Any]:
        params: dict[str, Any] = {
            "q": query,
            "format": "json",
            "page_start": self.page_start,
            "mkt": self.mkt,
            "setlang": self.setlang,
            "cc": self.cc,
        }
        if self.page_end is not None:
            params["page_end"] = self.page_end
        return params

    @staticmethod
    def _is_usable_result(result: dict) -> bool:
        return bool(
            str(result.get("title") or "").strip()
            and str(result.get("url") or "").strip()
            and str(result.get("content") or "").strip()
        )

    @tracer.start_as_current_span("BingSearch.search_once")
    async def search_once(
        self, query: str, *, trace_info: TraceInfo | None = None
    ) -> list[SearXNGRetrieval]:
        async with httpx.AsyncClient(
            base_url=self.base_url, timeout=self.timeout_sec
        ) as client:
            response = await client.get(
                "/api/v1/search",
                params=self._build_params(query),
            )
            response.raise_for_status()

        payload = response.json()
        results = payload.get("results") or []
        usable = [item for item in results if self._is_usable_result(item)]
        if not usable:
            # Defensive: playwright-server should already 502 on this case.
            raise RuntimeError("bing_no_usable_results")

        retrievals = [SearXNGRetrieval(result=item) for item in usable]
        if trace_info:
            trace_info.debug({"len(retrievals)": len(retrievals), "engine": "bing"})
        return retrievals

    @tracer.start_as_current_span("BingSearch.search")
    async def search(
        self,
        query: str,
        *,
        k: int | None = None,
        retry_cnt: int = 1,
        retry_sleep: float = 1,
        trace_info: TraceInfo | None = None,
    ) -> list[SearXNGRetrieval]:
        last_error: Exception | None = None
        for i in range(retry_cnt or 1):
            try:
                retrievals = await self.search_once(query, trace_info=trace_info)
                if retrievals:
                    return retrievals[:k] if k else retrievals
            except Exception as e:
                last_error = e
                if trace_info:
                    trace_info.warning(
                        {
                            "message": f"Bing search failed, retrying {i + 1}/{retry_cnt}",
                            "query": query,
                            "error": CommonException.parse_exception(e),
                        }
                    )
            await asyncio.sleep(retry_sleep)

        if last_error:
            raise last_error
        return []

    def get_function(self, tool: BaseTool, **kwargs) -> SearchFunction:
        return partial(self.search, **kwargs)

    @classmethod
    def get_schema(cls) -> dict:
        return cls.generate_schema(
            "web_search",
            'Search the web for public information. Return in <cite id=""></cite> format.',
            display_name={"zh": "网络搜索", "en": "Web Search"},
        )
