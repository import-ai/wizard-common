import httpx
from common.trace_info import TraceInfo

from wizard_common.grimoire.retriever.searxng import SearXNGRetrieval


class DeepSeekSearch:
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def search(
        self,
        query: str,
        *,
        k: int | None = None,
        trace_info: TraceInfo | None = None,
        **kwargs,
    ) -> list[SearXNGRetrieval]:
        if not query.strip() or len(query) > 128:
            raise ValueError("DeepSeek search query must contain 1 to 128 characters")

        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                "https://api.deepseek.com/v1/tools/search",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"query": query, "count": 10, "need_content": False},
            )
            response.raise_for_status()

        results = response.json()["results"]
        if not isinstance(results, list):
            raise ValueError("Invalid DeepSeek search results")
        retrievals = []
        for item in results:
            if not isinstance(item, dict):
                continue
            title, url = item.get("title"), item.get("url")
            content = item.get("summary")
            if not isinstance(content, str) or not content.strip():
                content = item.get("snippet")
            if not all(isinstance(v, str) and v.strip() for v in (title, url, content)):
                continue
            retrievals.append(
                SearXNGRetrieval(
                    result={"title": title, "url": url, "content": content}
                )
            )
        return retrievals[:k] if k else retrievals
