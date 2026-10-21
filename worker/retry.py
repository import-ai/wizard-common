import httpx
import openai
from weaviate.exceptions import WeaviateBaseError


class RetryableTaskError(ValueError):
    pass


def is_retryable(error: Exception) -> bool:
    if isinstance(error, (httpx.HTTPStatusError, openai.APIStatusError)):
        status = error.response.status_code
        return status in (409, 429) or status >= 500
    return isinstance(
        error,
        (
            RetryableTaskError,
            httpx.TransportError,
            openai.APIConnectionError,
            WeaviateBaseError,
        ),
    )
