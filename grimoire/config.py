from typing import Literal
from urllib.parse import urlsplit

from openai import NOT_GIVEN, NotGiven
from pydantic import BaseModel, Field, SecretStr, model_validator

from wizard_common.config import OpenAIConfig


class WeaviateConfig(BaseModel):
    host: str | None = Field(default=None)
    port: int = Field(default=8080)
    grpc_port: int = Field(default=50051)
    api_key: str | None = Field(default=None)


class VectorConfig(BaseModel):
    embedding: OpenAIConfig
    weaviate: WeaviateConfig = Field(default_factory=WeaviateConfig)
    batch_size: int = Field(default=1)
    dimension: int = Field(default=1536)


GrimoireOpenAIConfigKey = Literal["mini", "default", "large", "vision"]


class GrimoireOpenAIConfig(BaseModel):
    mini: OpenAIConfig = Field(default_factory=OpenAIConfig)
    mini_thinking: OpenAIConfig = Field(default=None)
    default: OpenAIConfig
    default_thinking: OpenAIConfig = Field(default=None)
    large: OpenAIConfig = Field(default_factory=OpenAIConfig)
    large_thinking: OpenAIConfig = Field(default=None)
    vision: OpenAIConfig = Field(default_factory=OpenAIConfig)
    vision_thinking: OpenAIConfig = Field(default=None)

    def get_config(
        self,
        key: GrimoireOpenAIConfigKey,
        thinking: bool = False,
        default: OpenAIConfig | None | NotGiven = NOT_GIVEN,
    ) -> OpenAIConfig | None:
        k = key if not thinking else f"{key}_thinking"
        openai_config: OpenAIConfig = getattr(self, k, None)
        if openai_config is None:
            if isinstance(default, NotGiven):
                raise KeyError(f"OpenAIConfig for key '{k}' not found.")
            return default
        return OpenAIConfig(
            base_url=openai_config.base_url or self.default.base_url,
            api_key=openai_config.api_key or self.default.api_key,
            model=openai_config.model or self.default.model,
        )


class GrimoireConfig(BaseModel):
    openai: GrimoireOpenAIConfig = Field(default=None)
    custom_tool_call: bool = Field(default=False)


class RerankerConfig(BaseModel):
    openai: OpenAIConfig = Field(default=None)
    threshold: float = Field(default=None)
    k: int = Field(default=None)


class SearXNGConfig(BaseModel):
    base_url: str
    engines: str | None = Field(default=None)


class BingSearchConfig(BaseModel):
    """playwright-server Bing search API (primary web search).

    If base_url is omitted, runtime should fall back to OBW_TASK_SCRAPE_BASE_URL
    because Bing search and scrape share the same playwright-server.
    """

    base_url: str | None = Field(
        default=None,
        description=(
            "Optional override for playwright-server base URL. "
            "When empty, fall back to OBW_TASK_SCRAPE_BASE_URL."
        ),
    )
    enabled: bool = Field(default=True)
    mkt: str = Field(default="zh-CN")
    setlang: str = Field(default="zh-Hans")
    cc: str = Field(default="CN")
    page_start: int = Field(default=1)
    page_end: int | None = Field(default=None)
    timeout_sec: float = Field(default=60.0)


class DeepSeekSearchConfig(BaseModel):
    api_key: str | None = Field(default=None)


class RelevanceConfig(BaseModel):
    base_url: str | None = None
    api_key: SecretStr | None = None
    model: str | None = None
    timeout_sec: float = Field(default=2.0, gt=0, allow_inf_nan=False)
    bing_enabled: bool = False
    bing_threshold: float = Field(default=0.1, ge=0, le=1)
    resource_enabled: bool = False
    resource_threshold: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_enabled_checks(self):
        if self.bing_enabled or self.resource_enabled:
            url = urlsplit(self.base_url or "")
            if (
                url.scheme not in ("http", "https")
                or not url.hostname
                or url.query
                or url.fragment
            ):
                raise ValueError("Relevance checks require an HTTP(S) base_url")
            if not self.model or not self.model.strip():
                raise ValueError("Relevance checks require a model")
        if self.resource_enabled and self.resource_threshold is None:
            raise ValueError("Resource checks require a calibrated resource_threshold")
        return self


class ToolsConfig(BaseModel):
    searxng: SearXNGConfig
    bing_search: BingSearchConfig = Field(default_factory=BingSearchConfig)
    deepseek_search: DeepSeekSearchConfig = Field(default_factory=DeepSeekSearchConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    relevance: RelevanceConfig = Field(default_factory=RelevanceConfig)


class GrimoireAgentConfig(BaseModel):
    vector: VectorConfig
    grimoire: GrimoireConfig
    tools: ToolsConfig
    max_tool_rounds: int = Field(default=10, ge=1)
