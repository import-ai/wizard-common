from typing import Literal

from pydantic import BaseModel, Field

from wizard_common.grimoire.entity.retrieval import Citation
from wizard_common.grimoire.entity.tools import PrivateSearchTool, WebSearchTool

ChatRole = Literal["system", "user", "assistant", "tool"]


class BaseChatRequest(BaseModel):
    query: str


class ChatRequestOptions(BaseModel):
    tools: list[PrivateSearchTool | WebSearchTool] | None = Field(default=None)
    enable_thinking: bool | None = Field(default=None)
    edition: Literal["basic", "pro"] | None = Field(default=None)
    level: str | None = Field(default=None)
    merge_search: bool | None = Field(
        default=None, description="Whether to merge search results from multiple tools."
    )
    force_search: bool | None = Field(
        default=None, description="Whether to force search."
    )
    lang: Literal["简体中文", "English"] | None = Field(
        default=None, description="Language of the response."
    )
    tool_call: dict | None = Field(default=None)


class MessageAttrs(ChatRequestOptions):
    citations: list[Citation] | None = Field(default=None)
    context: dict | None = Field(default=None)
    compact: dict | None = Field(default=None)
    user_context: dict | None = Field(default=None)
    usage: dict | None = Field(default=None)
    composer: dict | None = Field(default=None)


class MessageDto(BaseModel):
    message: dict
    attrs: MessageAttrs | None = Field(default=None)


class ChatImageInput(BaseModel):
    attachment_id: str
    url: str
    name: str


class AgentRequest(BaseChatRequest, ChatRequestOptions):
    user_id: str | None = Field(default=None, description="User ID")
    share_id: str | None = Field(default=None, description="Share ID")
    namespace_id: str = Field(description="Namespace ID")
    conversation_id: str
    current_resource_id: str | None = Field(
        default=None, description="Currently opened resource ID"
    )
    messages: list[MessageDto] | None = Field(default=None)
    images: list[ChatImageInput] | None = Field(default=None)
    query_persisted: bool = False


class ChatBaseResponse(BaseModel):
    response_type: Literal[
        "bos", "delta", "eos", "error", "done", "checkpoint", "metrics", "query_attrs"
    ]


class ChatBOSResponse(ChatBaseResponse):
    response_type: Literal["bos"] = "bos"
    role: ChatRole
    attrs: MessageAttrs | None = Field(default=None)


class ChatEOSResponse(ChatBaseResponse):
    response_type: Literal["eos"] = "eos"
    role: ChatRole | None = Field(default=None)


class DeltaOpenAIMessage(BaseModel):
    content: str | None = Field(default=None)
    reasoning_content: str | None = Field(default=None)
    tool_calls: list[dict] | None = Field(default=None)
    tool_call_id: str | None = Field(default=None)


class ChatDeltaResponse(ChatBaseResponse):
    response_type: Literal["delta"] = "delta"
    message: DeltaOpenAIMessage = Field(default_factory=DeltaOpenAIMessage)
    attrs: MessageAttrs | None = Field(
        default=None, description="Attributes of the message."
    )


class ChatCheckpointResponse(ChatBaseResponse):
    response_type: Literal["checkpoint"] = "checkpoint"
    checkpoint: dict


class ChatCitationsResponse(ChatBaseResponse):
    response_type: Literal["citations"] = "citations"
    citations: list[Citation]


class ChatMetricsResponse(ChatBaseResponse):
    response_type: Literal["metrics"] = "metrics"
    tps: float
    tokens: int


class ChatErrorResponse(ChatBaseResponse):
    response_type: Literal["error"] = "error"
    message: str


class ChatQueryAttrsResponse(ChatBaseResponse):
    """Internal update for the query already persisted by the backend."""

    response_type: Literal["query_attrs"] = "query_attrs"
    attrs: MessageAttrs
