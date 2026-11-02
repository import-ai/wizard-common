import json as jsonlib
from typing import AsyncIterable

from common.model_dump import model_dump
from common.trace_info import TraceInfo
from openai.types.chat import ChatCompletionAssistantMessageParam
from opentelemetry import trace

from wizard_common.grimoire.entity.api import (
    ChatBaseResponse,
    ChatBOSResponse,
    ChatDeltaResponse,
    ChatEOSResponse,
    MessageDto,
)
from wizard_common.grimoire.entity.chunk import ResourceChunkRetrieval
from wizard_common.grimoire.entity.retrieval import (
    BaseRetrieval,
    make_citation_id,
    retrievals2prompt,
)
from wizard_common.grimoire.entity.tools import ToolExecutorConfig
from wizard_common.grimoire.retriever.searxng import SearXNGRetrieval

tracer = trace.get_tracer(__name__)

UNKNOWN_TOOL_ERROR = (
    "The requested tool is not available to this agent. "
    "Use only the tools provided in this request."
)
INVALID_ARGUMENTS_ERROR = "Invalid arguments for the requested tool."
TOOL_FAILED_ERROR = "The requested tool failed."


def cmp(retrieval: BaseRetrieval) -> tuple[int, str, int, float]:
    if isinstance(
        retrieval, ResourceChunkRetrieval
    ):  # GROUP BY resource_id ORDER BY start_index ASC
        return 0, retrieval.chunk.resource_id, retrieval.chunk.start_index or 0, 0.0
    elif isinstance(retrieval, SearXNGRetrieval):  # ORDER BY score.rerank DESC
        return 1, "", 0, -(retrieval.score.rerank or 0.0)
    raise ValueError(f"Unknown retrieval type: {type(retrieval)}")


def retrieval_wrapper(tool_call_id: str, retrievals: list[BaseRetrieval]) -> MessageDto:
    content: str = retrievals2prompt(retrievals)
    return MessageDto.model_validate(
        {
            "message": {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            },
            "attrs": {
                "citations": [retrieval.to_citation() for retrieval in retrievals],
                "tool_call": {
                    "status": "success",
                },
            },
        }
    )


def get_citation_cnt(messages: list[MessageDto]) -> int:
    return sum(
        len(message.attrs.citations) if message.attrs and message.attrs.citations else 0
        for message in messages
    )


def error_tool_wrapper(tool_call_id: str, error: str) -> MessageDto:
    return MessageDto.model_validate(
        {
            "message": {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": f"Error: {error}",
            },
            "attrs": {"tool_call": {"status": "failed", "error": error}},
        }
    )


class ToolExecutor:
    def __init__(self, config: dict[str, ToolExecutorConfig]):
        self.config: dict[str, ToolExecutorConfig] = config
        self.tools: list[dict] = [config["schema"] for config in config.values()]

    @staticmethod
    def _parse_arguments(raw_arguments, logger: TraceInfo) -> dict | None:
        if isinstance(raw_arguments, dict):
            return raw_arguments
        try:
            function_args = jsonlib.loads(raw_arguments)
        except (TypeError, ValueError) as error:
            logger.error({"message": "Invalid arguments", "error": str(error)})
            return None
        if not isinstance(function_args, dict):
            logger.error(
                {
                    "message": "Invalid arguments",
                    "error": (
                        f"Expected a JSON object, got {type(function_args).__name__}"
                    ),
                }
            )
            return None
        return function_args

    @staticmethod
    async def _failed_tool_result(tool_call_id: str, error: str):
        message_dto = error_tool_wrapper(tool_call_id, error)
        yield ChatDeltaResponse.model_validate(
            message_dto.model_dump(exclude_none=True)
        )
        yield message_dto
        yield ChatEOSResponse()

    async def astream(
        self,
        message_dtos: list[MessageDto],
        trace_info: TraceInfo,
    ) -> AsyncIterable[ChatBaseResponse | MessageDto]:
        with tracer.start_as_current_span("tool_executor.astream"):
            message: ChatCompletionAssistantMessageParam = message_dtos[-1].message
            if tool_calls := message.get("tool_calls", []):
                for tool_call in tool_calls:
                    function = tool_call["function"]
                    tool_call_id: str = str(tool_call["id"])
                    function_name = function["name"]
                    logger = trace_info.get_child(
                        addition_payload={
                            "tool_call_id": tool_call_id,
                            "function_name": function_name,
                        }
                    )

                    yield ChatBOSResponse(role="tool")

                    function_args = self._parse_arguments(
                        function.get("arguments"), logger
                    )
                    if function_args is None:
                        async for chunk in self._failed_tool_result(
                            tool_call_id, INVALID_ARGUMENTS_ERROR
                        ):
                            yield chunk
                        continue

                    logger = logger.bind(function_args=function_args)
                    if function_name not in self.config or not function_name.endswith(
                        "search"
                    ):
                        logger.error({"message": "Unsupported function"})
                        async for chunk in self._failed_tool_result(
                            tool_call_id, UNKNOWN_TOOL_ERROR
                        ):
                            yield chunk
                        continue

                    try:
                        with tracer.start_as_current_span(
                            f"tool_executor.astream.{function_name}"
                        ) as func_span:
                            func_span.set_attributes(
                                {
                                    "tool_call_id": tool_call_id,
                                    "function_name": function_name,
                                    "function_args": jsonlib.dumps(
                                        function_args,
                                        ensure_ascii=False,
                                        separators=(",", ":"),
                                    ),
                                }
                            )
                            func = self.config[function_name]["func"]
                            result = await func(**function_args)
                            logger.info({"result": model_dump(result)})

                        current_cite_cnt: int = get_citation_cnt(message_dtos)
                        assert isinstance(result, list), (
                            f"Expected list of retrievals, got {type(result)}"
                        )
                        assert all(isinstance(r, BaseRetrieval) for r in result), (
                            f"Expected all items to be BaseRetrieval, got {[type(r) for r in result]}"
                        )
                        sorted_retrievals = sorted(result, key=cmp)
                        for i, r in enumerate(sorted_retrievals):
                            citation = r.to_citation()
                            line_range = (
                                r.chunk.line_range
                                if isinstance(r, ResourceChunkRetrieval)
                                else None
                            )
                            r.id = make_citation_id(
                                current_cite_cnt + i + 1,
                                citation.title,
                                citation.snippet,
                                line_range,
                            )
                        message_dto: MessageDto = retrieval_wrapper(
                            tool_call_id=tool_call_id, retrievals=sorted_retrievals
                        )
                    except Exception:
                        logger.exception({"message": "Tool execution failed"})
                        message_dto = error_tool_wrapper(
                            tool_call_id, TOOL_FAILED_ERROR
                        )
                        yield ChatDeltaResponse.model_validate(
                            message_dto.model_dump(exclude_none=True)
                        )
                        yield message_dto
                        raise

                    yield ChatDeltaResponse.model_validate(
                        message_dto.model_dump(exclude_none=True)
                    )
                    yield message_dto
                    yield ChatEOSResponse()
