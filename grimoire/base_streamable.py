from abc import abstractmethod
from typing import AsyncIterable, TypeVar

from common.trace_info import TraceInfo
from wizard_common.grimoire.entity.api import BaseChatRequest, ChatBaseResponse

ChatResponse = TypeVar("ChatResponse", bound=ChatBaseResponse)


class BaseStreamable:
    @abstractmethod
    async def astream(
        self, trace_info: TraceInfo, request: BaseChatRequest
    ) -> AsyncIterable[ChatResponse]:
        raise NotImplementedError
