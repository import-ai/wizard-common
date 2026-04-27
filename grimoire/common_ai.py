from opentelemetry import trace

from wizard_common.grimoire.config import GrimoireOpenAIConfig
from omnibox_wizard.worker.agent.chat_title_generator import (
    ChatTitleGenerator,
    ChatTitleGenerateOutput,
)

tracer = trace.get_tracer(__name__)


class CommonAI:
    def __init__(self, config: GrimoireOpenAIConfig):
        self.config: GrimoireOpenAIConfig = config
        self.chat_title_generator = ChatTitleGenerator(config)

    @tracer.start_as_current_span("CommonAI.title")
    async def title(self, text: str, *, lang: str | None = None) -> str:
        """
        Create title according to the given text
        """
        output: ChatTitleGenerateOutput = await self.chat_title_generator.ainvoke(
            {
                "text": text,
                "lang": lang or "简体中文",
            }
        )
        return output.title
