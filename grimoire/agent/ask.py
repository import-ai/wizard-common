from wizard_common.grimoire.config import GrimoireAgentConfig
from wizard_common.grimoire.agent.agent import Agent


class Ask(Agent):
    def __init__(self, config: GrimoireAgentConfig):
        super().__init__(config=config, system_prompt_template_name="ask.j2")
