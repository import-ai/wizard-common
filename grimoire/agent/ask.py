from omnibox_wizard.wizard.config import Config
from wizard_common.grimoire.agent.agent import Agent


class Ask(Agent):
    def __init__(self, config: Config):
        super().__init__(config=config, system_prompt_template_name="ask.j2")
