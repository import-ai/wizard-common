import os
from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from wizard_common.config import OpenAIConfig


class ThinkingParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enable_thinking: StrictBool | None = None
    reasoning_effort: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_mode(self):
        if self.enable_thinking is not None and self.reasoning_effort is not None:
            raise ValueError("Configure one thinking parameter per level")
        return self


class ThinkingLevel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    model: str = Field(min_length=1)
    parameters: ThinkingParameters = Field(default_factory=ThinkingParameters)

    def resolve(self, connection: OpenAIConfig) -> tuple[OpenAIConfig, dict]:
        parameters = self.parameters.model_dump(exclude_none=True)
        if "enable_thinking" in parameters:
            parameters = {"extra_body": parameters}
        return connection.model_copy(update={"model": self.model}), parameters


class ThinkingEdition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    default_level: str
    levels: list[ThinkingLevel] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_levels(self):
        ids = [level.id for level in self.levels]
        if len(ids) != len(set(ids)) or self.default_level not in ids:
            raise ValueError("Levels must be unique and include default_level")
        return self

    def select(self, level: str) -> ThinkingLevel:
        for item in self.levels:
            if item.id == level:
                return item
        raise ValueError("Unsupported thinking level")


class ThinkingModels(BaseModel):
    # Other editions are consumed by their respective Wizard services.
    basic: ThinkingEdition

    def public_config(self) -> dict:
        return {
            "basic": {
                "default": {"edition": "basic", "level": self.basic.default_level},
                "levels": [
                    {"edition": "basic", "level": item.id} for item in self.basic.levels
                ],
            }
        }


@lru_cache(maxsize=1)
def get_thinking_models() -> ThinkingModels | None:
    value = os.environ.get("OBW_THINKING_MODELS")
    return ThinkingModels.model_validate_json(value) if value else None


def validate_selection(edition: str | None, level: str | None):
    if edition is None and level is None:
        return
    models = get_thinking_models()
    if edition != "basic" or level is None or models is None:
        raise ValueError("Unsupported thinking selection")
    models.basic.select(level)
