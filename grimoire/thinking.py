import json
import os
from functools import lru_cache

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    model_validator,
)

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


class DefaultThinkingSteps(BaseModel):
    model_config = ConfigDict(extra="forbid")
    default_step: str
    steps: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_steps(self):
        if (
            len(self.steps) != len(set(self.steps))
            or self.default_step not in self.steps
        ):
            raise ValueError("Steps must be unique and include default_step")
        return self


class ModelPrice(BaseModel):
    """Displayed credits per million tokens; numerically internal units per token."""

    model_config = ConfigDict(extra="forbid")
    input: StrictInt = Field(ge=0)
    input_cached: StrictInt = Field(ge=0)
    output: StrictInt = Field(ge=0)


class ThinkingModels(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prices: dict[str, ModelPrice] = Field(default_factory=dict)
    basic: ThinkingEdition | None = None
    pro: ThinkingEdition | None = None
    default: DefaultThinkingSteps | None = None

    @model_validator(mode="after")
    def validate_references(self):
        if self.basic is None and self.pro is None:
            raise ValueError("At least one model edition is required")
        if self.default:
            for step in self.default.steps:
                edition, separator, level = step.partition(".")
                if not separator:
                    raise ValueError("Default steps must be edition.level references")
                self.select(edition, level)
        return self

    def select(self, edition: str, level: str) -> ThinkingLevel:
        group = getattr(self, edition, None) if edition in ("basic", "pro") else None
        if group is None:
            raise ValueError("Unsupported model edition")
        return group.select(level)

    def public_config(self, editions=("basic", "pro")) -> dict:
        config = {}
        steps = []
        for edition in editions:
            group = getattr(self, edition)
            if group is None:
                continue
            levels = [{"edition": edition, "level": item.id} for item in group.levels]
            config[edition] = {
                "default": {"edition": edition, "level": group.default_level},
                "levels": levels,
            }
            steps.extend(f"{edition}.{item.id}" for item in group.levels)
        if not steps:
            return config
        ordered = (
            [step for step in self.default.steps if step in steps]
            if self.default
            else steps
        )
        if not ordered:
            return config
        default = self.default.default_step if self.default else steps[0]
        if default not in ordered:
            default = ordered[0]

        def selection(step):
            edition, level = step.split(".", 1)
            return {"edition": edition, "level": level}

        config["default"] = {
            "default": selection(default),
            "levels": [selection(step) for step in ordered],
        }
        return config


@lru_cache(maxsize=1)
def get_thinking_models() -> ThinkingModels | None:
    value = os.environ.get("OBW_MODELS")
    return ThinkingModels.model_validate_json(value) if value else None


def validate_selection(edition: str | None, level: str | None, service_edition="basic"):
    if edition is None and level is None:
        return
    models = get_thinking_models()
    if edition != service_edition or level is None or models is None:
        raise ValueError("Unsupported thinking selection")
    models.select(edition, level)


def billing_headers(request, service_edition: str) -> dict[str, str]:
    """Snapshot server-selected pricing without exposing it in public SSE events."""
    validate_selection(request.edition, request.level, service_edition)
    models = get_thinking_models()
    group = getattr(models, service_edition) if models else None
    if group and request.level is None:
        request.edition = service_edition
        request.level = group.default_level
    price = None
    if group:
        selected = group.select(request.level)
        price = models.prices.get(selected.model)
    return {
        "X-Omnibox-Billing": json.dumps(
            {
                "edition": service_edition,
                "price": price.model_dump() if price else None,
            }
        )
    }
