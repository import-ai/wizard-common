from pydantic import BaseModel, Field, model_validator

BASIC_READ_ONLY_TOOLS = ["private_search", "web_search", "search"]

PRO_READ_ONLY_TOOLS = [
    "read_resource",
    "list_resources",
    "list_resource_attachments",
    "get_resource_attachment",
    "view_attachment",
    "describe_resource",
    "read_root",
    "get_agent_asset_resource_id",
    "resource_filter",
    "resource_search",
    "web_search",
    "list_tags",
    "tag_filter",
    "list_resource_comments",
    "get_rss_folder_config",
    "get_smart_folder_config",
    "load_docs",
    "get_current_info",
    "list_local_devices",
    "search_conversation_messages",
    "read_conversation_message",
]


class LoopGuardConfig(BaseModel):
    enabled: bool = True
    read_warn_occurrence: int = 2
    read_block_occurrence: int = 3
    read_stop_occurrence: int = 4
    write_block_occurrence: int = 2
    write_stop_occurrence: int = 3
    failed_block_occurrence: int = 2
    failed_stop_occurrence: int = 3
    cycle_warn_repetitions: int = 2
    cycle_block_repetitions: int = 3
    cycle_stop_repetitions: int = 4
    cycle_min_length: int = 2
    cycle_max_length: int = 4
    history_limit: int = 64
    read_only_tools: list[str] = Field(
        default_factory=lambda: list(BASIC_READ_ONLY_TOOLS)
    )
    exempt_tools: list[str] = Field(default_factory=lambda: ["wait"])
    polling_tools: list[str] = Field(default_factory=lambda: ["list_resource_tasks"])

    @model_validator(mode="after")
    def validate_thresholds(self):
        if not (
            self.read_warn_occurrence
            < self.read_block_occurrence
            < self.read_stop_occurrence
        ):
            raise ValueError("read thresholds must satisfy warn < block < stop")
        if not self.write_block_occurrence < self.write_stop_occurrence:
            raise ValueError("write thresholds must satisfy block < stop")
        if not self.failed_block_occurrence < self.failed_stop_occurrence:
            raise ValueError("failed thresholds must satisfy block < stop")
        if not (
            self.cycle_warn_repetitions
            <= self.cycle_block_repetitions
            <= self.cycle_stop_repetitions
        ):
            raise ValueError("cycle thresholds must be non-decreasing")
        if not 2 <= self.cycle_min_length <= self.cycle_max_length:
            raise ValueError("cycle lengths must satisfy 2 <= min <= max")
        if self.history_limit < 2:
            raise ValueError("history_limit must be at least 2")
        return self

    @classmethod
    def for_basic(cls) -> "LoopGuardConfig":
        return cls()


class LoopGuardProConfig(LoopGuardConfig):
    read_only_tools: list[str] = Field(
        default_factory=lambda: list(PRO_READ_ONLY_TOOLS)
    )
