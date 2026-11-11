import hashlib
import json
import re
import uuid
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wizard_common.grimoire.agent.tool_loop import canonical_tool_key

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

RESOURCE_HASH_NAMESPACE = "resource_hash"

_URL_PATTERN = re.compile(r"https?://[^\s\"'<>\\]+")
_ISO_TIMESTAMP_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
)
_CITATION_PREFIX_PATTERN = re.compile(r"\bC\d+-")
_UUID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_HISTORY_MESSAGE_PATTERN = re.compile(
    r'<message id="([^"]+)"[^>]*>(.*?)</message>', re.DOTALL
)
_WHITESPACE_PATTERN = re.compile(r"\s+")


class GuardAction(StrEnum):
    EXECUTE = "execute"
    WARN = "warn"
    BLOCK = "block"
    STOP = "stop"


class GuardReason(StrEnum):
    DUPLICATE = "duplicate"
    DUPLICATE_IN_ROUND = "duplicate_in_round"
    CYCLE = "cycle"
    WRITE_REPEAT = "write_repeat"
    FAILED_REPEAT = "failed_repeat"
    ROUND_LIMIT = "round_limit"


class GuardDecision(BaseModel):
    model_config = ConfigDict(frozen=True)
    tool_call_id: str
    tool_name: str
    action: GuardAction
    reason: GuardReason | None
    occurrence: int
    repeat_count: int
    cycle_length: int | None = None
    detail: str | None = None


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


class _SignatureState:
    __slots__ = (
        "attempts",
        "last_round",
        "last_fingerprint",
        "last_failed",
        "failed_blocks",
        "last_error",
    )

    def __init__(self) -> None:
        self.attempts = 0
        self.last_round: int | None = None
        self.last_fingerprint: str | None = None
        self.last_failed = False
        self.failed_blocks = 0
        self.last_error: str | None = None


class _HistoryEntry:
    __slots__ = ("key", "fingerprint")

    def __init__(self, key: tuple[str, str], fingerprint: str | None) -> None:
        self.key = key
        self.fingerprint = fingerprint

    def matches(self, other: "_HistoryEntry") -> bool:
        return self.key == other.key and self.fingerprint == other.fingerprint


_ACTION_RANK = {
    GuardAction.EXECUTE: 0,
    GuardAction.WARN: 1,
    GuardAction.BLOCK: 2,
    GuardAction.STOP: 3,
}


def compose_loop_guard_telemetry(
    stop_reason: str | GuardReason | None,
    tool_name: str | None,
    tool_rounds: int,
    *,
    repeat_count: int = 0,
    cycle_length: int | None = None,
    blocked_count: int = 0,
    detail: str | None = None,
) -> dict:
    telemetry = {
        "stop_reason": str(stop_reason) if stop_reason is not None else None,
        "tool_name": tool_name,
        "repeat_count": repeat_count,
        "cycle_length": cycle_length,
        "tool_rounds": tool_rounds,
        "blocked_count": blocked_count,
    }
    if detail is not None:
        telemetry["detail"] = detail
    return telemetry


class LoopGuard:
    def __init__(self, config: LoopGuardConfig):
        self.config = config
        self._states: dict[tuple[str, str], _SignatureState] = {}
        self._decisions: dict[str, GuardDecision] = {}
        self._calls: dict[str, tuple[str, dict | None, tuple[str, str]]] = {}
        self._recorded: set[str] = set()
        self._history: list[_HistoryEntry] = []
        self._history_index: dict[str, int] = {}
        self._stop_info: dict | None = None
        self._blocked_count = 0

    @property
    def stop_info(self) -> dict | None:
        return self._stop_info

    @property
    def blocked_count(self) -> int:
        return self._blocked_count

    def telemetry(self, tool_rounds: int) -> dict | None:
        if self._stop_info is None:
            return None
        return compose_loop_guard_telemetry(
            self._stop_info["stop_reason"],
            self._stop_info["tool_name"],
            tool_rounds,
            repeat_count=self._stop_info["repeat_count"],
            cycle_length=self._stop_info["cycle_length"],
            blocked_count=self._blocked_count,
            detail=self._stop_info["detail"],
        )

    def decide(
        self,
        tool_call_id: str,
        name: str,
        args: str | dict | list | None,
        round_index: int,
        over_limit: bool = False,
    ) -> GuardDecision:
        cached = self._decisions.get(tool_call_id)
        if cached is not None:
            return cached
        if over_limit:
            decision = self._make_decision(
                tool_call_id, name, GuardAction.BLOCK, GuardReason.ROUND_LIMIT
            )
            self._blocked_count += 1
            return self._remember(decision)
        if not self.config.enabled:
            return self._remember(
                self._make_decision(tool_call_id, name, GuardAction.EXECUTE, None)
            )
        if name in self.config.exempt_tools:
            return self._remember(
                self._make_decision(tool_call_id, name, GuardAction.EXECUTE, None)
            )
        key = canonical_tool_key(name, args)
        state = self._states.setdefault(key, _SignatureState())
        if name in self.config.polling_tools:
            if state.last_round == round_index:
                self._blocked_count += 1
                return self._remember(
                    self._make_decision(
                        tool_call_id,
                        name,
                        GuardAction.BLOCK,
                        GuardReason.DUPLICATE_IN_ROUND,
                    )
                )
            state.last_round = round_index
            return self._remember(
                self._make_decision(tool_call_id, name, GuardAction.EXECUTE, None)
            )
        if state.last_round == round_index:
            self._blocked_count += 1
            return self._remember(
                self._make_decision(
                    tool_call_id,
                    name,
                    GuardAction.BLOCK,
                    GuardReason.DUPLICATE_IN_ROUND,
                )
            )
        state.last_round = round_index
        state.attempts += 1
        occurrence = state.attempts
        is_read = name in self.config.read_only_tools
        detail = None
        cycle_length = None
        if state.last_failed:
            state.failed_blocks += 1
            stop_at = (
                self.config.failed_stop_occurrence
                - self.config.failed_block_occurrence
                + 1
            )
            action = (
                GuardAction.STOP
                if state.failed_blocks >= stop_at
                else GuardAction.BLOCK
            )
            reason = GuardReason.FAILED_REPEAT
            if state.last_error:
                detail = f"failed: {state.last_error}"
        elif not is_read:
            action = self._write_action(occurrence)
            reason = (
                GuardReason.WRITE_REPEAT if action is not GuardAction.EXECUTE else None
            )
            if action is GuardAction.STOP:
                detail = "write_repeat"
        else:
            action = self._read_action(occurrence)
            reason = (
                GuardReason.DUPLICATE if action is not GuardAction.EXECUTE else None
            )
            history_entry = _HistoryEntry(key, state.last_fingerprint)
            self._append_history(tool_call_id, history_entry)
            cycle = self._detect_cycle()
            if cycle is not None:
                cycle_length, repetitions = cycle
                cycle_action = self._cycle_action(repetitions)
                if _ACTION_RANK[cycle_action] > _ACTION_RANK[action]:
                    action = cycle_action
                if _ACTION_RANK[cycle_action] >= _ACTION_RANK[GuardAction.BLOCK]:
                    reason = GuardReason.CYCLE
        self._calls[tool_call_id] = (name, args, key)  # type: ignore[assignment]
        decision = self._make_decision(
            tool_call_id,
            name,
            action,
            reason,
            occurrence=occurrence,
            cycle_length=cycle_length,
            detail=detail,
        )
        if action is GuardAction.STOP:
            self._remember_stop(decision)
        elif action is GuardAction.BLOCK:
            self._blocked_count += 1
        return self._remember(decision)

    def record_result(
        self, tool_call_id: str, content: Any, attrs: dict | None = None
    ) -> bool:
        decision = self._decisions.get(tool_call_id)
        if decision is None or tool_call_id in self._recorded:
            return False
        self._recorded.add(tool_call_id)
        if decision.action in (GuardAction.BLOCK, GuardAction.STOP):
            return False
        call = self._calls.get(tool_call_id)
        if call is None:
            return False
        name, args, key = call
        if not self.config.enabled:
            return False
        if name in self.config.exempt_tools or name in self.config.polling_tools:
            return False
        state = self._states[key]
        if _is_failed(attrs):
            state.last_failed = True
            state.failed_blocks = 0
            state.last_error = _error_summary(content)
            self._remove_history(tool_call_id)
            return False
        state.last_failed = False
        state.failed_blocks = 0
        state.last_error = None
        try:
            fingerprint = result_fingerprint(name, args, content, attrs)
        except Exception:
            fingerprint = uuid.uuid4().hex
            state.last_fingerprint = fingerprint
            if name in self.config.read_only_tools:
                state.attempts = 0
                self._update_history(tool_call_id, fingerprint)
            return True
        previous = state.last_fingerprint
        state.last_fingerprint = fingerprint
        if name not in self.config.read_only_tools:
            return False
        changed = previous is not None and previous != fingerprint
        if changed:
            state.attempts = 0
        self._update_history(tool_call_id, fingerprint)
        return changed

    def _write_action(self, occurrence: int) -> GuardAction:
        if occurrence >= self.config.write_stop_occurrence:
            return GuardAction.STOP
        if occurrence >= self.config.write_block_occurrence:
            return GuardAction.BLOCK
        return GuardAction.EXECUTE

    def _read_action(self, occurrence: int) -> GuardAction:
        if occurrence >= self.config.read_stop_occurrence:
            return GuardAction.STOP
        if occurrence >= self.config.read_block_occurrence:
            return GuardAction.BLOCK
        if occurrence >= self.config.read_warn_occurrence:
            return GuardAction.WARN
        return GuardAction.EXECUTE

    def _cycle_action(self, repetitions: int) -> GuardAction:
        if repetitions >= self.config.cycle_stop_repetitions:
            return GuardAction.STOP
        if repetitions >= self.config.cycle_block_repetitions:
            return GuardAction.BLOCK
        if repetitions >= self.config.cycle_warn_repetitions:
            return GuardAction.WARN
        return GuardAction.EXECUTE

    def _make_decision(
        self,
        tool_call_id: str,
        name: str,
        action: GuardAction,
        reason: GuardReason | None,
        *,
        occurrence: int = 0,
        cycle_length: int | None = None,
        detail: str | None = None,
    ) -> GuardDecision:
        return GuardDecision(
            tool_call_id=tool_call_id,
            tool_name=name,
            action=action,
            reason=reason,
            occurrence=occurrence,
            repeat_count=max(occurrence - 2, 0) if occurrence else 0,
            cycle_length=cycle_length,
            detail=detail,
        )

    def _remember(self, decision: GuardDecision) -> GuardDecision:
        self._decisions[decision.tool_call_id] = decision
        return decision

    def _remember_stop(self, decision: GuardDecision) -> None:
        self._stop_info = {
            "stop_reason": (
                decision.reason.value if decision.reason is not None else None
            ),
            "tool_name": decision.tool_name,
            "repeat_count": decision.repeat_count,
            "cycle_length": decision.cycle_length,
            "detail": decision.detail,
        }

    def _append_history(self, tool_call_id: str, entry: _HistoryEntry) -> None:
        self._history.append(entry)
        self._history_index[tool_call_id] = len(self._history) - 1
        overflow = len(self._history) - self.config.history_limit
        for _ in range(max(overflow, 0)):
            self._history.pop(0)
        if overflow > 0:
            self._history_index = {
                call_id: index - overflow
                for call_id, index in self._history_index.items()
                if index >= overflow
            }

    def _update_history(self, tool_call_id: str, fingerprint: str) -> None:
        index = self._history_index.get(tool_call_id)
        if index is None or index >= len(self._history):
            return
        self._history[index].fingerprint = fingerprint

    def _remove_history(self, tool_call_id: str) -> None:
        index = self._history_index.pop(tool_call_id, None)
        if index is None or index >= len(self._history):
            return
        self._history.pop(index)
        for call_id, other_index in list(self._history_index.items()):
            if other_index > index:
                self._history_index[call_id] = other_index - 1

    def _detect_cycle(self) -> tuple[int, int] | None:
        history = self._history
        count = len(history)
        best: tuple[int, int] | None = None
        max_length = min(self.config.cycle_max_length, count)
        for length in range(self.config.cycle_min_length, max_length + 1):
            if len({entry.key for entry in history[-length:]}) < 2:
                continue
            matched = 0
            while matched + length < count:
                left = history[count - 1 - matched]
                right = history[count - 1 - matched - length]
                if not left.matches(right):
                    break
                matched += 1
            if matched == 0:
                continue
            span = matched + length
            complete, partial = divmod(span, length)
            repetitions = complete + (1 if partial else 0)
            if repetitions < 2:
                continue
            if best is None or (repetitions, -length) > (best[1], -best[0]):
                best = (length, repetitions)
        return best


def _is_failed(attrs: dict | None) -> bool:
    if not isinstance(attrs, dict):
        return False
    tool_call = attrs.get("tool_call")
    return isinstance(tool_call, dict) and tool_call.get("status") == "failed"


def _error_summary(content: Any) -> str | None:
    text = _content_text(content)
    if not text:
        return None
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    return text[:200] or None


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
            else:
                parts.append(str(part))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def result_fingerprint(
    tool_name: str,
    args: dict | None,
    content: Any,
    attrs: dict | None = None,
) -> str:
    value, customized = _tool_fingerprint_value(tool_name, args, content, attrs)
    if not isinstance(value, str):
        value = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
    normalized = strip_volatile(
        value, strip_timestamps=not customized, strip_uuids=not customized
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _tool_fingerprint_value(
    tool_name: str, args: dict | None, content: Any, attrs: dict | None
) -> tuple[Any, bool]:
    attrs = attrs or {}
    if tool_name == "read_resource":
        value = _resource_hash_operation(attrs, args)
        if value is not None:
            return value, True
    if tool_name == "list_resources":
        value = _list_resources_value(content)
        if value is not None:
            return value, True
    if tool_name == "read_root":
        value = _read_root_value(content)
        if value is not None:
            return value, True
    if tool_name == "describe_resource":
        value = _describe_resource_value(content)
        if value is not None:
            return value, True
    if tool_name in ("list_resource_attachments", "get_resource_attachment"):
        value = _attachments_value(content)
        if value is not None:
            return value, True
    if tool_name == "view_attachment":
        return _content_text(content), True
    if tool_name in ("resource_search", "web_search", "search", "private_search"):
        citations = _citation_values(
            attrs, include_updated_at=tool_name == "resource_search"
        )
        if citations is not None:
            return citations, True
    if tool_name == "resource_filter":
        value = _resource_filter_value(content)
        if value is not None:
            return value, True
    if tool_name in ("list_tags", "tag_filter"):
        value = _parse_json_value(content)
        if value is not None:
            return value, True
    if tool_name == "list_resource_comments":
        value = _comments_value(content)
        if value is not None:
            return value, True
    if tool_name in (
        "get_rss_folder_config",
        "get_smart_folder_config",
        "get_current_info",
        "load_docs",
        "list_local_devices",
    ):
        value = _parse_json_value(content)
        if value is not None:
            return value, True
        return _content_text(content), False
    if tool_name == "search_conversation_messages":
        value = _history_recall_value(content)
        if value is not None:
            return value, True
    if tool_name == "read_conversation_message":
        value = _conversation_message_value(content)
        if value is not None:
            return value, True
    return _generic_value(content), False


def _generic_value(content: Any) -> str:
    text = _content_text(content).strip()
    canonical = _canonical_json(text)
    return canonical if canonical is not None else text


def _canonical_json(text: str) -> str | None:
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return None
    return json.dumps(parsed, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _parse_json_value(content: Any) -> Any | None:
    text = _content_text(content).strip()
    canonical = _canonical_json(text)
    if canonical is None:
        return None
    return json.loads(canonical)


def _json_without_prefix(content: Any) -> Any | None:
    text = _content_text(content).strip()
    lines = text.splitlines()
    while lines and lines[0].startswith("Only "):
        lines.pop(0)
    if not lines:
        return None
    try:
        return json.loads("\n".join(lines))
    except (TypeError, ValueError):
        return None


def _resource_hash_operation(attrs: dict, args: dict | None) -> str | None:
    resource_id = (args or {}).get("resource_id")
    context = attrs.get("context")
    if not isinstance(context, dict):
        return None
    operations = context.get("operations")
    if not isinstance(operations, list):
        return None
    for operation in operations:
        if not isinstance(operation, dict):
            continue
        if (
            operation.get("namespace") == RESOURCE_HASH_NAMESPACE
            and operation.get("key") == resource_id
        ):
            value = operation.get("value")
            if value is not None:
                return str(value)
    return None


def _list_resources_value(content: Any) -> Any | None:
    parsed = _json_without_prefix(content)
    if not isinstance(parsed, dict):
        return None
    resources = parsed.get("resources")
    if not isinstance(resources, list):
        return None
    items = []
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        items.append(
            [
                resource.get("id"),
                resource.get("name"),
                resource.get("type") or resource.get("resource_type"),
                resource.get("has_children"),
            ]
        )
    items.sort(key=lambda item: json.dumps(item, ensure_ascii=False, default=str))
    return {"resources": items, "total": parsed.get("total")}


def _read_root_value(content: Any) -> Any | None:
    parsed = _parse_json_value(content)
    candidates: list[dict] = []
    if isinstance(parsed, dict):
        if isinstance(parsed.get("roots"), list):
            candidates = [root for root in parsed["roots"] if isinstance(root, dict)]
        else:
            candidates = [root for root in parsed.values() if isinstance(root, dict)]
    elif isinstance(parsed, list):
        candidates = [root for root in parsed if isinstance(root, dict)]
    if not candidates:
        return None
    items = [[root.get("id"), root.get("name")] for root in candidates]
    items.sort(key=lambda item: json.dumps(item, ensure_ascii=False, default=str))
    return items


def _describe_resource_value(content: Any) -> Any | None:
    parsed = _parse_json_value(content)
    if not isinstance(parsed, dict):
        return None
    return [
        parsed.get("id"),
        parsed.get("updated_at"),
        parsed.get("sub_resource_count"),
        parsed.get("content_line_count"),
        parsed.get("tags"),
    ]


def _attachments_value(content: Any) -> Any | None:
    parsed = _parse_json_value(content)
    attachments: Any = None
    if isinstance(parsed, dict):
        attachments = parsed.get("attachments")
        if attachments is None:
            attachment = parsed.get("attachment")
            if isinstance(attachment, dict):
                attachments = [attachment]
            elif "id" in parsed and "name" in parsed and "size" in parsed:
                attachments = [parsed]
    elif isinstance(parsed, list):
        attachments = parsed
    if not isinstance(attachments, list):
        return None
    items = []
    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue
        items.append(
            [
                attachment.get("id"),
                attachment.get("name"),
                attachment.get("size"),
            ]
        )
    items.sort(key=lambda item: json.dumps(item, ensure_ascii=False, default=str))
    return items


def _citation_values(attrs: dict, *, include_updated_at: bool) -> list | None:
    citations = attrs.get("citations")
    if not isinstance(citations, list):
        return None
    values = []
    for citation in citations:
        values.append(_citation_key(citation, include_updated_at))
    values.sort(key=lambda item: json.dumps(item, ensure_ascii=False, default=str))
    return values


def _citation_key(citation: Any, include_updated_at: bool) -> Any:
    if hasattr(citation, "model_dump"):
        citation = citation.model_dump(mode="json")
    if not isinstance(citation, dict):
        return str(citation)
    citation_id = citation.get("id")
    if isinstance(citation_id, str):
        citation_id = _CITATION_PREFIX_PATTERN.sub("", citation_id)
    key = [
        citation.get("link"),
        citation.get("title"),
        citation.get("snippet"),
        citation_id,
    ]
    if include_updated_at:
        key.append(citation.get("updated_at"))
    return key


def _resource_filter_value(content: Any) -> Any | None:
    parsed = _parse_json_value(content)
    if not isinstance(parsed, dict):
        return None
    resources = parsed.get("resources")
    if not isinstance(resources, list):
        return None
    items = []
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        items.append([resource.get("id"), resource.get("updated_at")])
    items.sort(key=lambda item: json.dumps(item, ensure_ascii=False, default=str))
    return items


def _comments_value(content: Any) -> Any | None:
    parsed = _parse_json_value(content)
    if not isinstance(parsed, dict):
        return None
    threads = parsed.get("threads")
    if not isinstance(threads, list):
        return None
    items = []
    for thread in threads:
        if not isinstance(thread, dict):
            continue
        comments = []
        for comment in thread.get("comments") or []:
            if isinstance(comment, dict):
                comments.append([comment.get("id"), comment.get("updated_at")])
        items.append([thread.get("id"), comments])
    items.sort(key=lambda item: json.dumps(item, ensure_ascii=False, default=str))
    return items


def _conversation_message_value(content: Any) -> Any | None:
    parsed = _parse_json_value(content)
    if not isinstance(parsed, dict):
        return None
    payload = parsed.get("message")
    if not isinstance(payload, dict):
        payload = parsed
    return {
        "id": payload.get("id") or payload.get("message_id"),
        "content": payload.get("content"),
    }


def _history_recall_value(content: Any) -> Any | None:
    text = _content_text(content)
    if "<history_recall>" not in text:
        return None
    hits = []
    for message_id, body in _HISTORY_MESSAGE_PATTERN.findall(text):
        digest = hashlib.sha256(body.strip().encode("utf-8")).hexdigest()
        hits.append([message_id, digest])
    if not hits:
        return None
    hits.sort(key=lambda item: json.dumps(item, ensure_ascii=False, default=str))
    return hits


def strip_volatile(
    text: str, *, strip_timestamps: bool = True, strip_uuids: bool = True
) -> str:
    canonical = _canonical_json(text.strip())
    if canonical is not None:
        text = canonical
    text = _URL_PATTERN.sub(_strip_url_volatile, text)
    if strip_timestamps:
        text = _ISO_TIMESTAMP_PATTERN.sub("<ts>", text)
    text = _CITATION_PREFIX_PATTERN.sub("", text)
    if strip_uuids:
        text = _UUID_PATTERN.sub("<id>", text)
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def _strip_url_volatile(match: re.Match[str]) -> str:
    url = match.group(0)
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return url
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def render_guard_user_text(decision: GuardDecision, lang: str | None = None) -> str:
    english = lang == "English"
    tool = decision.tool_name
    reason = decision.reason
    if reason is GuardReason.DUPLICATE_IN_ROUND:
        if english:
            return (
                f"Skipped the repeated `{tool}` call in this round: "
                "identical arguments in one round run only once."
            )
        return f"已跳过同一轮内再次出现的 `{tool}` 调用：相同参数在本轮只执行一次。"
    if reason is GuardReason.CYCLE:
        if english:
            return (
                f"Skipped this `{tool}` call: detected a cycle of "
                f"{decision.cycle_length} tool calls with unchanged results. "
                "Use the earlier result or a different approach."
            )
        return (
            f"已跳过这次 `{tool}` 调用：检测到 {decision.cycle_length} 个工具调用的"
            "循环且结果没有变化。请使用之前的结果或换一种方式。"
        )
    if reason is GuardReason.WRITE_REPEAT:
        if english:
            return (
                f"Skipped this `{tool}` call: the same write call already ran. "
                "Repeating it could duplicate the change. Use the earlier result "
                "or adjust the arguments."
            )
        return (
            f"已跳过这次 `{tool}` 调用：相同参数的写操作已经执行过，重复执行可能"
            "造成重复修改。请使用之前的结果或调整参数。"
        )
    if reason is GuardReason.FAILED_REPEAT:
        if english:
            return (
                f"Skipped this `{tool}` call: the previous call with the same "
                "arguments failed. Retrying it unchanged will not succeed. Fix "
                "the arguments or use another tool."
            )
        return (
            f"已跳过这次 `{tool}` 调用：相同参数的上一次调用已失败，直接重试不会"
            "成功。请修正参数或换用其他工具。"
        )
    if reason is GuardReason.ROUND_LIMIT:
        if english:
            return (
                f"Skipped this `{tool}` call: the tool round limit for this "
                "turn has been reached."
            )
        return f"已跳过这次 `{tool}` 调用：本轮工具调用次数已达到上限。"
    executed = max(decision.occurrence - 1, 0)
    if english:
        return (
            f"Skipped `{tool}`: the same arguments already ran {executed} "
            "time(s) with an unchanged result. Use the earlier result or a "
            "different approach."
        )
    return (
        f"已跳过这次 `{tool}` 调用：相同参数已执行 {executed} 次且结果没有变化，"
        "继续重复不会带来新信息。请使用之前的结果或换一种方式。"
    )


def render_guard_model_note(decision: GuardDecision) -> str:
    tool = decision.tool_name
    reason = decision.reason
    lines = [
        f"The call to `{tool}` was NOT executed. Reason: "
        f"{reason.value if reason is not None else decision.action.value}. "
        "Do not call it again with the same arguments; use the earlier result "
        "or another tool."
    ]
    if reason in (GuardReason.DUPLICATE, GuardReason.CYCLE):
        lines.append(
            f"This is occurrence {decision.occurrence} with the same arguments "
            "and an unchanged result."
        )
    elif reason is GuardReason.DUPLICATE_IN_ROUND:
        lines.append(
            "This call already appeared earlier in the current round; identical "
            "calls in one round run only once."
        )
    elif reason is GuardReason.WRITE_REPEAT:
        lines.append(
            "A write call with these arguments already ran; repeating it could "
            "duplicate the change."
        )
    elif reason is GuardReason.FAILED_REPEAT:
        lines.append(
            "The previous call with these arguments failed; retrying it "
            "unchanged will not help."
        )
    elif reason is GuardReason.ROUND_LIMIT:
        lines.append("The tool round limit was reached. Answer the user in text.")
    if decision.action is GuardAction.STOP:
        lines.append(
            "Tool use is disabled for the rest of this turn. Produce a final "
            "answer in text now."
        )
    return " ".join(lines)


def render_guard_warn_note(decision: GuardDecision) -> str:
    repeat = max(decision.occurrence - 1, 1)
    return (
        f"Note: this is repeat {repeat} of the same call with an unchanged "
        "result. If the result is still what you need, proceed without calling "
        "it again."
    )


_REASON_DESCRIPTIONS = {
    GuardReason.DUPLICATE: (
        "repeated read calls with identical arguments and unchanged results"
    ),
    GuardReason.CYCLE: "a cycle of read calls with unchanged results",
    GuardReason.WRITE_REPEAT: "the same write call was already executed",
    GuardReason.FAILED_REPEAT: "the same call previously failed",
    GuardReason.ROUND_LIMIT: "the tool round limit was reached",
    "invalid_resource_edit": "the resource edit failed validation",
}


def render_loop_stop_instruction(
    reason: str | GuardReason | None = None,
) -> str:
    key: Any = reason.value if isinstance(reason, GuardReason) else reason
    description = _REASON_DESCRIPTIONS.get(key, "no further tool calls are allowed")
    return (
        f"Tool use is disabled for the rest of this turn because {description}. "
        "Reply now in text: summarize what has been completed, what remains "
        "unfinished, and why you stopped. Do not request any more tools."
    )
