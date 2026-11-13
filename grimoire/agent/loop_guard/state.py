import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from wizard_common.grimoire.agent.loop_guard.config import LoopGuardConfig
from wizard_common.grimoire.agent.loop_guard.fingerprints import (
    content_text,
    result_fingerprint,
)
from wizard_common.grimoire.agent.loop_guard.telemetry import (
    compose_loop_guard_telemetry,
)
from wizard_common.grimoire.agent.tool_loop import canonical_tool_key

STATE_VERSION = 1


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


class LoopGuard:
    def __init__(self, config: LoopGuardConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
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

    def to_state(self) -> dict:
        return {
            "version": STATE_VERSION,
            "signatures": [
                {
                    "name": key[0],
                    "key": key[1],
                    "attempts": state.attempts,
                    "last_fingerprint": state.last_fingerprint,
                    "last_failed": state.last_failed,
                    "failed_blocks": state.failed_blocks,
                    "last_error": state.last_error,
                }
                for key, state in self._states.items()
            ],
            "history": [
                {
                    "name": entry.key[0],
                    "key": entry.key[1],
                    "fingerprint": entry.fingerprint,
                }
                for entry in self._history
            ],
            "stop_info": dict(self._stop_info) if self._stop_info is not None else None,
            "blocked_count": self._blocked_count,
        }

    def load_state(self, data: Any) -> None:
        self.reset()
        if not isinstance(data, dict) or data.get("version") != STATE_VERSION:
            return
        signatures = data.get("signatures")
        if isinstance(signatures, list):
            for item in signatures:
                key = _signature_key(item)
                if key is None:
                    continue
                state = _SignatureState()
                state.attempts = _as_count(item.get("attempts"))
                fingerprint = item.get("last_fingerprint")
                state.last_fingerprint = (
                    fingerprint if isinstance(fingerprint, str) else None
                )
                state.last_failed = item.get("last_failed") is True
                state.failed_blocks = _as_count(item.get("failed_blocks"))
                error = item.get("last_error")
                state.last_error = error if isinstance(error, str) else None
                self._states[key] = state
        history = data.get("history")
        if isinstance(history, list):
            for item in history:
                key = _signature_key(item)
                if key is None:
                    continue
                fingerprint = item.get("fingerprint")
                self._history.append(
                    _HistoryEntry(
                        key, fingerprint if isinstance(fingerprint, str) else None
                    )
                )
            overflow = len(self._history) - self.config.history_limit
            if overflow > 0:
                del self._history[:overflow]
        stop_info = data.get("stop_info")
        if isinstance(stop_info, dict):
            cycle_length = stop_info.get("cycle_length")
            detail = stop_info.get("detail")
            self._stop_info = {
                "stop_reason": stop_info.get("stop_reason"),
                "tool_name": stop_info.get("tool_name"),
                "repeat_count": _as_count(stop_info.get("repeat_count")),
                "cycle_length": (
                    cycle_length if isinstance(cycle_length, int) else None
                ),
                "detail": detail if isinstance(detail, str) else None,
            }
        self._blocked_count = _as_count(data.get("blocked_count"))

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
        is_read = name in self.config.read_only_tools
        try:
            fingerprint = result_fingerprint(name, args, content, attrs)
        except Exception:
            fingerprint = uuid.uuid4().hex
            state.last_fingerprint = fingerprint
            if is_read:
                state.attempts = 0
                self._update_history(tool_call_id, fingerprint)
            else:
                self._reset_read_progress()
            return True
        previous = state.last_fingerprint
        state.last_fingerprint = fingerprint
        if not is_read:
            self._reset_read_progress()
            return False
        changed = previous is not None and previous != fingerprint
        if changed:
            state.attempts = 0
        self._update_history(tool_call_id, fingerprint)
        return changed

    def _reset_read_progress(self) -> None:
        for key, state in self._states.items():
            if key[0] in self.config.read_only_tools:
                state.attempts = 0
                state.last_fingerprint = None
        self._history.clear()
        self._history_index.clear()

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


def _signature_key(item: Any) -> tuple[str, str] | None:
    if not isinstance(item, dict):
        return None
    name = item.get("name")
    key = item.get("key")
    if not isinstance(name, str) or not isinstance(key, str):
        return None
    return name, key


def _as_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(value, 0)


def _is_failed(attrs: dict | None) -> bool:
    if not isinstance(attrs, dict):
        return False
    tool_call = attrs.get("tool_call")
    return isinstance(tool_call, dict) and tool_call.get("status") == "failed"


def _error_summary(content: Any) -> str | None:
    text = content_text(content)
    if not text:
        return None
    text = " ".join(text.split())
    return text[:200] or None
