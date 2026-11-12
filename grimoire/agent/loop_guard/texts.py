from importlib.resources import files
from typing import Any

from common.template_parser import TemplateParser

from wizard_common.grimoire.agent.loop_guard.state import (
    GuardAction,
    GuardDecision,
    GuardReason,
)


def _render(template_name: str, **kwargs: Any) -> str:
    parser = TemplateParser(
        base_dir=str(files("wizard_common") / "resources" / "prompt_templates")
    )
    template = parser.get_template(f"loop_guard/{template_name}")
    return parser.render_template(template, **kwargs)


def _reason_value(reason: GuardReason | None) -> str:
    return reason.value if reason is not None else GuardReason.DUPLICATE.value


def render_guard_user_text(decision: GuardDecision, lang: str | None = None) -> str:
    template = "guard_user_en.j2" if lang == "English" else "guard_user_zh.j2"
    return _render(
        template,
        reason=_reason_value(decision.reason),
        tool=decision.tool_name,
        executed=max(decision.occurrence - 1, 0),
        cycle_length=decision.cycle_length,
    )


def render_guard_model_note(decision: GuardDecision) -> str:
    reason = decision.reason
    return _render(
        "guard_model_note.j2",
        tool=decision.tool_name,
        reason=reason.value if reason is not None else decision.action.value,
        occurrence=decision.occurrence,
        duplicate=reason in (GuardReason.DUPLICATE, GuardReason.CYCLE),
        duplicate_in_round=reason is GuardReason.DUPLICATE_IN_ROUND,
        write_repeat=reason is GuardReason.WRITE_REPEAT,
        failed_repeat=reason is GuardReason.FAILED_REPEAT,
        round_limit=reason is GuardReason.ROUND_LIMIT,
        stop=decision.action is GuardAction.STOP,
    )


def render_guard_warn_note(decision: GuardDecision) -> str:
    return _render(
        "guard_warn_note.j2",
        repeat=max(decision.occurrence - 1, 1),
    )


def render_loop_stop_instruction(
    reason: str | GuardReason | None = None,
) -> str:
    key = reason.value if isinstance(reason, GuardReason) else reason
    return _render("loop_stop_instruction.j2", reason=key)
