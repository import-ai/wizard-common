def compose_loop_guard_telemetry(
    stop_reason: str | None,
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
