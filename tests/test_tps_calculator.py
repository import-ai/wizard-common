from types import SimpleNamespace

import pytest
from wizard_common.grimoire.agent import tps_calculator as metrics_module
from wizard_common.grimoire.agent.tps_calculator import TPSCalculator


@pytest.mark.asyncio
async def test_rate_uses_the_recent_window_but_total_remains_cumulative(monkeypatch):
    now = 10.0
    monkeypatch.setattr(metrics_module, "time", SimpleNamespace(time=lambda: now))
    calculator = TPSCalculator()
    await calculator.count()
    now = 12.0
    await calculator.count()
    assert await calculator.tps() == 1.0
    assert calculator.total == 2

    now = 23.0
    assert await calculator.tps() == 0.0
    assert calculator.total == 2


@pytest.mark.asyncio
async def test_chunks_with_identical_timestamps_do_not_break_the_stream(monkeypatch):
    monkeypatch.setattr(metrics_module, "time", SimpleNamespace(time=lambda: 10.0))
    calculator = TPSCalculator()
    await calculator.count()
    await calculator.count()
    assert await calculator.tps() == 0.0
    assert calculator.total == 2
