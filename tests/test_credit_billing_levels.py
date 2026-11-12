import json
from types import SimpleNamespace

import pytest

from wizard_common.grimoire.thinking import billing_headers, get_thinking_models


@pytest.mark.parametrize(
    "level,expected",
    [
        ("low", {"input": 95, "input_cached": 28, "output": 335}),
        ("medium", {"input": 95, "input_cached": 28, "output": 335}),
        ("high", {"input": 1450, "input_cached": 190, "output": 4350}),
        ("max", {"input": 1450, "input_cached": 190, "output": 4350}),
    ],
)
def test_billing_uses_selected_level_price_for_same_model(monkeypatch, level, expected):
    low = {"input": 95, "input_cached": 28, "output": 335}
    high = {"input": 1450, "input_cached": 190, "output": 4350}
    monkeypatch.setenv(
        "OBW_MODELS",
        json.dumps(
            {
                "pro": {
                    "default_level": "low",
                    "levels": [
                        {
                            "id": name,
                            "model": "shared-model",
                            "price": low if name in ("low", "medium") else high,
                        }
                        for name in ("low", "medium", "high", "max")
                    ],
                },
                "prices": {
                    "shared-model": {"input": 1, "input_cached": 1, "output": 1}
                },
            }
        ),
    )
    get_thinking_models.cache_clear()
    try:
        headers = billing_headers(SimpleNamespace(edition="pro", level=level), "pro")
        assert json.loads(headers["X-Omnibox-Billing"])["price"] == expected
        assert "price" not in str(get_thinking_models().public_config())
    finally:
        get_thinking_models.cache_clear()
