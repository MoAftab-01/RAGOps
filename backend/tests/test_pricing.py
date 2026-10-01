"""Cost calculation and the honesty of the price labels.

§35 of the specification forbids fabricating cost savings, and the cheapest way
to honour that is to make it impossible for the pricing layer to claim a
number it cannot source. These tests pin down that behaviour: a local model
costs exactly zero, an unknown model costs zero *and says so*, and a priced
model costs precisely what the arithmetic says.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.utils.pricing import (
    calculate_cost,
    get_pricing,
    is_simulated,
    pricing_label,
    resolve_window,
)


class TestGetPricing:
    def test_known_model_is_found(self) -> None:
        entry = get_pricing("gpt-4o-mini")
        assert entry is not None
        assert entry.provider

    def test_unknown_model_is_none_not_a_default_price(self) -> None:
        # Returning a "default" entry here would silently invent a cost for
        # every model nobody thought to list.
        assert get_pricing("some-model-nobody-priced") is None


class TestCalculateCost:
    def test_local_model_costs_exactly_zero(self) -> None:
        # Ollama is running locally, so there is no API bill. Charging for it
        # would be the single most misleading thing this platform could do.
        assert calculate_cost(1_000_000, 1_000_000, "qwen2.5:3b") == 0.0

    def test_unknown_model_costs_zero_rather_than_guessing(self) -> None:
        assert calculate_cost(5_000, 5_000, "unpriced-model") == 0.0

    def test_priced_model_matches_hand_computed_arithmetic(self) -> None:
        entry = get_pricing("gpt-4o-mini")
        assert entry is not None
        input_tokens, output_tokens = 1500, 400
        expected = round(
            (input_tokens / 1000.0) * entry.input_cost_per_1k
            + (output_tokens / 1000.0) * entry.output_cost_per_1k,
            8,
        )
        cost = calculate_cost(input_tokens, output_tokens, "gpt-4o-mini")
        assert cost == expected
        assert cost > 0

    def test_cost_scales_linearly_with_tokens(self) -> None:
        one = calculate_cost(1000, 1000, "gpt-4o-mini")
        ten = calculate_cost(10000, 10000, "gpt-4o-mini")
        assert ten == pytest.approx(one * 10, rel=1e-6)

    def test_zero_tokens_costs_zero(self) -> None:
        assert calculate_cost(0, 0, "gpt-4o-mini") == 0.0

    def test_explicit_pricing_overrides_the_table(self) -> None:
        from app.utils.pricing import ModelPricing

        custom = ModelPricing(
            model_name="custom",
            provider="custom",
            input_cost_per_1k=1.0,
            output_cost_per_1k=2.0,
            context_window=8192,
            is_local=False,
            description="",
        )
        # 2000 in * 1.0/1k + 1000 out * 2.0/1k = 2.0 + 2.0
        assert calculate_cost(2000, 1000, "custom", pricing=custom) == 4.0

    def test_disabled_pricing_returns_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from app.config import settings

        monkeypatch.setattr(settings, "pricing_enabled", False)
        assert calculate_cost(1_000_000, 1_000_000, "gpt-4o-mini") == 0.0
        # And the label has to say so, or the UI shows "$0.00" with no reason.
        assert pricing_label("gpt-4o-mini") == "Pricing disabled"


class TestPriceProvenance:
    """The labels are what stop a simulated price reading as a real bill."""

    def test_local_model_is_labelled_local(self) -> None:
        assert pricing_label("qwen2.5:3b") == "Local inference — no API cost"
        assert is_simulated("qwen2.5:3b") is False

    def test_priced_model_is_labelled_simulated(self) -> None:
        # RAGOps never calls these APIs, so their price is a what-if figure and
        # must be presented as one everywhere it appears.
        assert pricing_label("gpt-4o-mini") == "Simulated list price"
        assert is_simulated("gpt-4o-mini") is True

    def test_unknown_model_is_labelled_unpriced(self) -> None:
        assert pricing_label("nobody-priced-this") == "Unpriced model"
        assert is_simulated("nobody-priced-this") is False

    def test_every_priced_entry_declares_itself_simulated(self) -> None:
        from app.utils.pricing import DEFAULT_PRICING

        nonlocal_entries = [p for p in DEFAULT_PRICING if not p.is_local]
        assert nonlocal_entries, "expected at least one reference price"
        for entry in nonlocal_entries:
            assert "SIMULATED" in (entry.description or ""), (
                f"{entry.model_name} is priced but its description does not say "
                f"the price is simulated."
            )


class TestResolveWindow:
    def test_known_window_maps_to_the_right_span(self) -> None:
        end = datetime(2026, 1, 1, tzinfo=timezone.utc)
        start, got_end, label = resolve_window("24h", end=end)
        assert got_end == end
        assert label == "24h"
        assert got_end - start == timedelta(hours=24)

    @pytest.mark.parametrize(
        ("window", "seconds"),
        [
            ("1h", 3600),
            ("24h", 86400),
            ("7d", 604800),
            ("30d", 2592000),
            ("90d", 7776000),
        ],
    )
    def test_every_advertised_window(self, window: str, seconds: int) -> None:
        end = datetime(2026, 6, 1, tzinfo=timezone.utc)
        start, _, label = resolve_window(window, end=end)
        assert label == window
        assert (end - start).total_seconds() == seconds

    def test_explicit_range_overrides_the_window_selector(self) -> None:
        # A custom range must win, or a caller who supplies both silently
        # analyses the wrong period.
        start_given = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end_given = datetime(2026, 1, 2, tzinfo=timezone.utc)
        start, end, label = resolve_window("90d", start=start_given, end=end_given)
        assert (start, end, label) == (start_given, end_given, "custom")

    def test_unknown_window_falls_back_to_seven_days(self) -> None:
        end = datetime(2026, 6, 1, tzinfo=timezone.utc)
        start, _, label = resolve_window("banana", end=end)
        assert (end - start) == timedelta(days=7)
        assert label == "banana", "echoes back what was asked, not what was used"

    def test_none_defaults_to_seven_days(self) -> None:
        end = datetime(2026, 6, 1, tzinfo=timezone.utc)
        start, _, label = resolve_window(None, end=end)
        assert label == "7d"
        assert (end - start) == timedelta(days=7)
