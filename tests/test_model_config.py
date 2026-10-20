import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.config import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODELS,
    THINKING_MODEL_CONFIG,
    LLM_MODELS,
    LEGACY_LLM_MODELS,
    DEFAULT_MODEL,
    embedding_prices_per_million_tokens,
    estimate_generation_token_cost,
    get_model_provider,
    model_prices_per_million_tokens,
    resolve_model_token_prices,
)
from idea.diversity import DiversityCalculator
from idea.viewer import _get_default_thinking_settings, _normalize_thinking_level


def test_gemini_31_flash_lite_is_available_with_pricing_and_thinking_defaults():
    model_id = "gemini-3.1-flash-lite-preview"
    model_ids = [model["id"] for model in LLM_MODELS]

    assert model_id not in model_ids
    assert model_id in {model["id"] for model in LEGACY_LLM_MODELS}
    assert model_prices_per_million_tokens[model_id]["input"] == 0.25
    assert model_prices_per_million_tokens[model_id]["output"] == 1.50
    assert model_prices_per_million_tokens[model_id]["cached_input"] == 0.025

    thinking_config = THINKING_MODEL_CONFIG[model_id]
    assert thinking_config["supports_thinking"] is True
    assert thinking_config["supports_thinking_level"] is True
    assert thinking_config["default_level"] == "low"
    assert thinking_config["allow_off"] is False
    assert _get_default_thinking_settings(model_id) == ("low", None)


def test_gpt_56_luna_is_provider_aware_with_cut_pricing_and_reasoning_efforts():
    model_id = "gpt-5.6-luna"
    model_ids = [model["id"] for model in LLM_MODELS]

    assert model_id not in model_ids
    assert model_id in {model["id"] for model in LEGACY_LLM_MODELS}
    assert get_model_provider(model_id) == "openai"
    assert model_prices_per_million_tokens[model_id]["input"] == 0.20
    assert model_prices_per_million_tokens[model_id]["cached_input"] == 0.02
    assert model_prices_per_million_tokens[model_id]["output"] == 1.20

    thinking_config = THINKING_MODEL_CONFIG[model_id]
    assert thinking_config["supports_thinking_budget"] is False
    assert thinking_config["allowed_levels"] == [
        "none",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ]
    assert _get_default_thinking_settings(model_id) == ("medium", None)


def test_gpt_56_long_context_pricing_applies_to_the_full_request():
    cost = estimate_generation_token_cost(
        "gpt-5.6-luna",
        input_tokens=300_000,
        cached_input_tokens=100_000,
        cache_write_input_tokens=50_000,
        output_tokens=1_000,
        thought_tokens=2_000,
    )

    assert cost["large_prompt_tier"] is True
    assert cost["input_rate"] == 0.40
    assert cost["cached_input_rate"] == 0.04
    assert cost["cache_write_input_rate"] == 0.50
    assert cost["output_rate"] == 1.80
    assert cost["total_cost"] == (
        150_000 * 0.40 / 1_000_000
        + 100_000 * 0.04 / 1_000_000
        + 50_000 * 0.50 / 1_000_000
        + 1_000 * 1.80 / 1_000_000
        + 2_000 * 1.80 / 1_000_000
    )


def test_gemini_31_pro_uses_current_level_defaults():
    model_id = "gemini-3.1-pro-preview"
    thinking_config = THINKING_MODEL_CONFIG[model_id]

    assert thinking_config["supports_thinking"] is True
    assert thinking_config["supports_thinking_level"] is True
    assert thinking_config["supports_thinking_budget"] is False
    assert thinking_config["allow_off"] is False
    assert thinking_config["allowed_levels"] == ["low", "medium", "high"]
    assert thinking_config["default_level"] == "low"
    assert _get_default_thinking_settings(model_id) == ("low", None)


def test_gemini_36_flash_is_available_with_current_pricing_and_thinking_defaults():
    model_id = "gemini-3.6-flash"
    model_ids = [model["id"] for model in LLM_MODELS]

    assert model_id in model_ids
    prices = resolve_model_token_prices(model_id, priced_at="2026-10-13")
    assert prices["input"] == 0.75
    assert prices["output"] == 3.75
    assert prices["cached_input"] == 0.075

    thinking_config = THINKING_MODEL_CONFIG[model_id]
    assert thinking_config["supports_thinking"] is True
    assert thinking_config["supports_thinking_level"] is True
    assert thinking_config["supports_thinking_budget"] is False
    assert thinking_config["supports_sampling_parameters"] is False
    assert thinking_config["allowed_levels"] == ["minimal", "low", "medium", "high"]
    assert thinking_config["default_level"] == "medium"
    assert _get_default_thinking_settings(model_id) == ("medium", None)


def test_pro_pricing_tips_to_large_prompt_regime_and_prices_thinking_as_output():
    cost = estimate_generation_token_cost(
        "gemini-2.5-pro",
        input_tokens=210_000,
        output_tokens=1_000,
        thought_tokens=2_000,
        cached_input_tokens=50_000,
    )

    assert cost["uncached_input_tokens"] == 160_000
    assert cost["cached_input_tokens"] == 50_000
    assert cost["input_rate"] == 2.50
    assert cost["cached_input_rate"] == 0.25
    assert cost["output_rate"] == 15.00
    assert cost["large_prompt_tier"] is True
    assert cost["thought_cost"] == 2_000 * 15.00 / 1_000_000
    assert cost["total_cost"] == (
        160_000 * 2.50 / 1_000_000
        + 50_000 * 0.25 / 1_000_000
        + 1_000 * 15.00 / 1_000_000
        + 2_000 * 15.00 / 1_000_000
    )


def test_gemini_31_pro_pricing_tips_to_large_prompt_regime():
    cost = estimate_generation_token_cost(
        "gemini-3.1-pro-preview",
        input_tokens=210_000,
        output_tokens=1_000,
        thought_tokens=2_000,
        cached_input_tokens=50_000,
    )

    assert cost["input_rate"] == 4.00
    assert cost["cached_input_rate"] == 0.40
    assert cost["output_rate"] == 18.00
    assert cost["large_prompt_tier"] is True
    assert cost["total_cost"] == (
        160_000 * 4.00 / 1_000_000
        + 50_000 * 0.40 / 1_000_000
        + 1_000 * 18.00 / 1_000_000
        + 2_000 * 18.00 / 1_000_000
    )


def test_gemini_3_models_default_to_low_thinking_level():
    assert _normalize_thinking_level("minimal") == "minimal"
    assert _get_default_thinking_settings("gemini-3-flash-preview") == (
        "low",
        None,
    )
    assert _get_default_thinking_settings("gemini-3.1-flash-lite-preview") == (
        "low",
        None,
    )
    assert _get_default_thinking_settings("gemini-3.1-pro-preview") == (
        "low",
        None,
    )
    assert _get_default_thinking_settings("gemini-3.6-flash") == (
        "medium",
        None,
    )


def test_current_catalog_removes_retired_ids_and_keeps_legacy_routing():
    ids = {row["id"] for row in LLM_MODELS}
    assert {"gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-4-argon",
            "gpt-6-luna", "gpt-6.1-sol", "gpt-6-astra"} <= ids
    assert not {"gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-3.1-flash-lite-preview"} & ids
    assert DEFAULT_MODEL == "gemini-3.5-flash-lite"
    assert get_model_provider("models/gpt-5.6-luna") == "openai"
    assert get_model_provider("gpt-6.1-sol") == "openai"


@pytest.mark.parametrize("model", ["gemini-3.8-flash", "gemini-4-argon", "gpt-6.1-sol", "gpt-6-astra"])
def test_current_reasoning_models_reject_off_and_omit_sampling(model):
    config = THINKING_MODEL_CONFIG[model]
    assert config["allow_off"] is False
    assert "none" not in config["allowed_levels"]
    assert "minimal" not in config["allowed_levels"]
    assert config["supports_sampling_parameters"] is False


@pytest.mark.parametrize("model,expiry,before,after", [
    ("gemini-4-argon", "2026-12-01", (2.0, 0.1, 10.0), (4.0, 0.2, 20.0)),
    ("gemini-3.8-flash", "2027-01-01", (0.75, 0.075, 3.75), (1.5, 0.15, 7.5)),
    ("gemini-3.6-flash", "2027-01-01", (0.75, 0.075, 3.75), (1.5, 0.15, 7.5)),
])
def test_published_promotional_expiry_changes_new_request_prices(model, expiry, before, after):
    old = resolve_model_token_prices(model, priced_at="2026-10-13")
    new = resolve_model_token_prices(model, priced_at=expiry)
    assert tuple(old[k] for k in ("input", "cached_input", "output")) == before
    assert tuple(new[k] for k in ("input", "cached_input", "output")) == after
    assert new["price_effective_date"] == expiry


def test_gpt_61_cache_writes_replace_normal_input_and_long_context_applies_to_full_request():
    cost = estimate_generation_token_cost("gpt-6.1-sol", input_tokens=300_000,
        cached_input_tokens=100_000, cache_write_input_tokens=50_000,
        output_tokens=1_000, thought_tokens=2_000, priced_at="2026-10-13")
    assert cost["uncached_input_tokens"] == 150_000
    assert cost["large_prompt_tier"] is True
    assert cost["total_cost"] == pytest.approx(.6 + .02 + .25 + .015 + .03)
    assert cost["pricing_known"] is True
    assert cost["price_source"].startswith("https://developers.openai.com/")
    assert cost["price_version"] == "2026-10-13"


def test_unknown_prices_are_flagged_instead_of_using_an_arbitrary_cheap_model():
    cost = estimate_generation_token_cost("unlisted-model", 1000, 500)
    assert cost["pricing_known"] is False
    assert cost["price_source"] is None
    assert cost["total_cost"] == 0


def test_blank_legacy_model_settings_use_current_default():
    from idea.evolution_types import EvolutionRunConfig
    for value in (None, "", "  "):
        assert EvolutionRunConfig(model_type=value).model_type == DEFAULT_MODEL


def test_new_legacy_sol_requests_use_current_promotional_price_without_inventing_expiry():
    for day in ("2026-10-13", "2026-11-22"):
        p = resolve_model_token_prices("gpt-5.6-sol", priced_at=day)
        assert (p["input"], p["cached_input"], p["output"]) == (4, .4, 20)


def test_invalid_token_counts_are_not_allowed_to_crash_price_resolution():
    p = estimate_generation_token_cost("gpt-6-luna", input_tokens=float("inf"), output_tokens=float("nan"))
    assert p["input_tokens"] == p["output_tokens"] == 0


def test_gemini_embedding_2_is_default_and_one_embedding_space_is_pinned():
    assert DEFAULT_EMBEDDING_MODEL == "gemini-embedding-2"
    assert EMBEDDING_MODELS[0]["id"] == "gemini-embedding-2"
    assert embedding_prices_per_million_tokens["gemini-embedding-2"]["text_input"] == 0.20
    assert embedding_prices_per_million_tokens["gemini-embedding-001"]["input"] == 0.15

    calculator = DiversityCalculator(api_key=None)

    assert calculator.embedding_model == "gemini-embedding-2"
    assert calculator._embedding_models == ["gemini-embedding-2"]
    assert calculator._embedding_dimensions is None
