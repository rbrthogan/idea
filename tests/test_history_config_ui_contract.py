from pathlib import Path


ROOT = Path(__file__).parent.parent
VIEWER_HTML = ROOT / "idea" / "static" / "html" / "viewer.html"
VIEWER_JS = ROOT / "idea" / "static" / "js" / "viewer.js"


def test_advanced_config_exposes_age_decay_floor_control():
    html = VIEWER_HTML.read_text()
    script = VIEWER_JS.read_text()

    assert 'id="ageDecayFloor"' in html
    assert 'id="ageDecayFloorValue"' in html
    assert "ageDecayFloor" in script
    assert "age_decay_floor" in script


def test_history_panel_can_show_and_reuse_full_saved_config():
    script = VIEWER_JS.read_text()

    assert "buildEvolutionConfigDetailsHtml" in script
    assert "applyEvolutionConfigToForm" in script
    assert "handleUseHistoryConfig" in script
    assert "Use Config" in script
    for label in (
        "Creative Temperature",
        "Top P",
        "Thinking",
        "Tournament Count",
        "Replacement Rate",
        "Fitness Alpha",
        "Age Decay Floor",
        "Survivor Refinement",
        "Island Count",
        "Migration Novelty Protection",
    ):
        assert label in script


def test_thinking_low_and_custom_budget_options_have_distinct_containers():
    html = VIEWER_HTML.read_text()

    assert 'id="thinkingLowOption"' in html
    assert 'id="thinkingHighOption"' in html
    assert 'id="thinkingCustomBudgetOption"' in html
    assert html.index('id="thinkingLowOption"') < html.index('id="thinkingCustomBudgetOption"')
    assert html.index('id="thinkingCustomBudgetOption"') < html.index('id="thinkingCustomBudget"')
