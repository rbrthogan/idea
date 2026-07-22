import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parent.parent
VIEWER_JS = ROOT / "idea" / "static" / "js" / "viewer.js"
VIEWER_HTML = ROOT / "idea" / "static" / "html" / "viewer.html"


def _extract_function_bundle(script: str, first_function: str, last_function: str) -> str:
    async_marker = f"async function {first_function}"
    function_marker = f"function {first_function}"
    async_start = script.find(async_marker)
    function_start = script.find(function_marker)
    if async_start >= 0 and (function_start < 0 or async_start < function_start):
        start = async_start
    else:
        start = script.index(function_marker)
    end_marker = f"function {last_function}"
    if end_marker not in script[start:]:
        end_marker = last_function
    end = script.index(end_marker, start)
    return script[start:end]


def _process_fitness_plot(diversity_history, generation_history, island_diversity_history=None):
    script = VIEWER_JS.read_text()
    bundle = _extract_function_bundle(script, "toFiniteDiversityNumber", "getDiversityDatasetStyle")
    node_script = f"""
{bundle}
const diversityHistory = {json.dumps(diversity_history)};
const generationHistory = {json.dumps(generation_history)};
const islandDiversityHistory = {json.dumps(island_diversity_history or [])};
const processed = processDiversityData(diversityHistory, generationHistory, islandDiversityHistory);
process.stdout.write(JSON.stringify({{
  labels: processed.labels,
  metrics: processed.metrics,
  latest: processed.latest,
  islandDiversity: processed.islandDiversity
}}));
"""
    result = subprocess.run(
        ["node", "-e", node_script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def _call_handle_diversity_update(progress_data, rendered_generations):
    script = VIEWER_JS.read_text()
    bundle = _extract_function_bundle(script, "handleDiversityUpdate", "document.addEventListener")
    node_script = f"""
let currentEvolutionData = null;
let generations = {json.dumps(rendered_generations)};
let captured = null;
function updateDiversityChart(diversityHistory, generationHistory, islandDiversityHistory) {{
  captured = {{ diversityHistory, generationHistory, islandDiversityHistory }};
  return true;
}}
{bundle}
handleDiversityUpdate({json.dumps(progress_data)});
process.stdout.write(JSON.stringify(captured));
"""
    result = subprocess.run(
        ["node", "-e", node_script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def _restore_completed_evolution_from_local_storage(stored_state, progress_response):
    script = VIEWER_JS.read_text()
    bundle = _extract_function_bundle(script, "restoreCurrentEvolution", "function setupDownloadButton")
    node_script = f"""
let currentEvolutionData = null;
let contexts = [];
let specificPrompts = [];
let breedingPrompts = [];
let currentContextIndex = 0;
let lastRenderedHistoryVersion = -1;
let lastStoredHistoryVersion = -1;
let tournamentHistory = [];
let renderedHistory = null;
let chartArgs = null;
let pollingStarted = false;
global.fetch = async () => ({{
  ok: true,
  json: async () => ({json.dumps(progress_response)})
}});
global.localStorage = {{
  getItem: () => JSON.stringify({json.dumps(stored_state)}),
  removeItem: () => {{}}
}};
global.document = {{
  getElementById: () => ({{ style: {{}}, disabled: false }}),
  querySelector: () => ({{ style: {{}} }})
}};
global.setTimeout = (fn) => fn();
function renderGenerations(history) {{ renderedHistory = history; }}
function updateDiversityChart(diversityHistory, generationHistory, islandDiversityHistory) {{
  chartArgs = {{ diversityHistory, generationHistory, islandDiversityHistory }};
  return true;
}}
function ensureDiversityChartSizing() {{}}
function updateContextDisplay() {{}}
function renderTournamentDetails() {{}}
function displayTokenCounts() {{}}
function setupDownloadButton() {{}}
function startPollingActiveEvolution() {{ pollingStarted = true; }}
{bundle}
restoreCurrentEvolution().then((restored) => {{
  process.stdout.write(JSON.stringify({{
    restored,
    renderedHistory,
    chartArgs,
    pollingStarted,
    currentEvolutionData
  }}));
}});
"""
    result = subprocess.run(
        ["node", "-e", node_script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_fitness_plot_processes_population_metrics_for_toggle_modes():
    processed = _process_fitness_plot(
        diversity_history=[
            {"generation": 0, "diversity_score": 0.25},
            {"generation": 1, "diversity_score": 0.75},
        ],
        generation_history=[
            [
                {"selection_metrics": {"fitness": 0.2, "elo": 1400, "diversity": 0.1}},
                {"selection_metrics": {"fitness": 0.6, "elo": 1600, "diversity": 0.5}},
            ],
            [
                {"selection_metrics": {"fitness": 0.4, "elo": 1500, "diversity": 0.8}},
                {"fitness": 0.8, "elo": 1700, "diversity_score": 0.2},
            ],
        ],
    )

    assert processed["labels"] == ["0", "1"]
    assert processed["metrics"]["fitness"]["average"] == pytest.approx([0.4, 0.6])
    assert processed["metrics"]["fitness"]["best"] == pytest.approx([0.6, 0.8])
    assert processed["metrics"]["fitness"]["low"] == pytest.approx([0.2, 0.4])
    assert processed["metrics"]["fitness"]["spread"] == pytest.approx([0.4, 0.4])

    assert processed["metrics"]["elo"]["average"] == pytest.approx([1500, 1600])
    assert processed["metrics"]["elo"]["best"] == pytest.approx([1600, 1700])
    assert processed["metrics"]["elo"]["low"] == pytest.approx([1400, 1500])

    assert processed["metrics"]["diversity"]["average"] == pytest.approx([0.3, 0.5])
    assert processed["metrics"]["diversity"]["best"] == pytest.approx([0.5, 0.8])
    assert processed["metrics"]["diversity"]["low"] == pytest.approx([0.1, 0.2])
    assert processed["latest"]["fitness"]["average"] == pytest.approx(0.6)
    assert processed["latest"]["elo"]["best"] == pytest.approx(1700)
    assert processed["latest"]["diversity"]["count"] == 2


def test_fitness_plot_ignores_stale_or_unevaluated_metric_zeros():
    processed = _process_fitness_plot(
        diversity_history=[
            {"generation": 0, "diversity_score": 0.25},
            {"generation": 1, "diversity_score": 0.75},
        ],
        generation_history=[
            [
                {"selection_metrics": {"evaluation_generation": 0, "fitness": 0.3, "elo": 1400, "diversity": 0.2}},
                {"selection_metrics": {"evaluation_generation": 0, "fitness": 0.7, "elo": 1600, "diversity": 0.6}},
            ],
            [
                {
                    "selection_metrics": {"evaluation_generation": 0, "fitness": 0.0, "elo": 0, "diversity": 0.0},
                    "fitness": 0.0,
                    "elo": 0,
                    "diversity_score": 0.0,
                },
                {
                    "selection_metrics": {"evaluation_generation": 0, "fitness": 0.6, "elo": 1550, "diversity": 0.5},
                    "fitness": 0.6,
                    "elo": 1550,
                    "diversity_score": 0.5,
                },
            ],
        ],
    )

    assert processed["labels"] == ["0", "1"]
    assert processed["metrics"]["fitness"]["low"][0] == pytest.approx(0.3)
    assert processed["metrics"]["fitness"]["low"][1] is None
    assert processed["metrics"]["elo"]["low"][0] == pytest.approx(1400)
    assert processed["metrics"]["elo"]["low"][1] is None
    assert processed["metrics"]["diversity"]["low"][0] == pytest.approx(0.2)
    assert processed["metrics"]["diversity"]["low"][1] is None
    assert processed["latest"]["fitness"]["count"] == 0
    assert processed["latest"]["elo"]["count"] == 0
    assert processed["latest"]["diversity"]["count"] == 0


def test_fitness_plot_prefers_generation_metric_snapshots():
    processed = _process_fitness_plot(
        diversity_history=[
            {"generation": 0, "diversity_score": 0.25},
            {"generation": 1, "diversity_score": 0.75},
        ],
        generation_history=[
            [
                {
                    "selection_metrics": {"evaluation_generation": 1, "fitness": 0.9, "elo": 1900, "diversity": 0.9},
                    "generation_metrics": {
                        "0": {"evaluation_generation": 0, "fitness": 0.3, "elo": 1400, "diversity": 0.2},
                    },
                },
                {
                    "selection_metrics": {"evaluation_generation": 1, "fitness": 0.8, "elo": 1800, "diversity": 0.8},
                    "generation_metrics": {
                        "0": {"evaluation_generation": 0, "fitness": 0.7, "elo": 1600, "diversity": 0.6},
                    },
                },
            ],
            [
                {
                    "selection_metrics": {"evaluation_generation": 1, "fitness": 0.4, "elo": 1500, "diversity": 0.3},
                    "generation_metrics": {
                        "1": {"evaluation_generation": 1, "fitness": 0.45, "elo": 1510, "diversity": 0.35},
                    },
                },
                {
                    "selection_metrics": {"evaluation_generation": 1, "fitness": 0.8, "elo": 1700, "diversity": 0.5},
                    "generation_metrics": {
                        "1": {"evaluation_generation": 1, "fitness": 0.85, "elo": 1710, "diversity": 0.55},
                    },
                },
            ],
        ],
    )

    assert processed["metrics"]["fitness"]["average"] == pytest.approx([0.5, 0.65])
    assert processed["metrics"]["elo"]["average"] == pytest.approx([1500, 1610])
    assert processed["metrics"]["diversity"]["average"] == pytest.approx([0.4, 0.45])


def test_live_diversity_update_reuses_last_rendered_history_when_progress_omits_history():
    rendered_generations = [
        [{"generation_metrics": {"0": {"fitness": 0.2, "elo": 1400, "diversity": 0.1}}}],
        [{"generation_metrics": {"1": {"fitness": 0.6, "elo": 1600, "diversity": 0.5}}}],
    ]

    captured = _call_handle_diversity_update(
        {
            "diversity_history": [{"generation": 1, "diversity_score": 0.5}],
            "island_diversity_history": [{"generation": 1, "current_islands": []}],
        },
        rendered_generations,
    )

    assert captured["generationHistory"] == rendered_generations
    assert captured["islandDiversityHistory"] == [{"generation": 1, "current_islands": []}]


def test_completed_evolution_restore_uses_local_storage_without_active_run():
    stored_state = {
        "history": [[{"generation_metrics": {"0": {"fitness": 0.5, "elo": 1500, "diversity": 0.2}}}]],
        "history_version": 4,
        "diversity_history": [{"generation": 0, "diversity_score": 0.2}],
        "island_diversity_history": [{"generation": 0, "current_islands": []}],
        "tournament_history": [],
    }

    restored = _restore_completed_evolution_from_local_storage(
        stored_state,
        {"is_running": False, "history_available": False},
    )

    assert restored["restored"] is True
    assert restored["renderedHistory"] == stored_state["history"]
    assert restored["chartArgs"]["generationHistory"] == stored_state["history"]
    assert restored["chartArgs"]["islandDiversityHistory"] == stored_state["island_diversity_history"]
    assert restored["pollingStarted"] is False


def test_fitness_plot_preserves_island_diversity_series():
    processed = _process_fitness_plot(
        diversity_history=[
            {"generation": 0, "diversity_score": 0.25},
            {"generation": 1, "diversity_score": 0.75},
        ],
        generation_history=[
            [{"generation_metrics": {"0": {"fitness": 0.2, "elo": 1400, "diversity": 0.1}}}],
            [{"generation_metrics": {"1": {"fitness": 0.6, "elo": 1600, "diversity": 0.5}}}],
        ],
        island_diversity_history=[
            {
                "generation": 0,
                "mean_island_diversity": 0.2,
                "inter_island_diversity": 0.5,
                "island_separation_ratio": 2.5,
                "current_islands": [
                    {"island_id": 0, "diversity_score": 0.1},
                    {"island_id": 1, "diversity_score": 0.3},
                ],
            },
            {
                "generation": 1,
                "mean_island_diversity": 0.4,
                "inter_island_diversity": 0.8,
                "island_separation_ratio": 2.0,
                "current_islands": [
                    {"island_id": 0, "diversity_score": 0.2},
                    {"island_id": 1, "diversity_score": 0.6},
                ],
            },
        ],
    )

    assert processed["islandDiversity"]["mean"] == pytest.approx([0.2, 0.4])
    assert processed["islandDiversity"]["inter"] == pytest.approx([0.5, 0.8])
    assert processed["islandDiversity"]["separation"] == pytest.approx([2.5, 2.0])
    assert processed["islandDiversity"]["islandSeries"]["0"] == pytest.approx([0.1, 0.2])
    assert processed["islandDiversity"]["islandSeries"]["1"] == pytest.approx([0.3, 0.6])


def test_fitness_plot_html_exposes_metric_toggle():
    html = VIEWER_HTML.read_text()

    assert "Fitness" in html
    assert 'data-fitness-metric="fitness"' in html
    assert 'data-fitness-metric="elo"' in html
    assert 'data-fitness-metric="diversity"' in html
    assert 'data-fitness-scope="population"' in html
    assert 'data-fitness-scope="islands"' in html
