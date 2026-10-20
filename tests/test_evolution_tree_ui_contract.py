from pathlib import Path
import json
import subprocess


ROOT = Path(__file__).parent.parent
VIEWER_JS = ROOT / "idea" / "static" / "js" / "viewer.js"
VIEWER_CSS = ROOT / "idea" / "static" / "css" / "viewer.css"
VIEWER_HTML = ROOT / "idea" / "static" / "html" / "viewer.html"


def _extract_between(script: str, start_marker: str, end_marker: str) -> str:
    start = script.index(start_marker)
    end = script.index(end_marker, start)
    return script[start:end]


def _build_rank_rows(history, options=None):
    script = VIEWER_JS.read_text()
    helper_bundle = _extract_between(script, "function getIdeaData", "function getTreeParentIds")
    metric_bundle = _extract_between(script, "function getTreeMetricValue", "function getTreeFitnessRange")
    rank_bundle = _extract_between(script, "function compareRankMetricDesc", "function renderEvolutionRankedList")
    node_script = f"""
global.window = {{ CSS: {{ escape: (value) => String(value) }} }};
{helper_bundle}
{metric_bundle}
{rank_bundle}
const rows = buildEvolutionRankRows({json.dumps(history)}, {json.dumps(options or {})});
process.stdout.write(JSON.stringify(rows.map(row => ({{
  rank: row.rank,
  id: row.id,
  title: row.title,
  generationIndex: row.generationIndex,
  firstGeneration: row.firstGeneration,
  lastGeneration: row.lastGeneration,
  generationSpan: row.generationSpan,
  ideaIndex: row.ideaIndex,
  fitness: row.fitness,
  elo: row.elo,
  diversity: row.diversity,
  islandId: row.islandId,
  sortMetric: row.sortMetric,
  sortValue: row.sortValue,
  metricRank: row.metricRank,
  metricPopulationSize: row.metricPopulationSize
}}))));
"""
    result = subprocess.run(
        ["node", "-e", node_script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_tree_marks_historical_rows_as_ghosts_and_current_rows_as_solid():
    viewer_js = VIEWER_JS.read_text()
    viewer_css = VIEWER_CSS.read_text()

    assert "latestGeneration" in viewer_js
    assert "tree-historical" in viewer_js
    assert "tree-current-population" in viewer_js
    assert "button.dataset.generation" in viewer_js
    assert "--tree-historical-fitness-strength" in viewer_js
    assert ".evolution-tree-node.tree-historical" in viewer_css
    assert ".evolution-tree-node.tree-current-population" in viewer_css
    assert "var(--tree-historical-fitness-strength)" in viewer_css


def test_tree_renders_survivor_progression_as_dotted_trails_without_labels():
    viewer_js = VIEWER_JS.read_text()
    viewer_css = VIEWER_CSS.read_text()

    assert "TREE_CONNECTOR_BOUNDARY_GAP" in viewer_js
    assert "tree-connector-continuation" in viewer_js
    assert "tree-survivor-step" in viewer_js
    assert ".tree-connector-continuation" in viewer_css
    assert "stroke-linecap: butt" in viewer_css
    assert "survived unchanged" not in viewer_js.lower()


def test_tree_uses_s_curve_paths_for_hereditary_connectors():
    viewer_js = VIEWER_JS.read_text()

    assert "function assignTreeConnectorTracks" in viewer_js
    assert "function buildTreeSCurvePath" in viewer_js
    assert " S " in viewer_js
    assert "assignTreeConnectorTracks(edgeRoutes)" in viewer_js
    assert "buildTreeSCurvePath(route.from, route.to, route.track)" in viewer_js


def test_ranked_view_is_registered_as_third_evolution_view():
    viewer_js = VIEWER_JS.read_text()
    viewer_css = VIEWER_CSS.read_text()
    viewer_html = VIEWER_HTML.read_text()

    assert 'data-view-mode="ranked"' in viewer_html
    assert 'id="evolution-ranked-container"' in viewer_html
    assert "function buildEvolutionRankRows" in viewer_js
    assert "function renderEvolutionRankedList" in viewer_js
    assert 'data-rank-sort="fitness"' in viewer_js
    assert 'data-rank-sort="elo"' in viewer_js
    assert 'data-rank-sort="diversity"' in viewer_js
    assert 'data-rank-scope="history"' in viewer_js
    assert 'data-rank-scope="current"' in viewer_js
    assert "evolutionViewMode === 'ranked'" in viewer_js
    assert ".evolution-ranked-container" in viewer_css
    assert ".evolution-rank-row" in viewer_css
    assert ".evolution-rank-island" in viewer_css
    assert ".evolution-rank-metric" in viewer_css


def test_ranked_view_defaults_to_global_history_sorted_by_fitness_then_elo():
    rows = _build_rank_rows(
        [
            [
                {"id": "seed", "title": "Seed", "selection_metrics": {"fitness": 0.9, "elo": 1400, "diversity": 0.2}},
            ],
            [
                {"id": "low", "title": "Low", "island_id": 0, "selection_metrics": {"fitness": 0.5, "elo": 1800, "diversity": 0.4}},
                {"id": "high", "title": "High", "island_id": 1, "selection_metrics": {"fitness": 0.8, "elo": 1500, "diversity": 0.3}},
                {"id": "tie", "title": "Tie", "island_id": 2, "selection_metrics": {"fitness": 0.5, "elo": 1900, "diversity": 0.5}},
                {"id": "missing", "title": "Missing", "island_id": 3},
            ],
        ]
    )

    assert [row["id"] for row in rows] == ["seed", "high", "tie", "low", "missing"]
    assert [row["rank"] for row in rows] == [1, 2, 3, 4, 5]
    assert rows[0]["firstGeneration"] == 0
    assert rows[0]["lastGeneration"] == 0
    assert rows[0]["generationSpan"] == "Gen 0"
    assert rows[0]["sortMetric"] == "fitness"
    assert rows[0]["sortValue"] == 0.9
    assert rows[0]["metricRank"] == 1
    assert rows[0]["metricPopulationSize"] == 4
    assert rows[2]["elo"] == 1900
    assert rows[4]["fitness"] is None


def test_ranked_view_can_sort_by_elo_and_filter_current_population():
    history = [
        [
            {"id": "old", "title": "Old", "selection_metrics": {"fitness": 0.9, "elo": 1300, "diversity": 0.7}},
            {"id": "survivor", "title": "Survivor", "selection_metrics": {"fitness": 0.4, "elo": 1700, "diversity": 0.3}},
        ],
        [
            {"id": "survivor", "title": "Survivor", "selection_metrics": {"fitness": 0.45, "elo": 1750, "diversity": 0.32}},
            {"id": "elo-winner", "title": "ELO Winner", "selection_metrics": {"fitness": 0.55, "elo": 2100, "diversity": 0.2}},
            {"id": "fitness-winner", "title": "Fitness Winner", "selection_metrics": {"fitness": 0.8, "elo": 1500, "diversity": 0.5}},
        ],
    ]

    elo_rows = _build_rank_rows(history, {"sortMetric": "elo"})
    assert [row["id"] for row in elo_rows[:3]] == ["elo-winner", "survivor", "fitness-winner"]
    assert elo_rows[1]["generationSpan"] == "Gen 0-1"
    assert elo_rows[1]["metricRank"] == 2

    current_rows = _build_rank_rows(history, {"scope": "current", "sortMetric": "fitness"})
    assert [row["id"] for row in current_rows] == ["fitness-winner", "elo-winner", "survivor"]
    assert all(row["lastGeneration"] == 1 for row in current_rows)
