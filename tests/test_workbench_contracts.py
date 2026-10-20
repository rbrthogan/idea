import json
import subprocess
from pathlib import Path

from tests.test_evolution_tree_ui_contract import _build_rank_rows
from tests.test_fitness_plot_ui_contract import _process_fitness_plot


ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "idea/static/js/workbench.js"


def evaluate_shared(expression):
    script = """
global.window = { matchMedia: () => ({ matches: false, addEventListener() {} }) };
global.document = { addEventListener() {} };
""" + SHARED.read_text() + f"\nprocess.stdout.write(JSON.stringify({expression}));"
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def test_evaluator_evidence_preserves_zero_and_missing_or_failed_status():
    evidence = evaluate_shared("""window.IdeaWorkbench.metricEvidence({
        evaluations: {
            good: { status: 'cached', metrics: { score: 0 } },
            missing: { status: 'ok', metrics: { score: null } },
            failed: { status: 'failed', metrics: { score: 99 }, error: 'timeout' }
        }
    }, { fitness: { metrics: [
        { component_id: 'good', metric_name: 'score', weight: 0 },
        { component_id: 'missing', metric_name: 'score', weight: 1 },
        { component_id: 'failed', metric_name: 'score', weight: 2 }
    ] } })""")
    assert evidence[0]["value"] == 0
    assert evidence[0]["weight"] == 0
    assert evidence[0]["status"] == "cached"
    assert evidence[1]["value"] is None
    assert evidence[2]["value"] is None
    assert evidence[2]["status"] == "failed"
    assert evidence[2]["reason"] == "timeout"


def test_ranked_view_sorts_custom_evaluator_metrics_without_promoting_failure():
    rows = _build_rank_rows([[
        {"id": "bad", "title": "Failed", "evaluations": {"clarity": {"status": "failed", "metrics": {"score": 100}}}},
        {"id": "zero", "title": "Zero", "evaluations": {"clarity": {"status": "ok", "metrics": {"score": 0}}}},
        {"id": "best", "title": "Best", "evaluations": {"clarity": {"status": "cached", "metrics": {"score": 0.8}}}},
    ]], {"sortMetric": "eval:clarity.score"})
    assert [row["id"] for row in rows] == ["best", "zero", "bad"]
    assert rows[0]["metricPopulationSize"] == 2


def test_both_pages_use_the_shared_responsive_inspector_and_pinned_chart_bundle():
    for name in ("viewer", "rater"):
        source = (ROOT / f"idea/static/html/{name}.html").read_text()
        assert '/static/js/workbench.js' in source
        assert '/static/css/workbench.css' in source
        assert 'class="candidate-inspector" id="ideaModal"' in source
        assert 'data-inspector-close' in source
        assert source.index('id="sidebarToggle"') < source.index('id="controlsPanel"')
        assert 'chart.js@4.5.1/dist/chart.umd.min.js' in source
        assert 'window.innerWidth <= 576' not in source


def test_main_template_selection_is_a_native_button_separate_from_edit():
    source = (ROOT / "idea/static/html/viewer.html").read_text()
    assert '<button type="button" class="template-library-select"' in source
    assert "row.setAttribute('role', 'button')" not in source
    assert "selectButton.addEventListener('click'" in source
    assert 'aria-label="Search templates"' in source


def test_failed_start_keeps_the_displayed_experiment_until_success():
    source = (ROOT / "idea/static/js/viewer.js").read_text()
    start = source.index("addListener('startButton'")
    end = source.index("addListener('stopButton'", start)
    handler = source[start:end]
    assert handler.index('resetUIState();') > handler.index('if (response.ok)')


def test_budget_and_pending_cancellation_copy_matches_the_runtime_contract():
    html = (ROOT / "idea/static/html/viewer.html").read_text()
    source = (ROOT / "idea/static/js/viewer.js").read_text()
    assert "Stops new work at this amount. Requests already running may finish." in html
    assert "cancellationPending = Boolean(data.cancel_pending || data.is_running)" in source
    assert "if (stopButton && cancellationConfirmed && !cancellationPending)" in source


def test_fitness_chart_can_show_saved_scores_without_claiming_the_legacy_formula():
    source = (ROOT / "idea/static/js/rater.js").read_text()
    assert "Mean of saved candidate fitness." in source
    assert "const displayedFitnessSeries = usesSavedFitness ? savedFitnessSeries : combinedFitnessSeries" in source
    assert ".filter((_, index) => !usesSavedFitness || index === 0)" in source


def test_sampled_history_notice_reports_unique_revisions_without_calling_current_metrics_sampled():
    result = evaluate_shared("""(() => {
        const note = {};
        const text = window.IdeaWorkbench.renderHistorySamplingNote(note, [
            { metadata: { embedding_scope: 'historical_sample', history_sampling: {
                sample_size: 256, total_size: 1024, total_history_slots: 2048, truncated: true
            } } }
        ]);
        const visible = !note.hidden;
        const exactCurrent = window.IdeaWorkbench.historySamplingNotice({
            embedding_scope: 'current_generation', history_sampling: {
                sample_size: 256, total_size: 1024, truncated: true
            }
        });
        window.IdeaWorkbench.renderHistorySamplingNote(note, [{ history_sampling: {
            sample_size: 10, total_size: 10, total_history_slots: 20, truncated: false
        } }]);
        return { text, visible, exactCurrent, hiddenWhenComplete: note.hidden };
    })()""")
    assert result["visible"] is True
    assert "Historical diversity is sampled" in result["text"]
    assert "256 of 1,024 unique revisions" in result["text"]
    assert "2,048 recorded history slots" in result["text"]
    assert "Current population and island metrics remain exact" in result["text"]
    assert result["exactCurrent"] == ""
    assert result["hiddenWhenComplete"] is True


def test_history_sampling_metadata_reaches_the_plot_without_changing_exact_population_values():
    sample = {"sample_size": 256, "total_size": 1024, "total_history_slots": 2048, "truncated": True}
    processed = _process_fitness_plot(
        diversity_history=[{"generation": 0, "diversity_score": 0.9, "history_sampling": sample}],
        generation_history=[[
            {"selection_metrics": {"diversity": 0.2}},
            {"selection_metrics": {"diversity": 0.4}},
        ]],
    )
    assert processed["historySampling"] == sample
    assert round(processed["metrics"]["diversity"]["average"][0], 8) == 0.3
