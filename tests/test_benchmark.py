import copy
import json

import pytest

from idea.benchmark import compare_manifest, main, order_swap_report


def manifest():
    runs = []
    for seed in (13, 29):
        for strategy, score in (("evolution", .8), ("independent_generation", .5),
                                ("iterative_refinement", .6)):
            runs.append({"strategy": strategy, "seed": seed, "template_id": "research",
                "export": {"config": {"random_seed": seed, "idea_type": "research"},
                           "population": [{"id": "early", "idea": "before budget"},
                                          {"id": "late", "idea": "after budget"}],
                           "token_counts": {"cost": {"total_cost": 1.2}}},
                "measurements": [{"candidate_id": "early", "cumulative_cost_usd": .8,
                                  "holdout_score": score, "feasible": True},
                                 {"candidate_id": "late", "cumulative_cost_usd": 1.1,
                                  "holdout_score": 1, "feasible": True}]})
    return {"schema_version": 1, "spend_basis": "generation_selection_artifacts",
            "evaluator": {"id": "blind-human-rubric", "version": "1", "holdout": True,
                          "direction": "maximize", "score_bounds": [0, 1]}, "runs": runs}


def test_equal_budget_excludes_late_champion_and_pairs_seeds():
    result = compare_manifest(manifest(), budget_usd=1)
    assert all(run["best_candidate_id"] == "early" and run["eligible_candidates"] == 1
               for run in result["runs"])
    comparisons = result["comparisons"]
    assert comparisons[0]["mean_paired_improvement"] == pytest.approx(.3)
    assert comparisons[1]["mean_paired_improvement"] == pytest.approx(.2)
    assert comparisons[0]["paired_seeds"] == 2
    assert comparisons[0]["paired_standard_error"] == 0


def test_minimize_direction_and_missing_feasible_outcomes_are_explicit():
    data = manifest()
    data["evaluator"]["direction"] = "minimize"
    data["runs"][0]["measurements"][0]["feasible"] = False
    result = compare_manifest(data, budget_usd=1)
    assert result["comparisons"][0]["mean_paired_improvement"] == pytest.approx(-.3)
    assert result["comparisons"][0]["unscorable_seeds"] == [13]
    assert result["comparisons"][0]["paired_standard_error"] is None


@pytest.mark.parametrize("change,match", [
    (lambda data: data["evaluator"].pop("holdout"), "evaluator"),
    (lambda data: data["runs"].pop(), "all three strategies"),
    (lambda data: data["runs"][0]["measurements"].pop(), "every exported"),
    (lambda data: data["runs"][0]["measurements"][0].update(candidate_id="invented"), "candidate IDs"),
    (lambda data: data["runs"][0]["export"]["token_counts"]["cost"].update(total_cost=.9), "spend ceiling"),
    (lambda data: data["runs"][0]["measurements"][0].pop("holdout_score"), "holdout_score"),
    (lambda data: data["runs"][0]["measurements"][0].update(holdout_score=float("nan")), "finite"),
])
def test_unfair_or_unverifiable_comparisons_fail(change, match):
    data = manifest()
    change(data)
    with pytest.raises(ValueError, match=match):
        compare_manifest(data, budget_usd=1)


def test_exported_fitness_cannot_substitute_for_holdout_evidence():
    data = manifest()
    data["runs"][0]["export"]["population"][0]["fitness"] = 999
    before = compare_manifest(copy.deepcopy(data), budget_usd=1)
    data["runs"][0]["export"]["population"][0]["fitness"] = -999
    assert compare_manifest(data, budget_usd=1) == before


def test_swapped_winners_are_compared_as_candidate_identities():
    result = order_swap_report([
        {"pair_id": "consistent", "candidate_a": "a", "candidate_b": "b",
         "winner_ab": "A", "winner_ba": "B"},
        {"pair_id": "position-sensitive", "candidate_a": "c", "candidate_b": "d",
         "winner_ab": "A", "winner_ba": "A"},
        {"pair_id": "neutral", "candidate_a": "e", "candidate_b": "f",
         "winner_ab": "tie", "winner_ba": "tie"}])
    assert result["agreement_rate"] == pytest.approx(2/3)
    assert result["first_position_win_fraction"] == .75


def test_cli_reads_relative_exports_and_writes_reproducible_json(tmp_path):
    data = manifest()
    for index, run in enumerate(data["runs"]):
        path = tmp_path / f"run-{index}.json"
        path.write_text(json.dumps(run.pop("export")))
        run["export_path"] = path.name
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(data))
    output = tmp_path / "report.json"
    assert main([str(manifest_path), "--budget", "1", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["evaluator"]["id"] == "blind-human-rubric"
    assert report["budget_ceiling_usd"] == 1
    assert len(report["runs"]) == 6
