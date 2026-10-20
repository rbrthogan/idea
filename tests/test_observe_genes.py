"""Saved-data observation is isolated from evolution and cannot overwrite its input."""
import copy
import json

import pytest

from idea.llm import GenerationCancelledError
from scripts.observe_genes import candidate_revision, observe_export, observe_history, select_candidates, spread_revisions


class FakeBreeder:
    provider = "google_generative_ai"
    model_name = "offline-observer"

    def __init__(self):
        self.total_cost = 0.0
        self.calls = []

    def observe_genes(self, candidate, idea_type):
        if self.cancel_check():
            raise GenerationCancelledError()
        self.calls.append(("extract", str(candidate["id"])))
        self.total_cost += 0.01
        texts = ["shared relationship", f"action {candidate['id']}"]
        candidate["gene_observation"] = {
            "version": 1, "status": "ok", "fingerprint": self._genotype_prompt_and_fingerprint(candidate, idea_type)[1],
            "genotype": "; ".join(texts), "mapping_status": "pending",
            "genes": [{"text": text, "gene_id": None} for text in texts],
        }

    def _genotype_prompt_and_fingerprint(self, candidate, idea_type):
        document = candidate.get("idea", candidate)
        content = document.content if hasattr(document, "content") else document["content"]
        return "", f"{self.model_name}:{idea_type}:{content}"

    def generate_text(self, prompt, response_schema=None):
        if self.cancel_check():
            raise GenerationCancelledError()
        self.calls.append(("match", None))
        self.total_cost += 0.01
        terms = json.loads(prompt.split("Input genes: ", 1)[1])
        return json.dumps({"groups": [
            {"existing_gene_id": None, "input_indices": [index]}
            for index in range(len(terms))
        ]})

    def get_token_usage_summary(self):
        return {"total": 10 * len(self.calls), "input": 7 * len(self.calls),
                "output": 3 * len(self.calls), "cost": self.total_cost, "requests": len(self.calls)}


def source_export():
    return {
        "evolution_id": "original-run", "name": "Existing designs", "current_generation": 3,
        "config": {"idea_type": "game_design", "model_type": "source-model", "fitness_alpha": 0.7},
        "history": [[
            {"id": "a", "title": "First", "content": "A real design", "elo": 1200},
            {"id": "a", "title": "First", "content": "A real design", "elo": 1200},
            {"id": "b", "title": "Second", "content": "Another real design", "elo": 1300},
        ]],
    }


def test_select_reads_flattened_history_and_deduplicates_without_changing_source():
    export = source_export()
    before = copy.deepcopy(export)
    selected, generation = select_candidates(export, 4)
    assert len(selected) == 2
    assert generation == 0
    assert selected[0]["idea"].content == "A real design"
    assert selected[0]["elo"] == 1200
    assert export == before


def test_select_prefers_current_population_and_accepts_explicit_generation():
    export = source_export()
    export["population"] = [{"id": "current", "idea": {"title": "Now", "content": "Current design"}}]
    current, generation = select_candidates(export, 4)
    assert current[0]["id"] == "current" and generation == 3
    historical, generation = select_candidates(export, 1, 0)
    assert historical[0]["id"] == "a" and generation == 0
    with pytest.raises(ValueError, match="zero-based"):
        select_candidates(export, 4, 20)


def test_annotation_export_preserves_source_and_accounts_only_observation_cost(tmp_path):
    export = source_export()
    before = copy.deepcopy(export)
    input_path, output_path = tmp_path / "input.json", tmp_path / "output.json"
    input_path.write_text(json.dumps(export))
    selected, generation = select_candidates(export, 4)
    breeder = FakeBreeder()
    result = observe_export(export, input_path=input_path, output_path=output_path,
                            candidates=selected, original_generation=generation, breeder=breeder,
                            template_data={"name": "source snapshot"}, max_budget=0.25)
    assert json.loads(input_path.read_text()) == before == export
    assert result == json.loads(output_path.read_text())
    assert result["observation_only"] is True
    assert result["status"] == "complete"
    assert result["population"] == result["history"][0]
    assert result["population"][0]["elo"] == 1200
    assert result["template_data"]["name"] == "source snapshot"
    assert result["config"]["fitness_alpha"] == 0.7
    assert result["token_counts"]["cost"]["total_cost"] == pytest.approx(0.03)
    shared = next(gene for gene in result["gene_catalogue"]["genes"] if gene["label"] == "shared relationship")
    assert shared["current_candidate_count"] == 2
    assert shared["current_candidate_ids"] == ["a", "b"]


@pytest.mark.parametrize("same_input", [False, True])
def test_existing_output_is_rejected_before_provider_calls(tmp_path, same_input):
    export = source_export()
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(export))
    output_path = input_path if same_input else tmp_path / "output.json"
    if not same_input:
        output_path.write_text("keep this")
    original = output_path.read_text()
    breeder = FakeBreeder()
    selected, generation = select_candidates(export, 4)
    with pytest.raises((FileExistsError, ValueError)):
        observe_export(export, input_path=input_path, output_path=output_path,
                       candidates=selected, original_generation=generation, breeder=breeder,
                       template_data={}, max_budget=0.25)
    assert output_path.read_text() == original
    assert breeder.calls == []


@pytest.mark.parametrize("budget,extraction_count", [(0, 0), (0.015, 2)])
def test_budget_stops_new_calls_and_keeps_completed_raw_observations(tmp_path, budget, extraction_count):
    export = source_export()
    selected, generation = select_candidates(export, 4)
    breeder = FakeBreeder()
    result = observe_export(export, input_path=tmp_path / "source.json", output_path=tmp_path / "annotated.json",
                            candidates=selected, original_generation=generation, breeder=breeder,
                            template_data={}, max_budget=budget)
    assert result["status"] == "paused"
    assert breeder.calls == [("extract", identity) for identity in ["a", "b"][:extraction_count]]
    assert sum(candidate.get("gene_observation", {}).get("status") == "ok"
               for candidate in result["population"]) == extraction_count
    assert result["gene_observation_run"]["budget_reached"] is True


def history_export():
    first = {"id": "a", "title": "Original", "content": "Original mechanism", "elo": 1200, "parent_ids": []}
    survivor = {**first, "elo": 1250}
    edited = {**survivor, "content": "A changed mechanism", "elo": 1400}
    result = {
        "evolution_id": "history-source", "name": "History source", "current_generation": 2,
        "config": {"idea_type": "game_design", "pop_size": 2, "generations": 3, "fitness_alpha": 0.7},
        "history": [
            [first, {"id": "b", "idea": {"title": "Second", "content": "Second mechanism"}, "elo": 1250}],
            [survivor, {"id": "c", "title": "Child", "content": "Child mechanism", "parent_ids": ["a", "b"], "elo": 1300}],
            [edited, {"id": "d", "title": "Late", "content": "Late mechanism", "parent_ids": ["c"], "elo": 1500}],
        ],
        "quality_archive": [{"id": "archive-only", "content": "Absent from history", "elo": 1600}],
        "token_counts": {"total": 999, "cost": {"total_cost": 9.99}},
    }
    result["population"] = copy.deepcopy(result["history"][-1])
    return result


def run_history(tmp_path, export=None, breeder=None, **kwargs):
    return observe_history(export or history_export(), input_path=tmp_path / "source.json",
                           output_path=tmp_path / "observed.json", breeder=breeder or FakeBreeder(),
                           template_data={"name": "template snapshot"}, max_budget=kwargs.pop("max_budget", 1),
                           **kwargs)


def test_history_preserves_snapshots_and_reuses_survivor_but_reextracts_changed_phenotype(tmp_path):
    export = history_export()
    before = copy.deepcopy(export)
    breeder = FakeBreeder()
    result = run_history(tmp_path, export, breeder)
    assert export == before
    for old, new in zip(export["history"], result["history"]):
        assert [{key: value for key, value in raw.items() if key != "gene_observation"} for raw in new] == old
    assert result["config"] == export["config"]
    assert result["quality_archive"] == export["quality_archive"]
    assert result["current_generation"] == 2
    assert result["population"] == result["history"][-1]
    assert result["gene_observation_run"]["source_token_counts"] == export["token_counts"]
    assert result["status"] == "complete"
    assert breeder.calls == [("extract", "a"), ("extract", "b"), ("match", None),
                             ("extract", "c"), ("match", None),
                             ("extract", "a"), ("extract", "d"), ("match", None)]
    assert result["gene_observation_run"]["observed_unique_revisions"] == 5
    assert result["gene_observation_run"]["total_coverage"] == 1
    genes = {gene["label"]: gene for gene in result["gene_catalogue"]["genes"]}
    assert genes["action b"]["first_seen_generation"] == genes["action b"]["last_seen_generation"] == 0
    assert genes["action c"]["first_seen_generation"] == genes["action c"]["last_seen_generation"] == 1
    assert genes["action d"]["first_seen_generation"] == 2
    assert genes["action b"]["current_candidate_count"] == 0
    assert genes["shared relationship"]["current_candidate_ids"] == ["a", "d"]


def test_history_pilot_spans_early_and_late_revisions_and_reports_missing_coverage(tmp_path):
    result = run_history(tmp_path, limit=2)
    metadata = result["gene_observation_run"]
    assert result["status"] == "complete"
    assert [row["id"] for row in metadata["selected_candidate_revisions"]] == ["a", "d"]
    assert metadata["selected_unique_revisions"] == 2
    assert metadata["total_unique_revisions"] == 5
    assert metadata["selected_coverage"] == 1
    assert metadata["total_coverage"] == pytest.approx(0.4)
    assert result["history"][0][0]["gene_observation"]["status"] == "ok"
    assert "gene_observation" not in result["history"][0][1]
    assert "gene_observation" not in result["history"][2][0]  # Same ID, different revision.
    assert result["history"][2][1]["gene_observation"]["status"] == "ok"
    assert [row["observed_unique_revisions"] for row in metadata["generation_coverage"]] == [1, 1, 1]


def test_history_reuses_valid_raw_pilot_and_rebuilds_mappings_chronologically(tmp_path):
    pilot = run_history(tmp_path / "pilot", limit=2)
    # Invalid canonical IDs are deliberately irrelevant to reuse of valid raw extraction.
    for snapshot in [pilot["population"], *pilot["history"]]:
        for candidate in snapshot:
            for gene in candidate.get("gene_observation", {}).get("genes", []):
                gene["gene_id"] = "pilot-catalogue-must-not-leak"
    breeder = FakeBreeder()
    full = run_history(tmp_path / "full", breeder=breeder, reuse_export=pilot)
    assert [identity for kind, identity in breeder.calls if kind == "extract"] == ["b", "c", "a"]
    assert full["gene_observation_run"]["reused_extractions"] == 2
    assert full["gene_observation_run"]["observed_unique_revisions"] == 5
    assert len([call for call in breeder.calls if call[0] == "match"]) == 3
    assert "pilot-catalogue-must-not-leak" not in json.dumps(full)
    late = next(gene for gene in full["gene_catalogue"]["genes"] if gene["label"] == "action d")
    assert late["first_seen_generation"] == 2


def test_history_rejects_reuse_when_active_extraction_fingerprint_changes(tmp_path):
    pilot = run_history(tmp_path / "pilot", limit=2)
    breeder = FakeBreeder()
    breeder.model_name = "changed-model"
    full = run_history(tmp_path / "full", breeder=breeder, reuse_export=pilot)
    assert full["gene_observation_run"]["reused_extractions"] == 0
    assert len([call for call in breeder.calls if call[0] == "extract"]) == 5


def test_history_budget_retains_partial_observations_and_all_unselected_snapshots(tmp_path):
    breeder = FakeBreeder()
    result = run_history(tmp_path, breeder=breeder, max_budget=0.025)
    assert result["status"] == "paused"
    assert len(result["history"]) == 3
    assert breeder.calls == [("extract", "a"), ("extract", "b"), ("match", None)]
    metadata = result["gene_observation_run"]
    assert metadata["budget_reached"] is True
    assert metadata["observed_unique_revisions"] == 2
    assert metadata["mapped_unique_revisions"] == 2
    assert metadata["completed_snapshots"] == 3
    assert "gene_observation" not in result["history"][2][0]
    assert result == json.loads((tmp_path / "observed.json").read_text())


def test_history_saves_after_each_extraction_and_snapshot(tmp_path, monkeypatch):
    import scripts.observe_genes as script
    saved = []
    real_write = script.write_progress

    def record_write(path, result):
        saved.append(copy.deepcopy(result["gene_observation_run"]))
        real_write(path, result)

    monkeypatch.setattr(script, "write_progress", record_write)
    run_history(tmp_path)
    assert {row["attempted_extractions"] for row in saved} == {0, 1, 2, 3, 4, 5}
    assert {row["completed_snapshots"] for row in saved} == {0, 1, 2, 3}


def test_history_revision_identity_ignores_scores_and_representation():
    flat = {"id": "a", "title": "Title", "content": "Mechanism", "elo": 1300}
    nested = {"id": "a", "idea": {"title": "Title", "content": "Mechanism"}, "elo": 1400}
    assert candidate_revision(flat)[0] == candidate_revision(nested)[0]
    assert spread_revisions(list(range(64)), 4) == [0, 21, 42, 63]
