"""The export inspector preserves evidence and cannot execute evolution actions."""
import copy
import json

from fastapi.testclient import TestClient
import pytest

from scripts.preview_export import create_app, prepare_export


def export():
    observation = {"status": "ok", "genotype": "joint movement", "mapping_status": "ok",
                   "genes": [{"text": "joint movement", "gene_id": "gene-1"}]}
    first = {"id": "parent", "idea": {"title": "Parent", "content": "A saved design"},
             "elo": 1288, "gene_observation": observation, "parent_genotypes": ["original parent"]}
    child = {"id": "child", "title": "Child", "content": "A later design", "parent_ids": ["parent"],
             "gene_observation": observation, "selection_metrics": {"fitness": 0.72}}
    return {
        "evolution_id": "actual-run", "name": "Actual saved run", "status": "complete",
        "config": {"idea_type": "game_design", "generations": 1},
        "history": [[first], [first, child]], "population": [first, child],
        "gene_catalogue": {"version": 1, "revision": 1, "genes": [
            {"id": "gene-1", "label": "joint movement", "current_candidate_count": 2}]},
        "token_counts": {"cost": {"total_cost": 0.037}}, "observation_only": True,
    }


def test_preview_preserves_history_observations_scores_costs_and_source():
    source = export()
    before = copy.deepcopy(source)
    with TestClient(create_app(source)) as client:
        progress = client.get("/api/progress?includeHistory=1").json()
        loaded = client.get("/api/evolution/actual-run").json()["data"]
        assert loaded == progress
        assert progress["history"][0][0]["content"] == "A saved design"
        assert progress["history"][0][0]["gene_observation"] == source["history"][0][0]["gene_observation"]
        assert progress["history"][0][0]["elo"] == 1288
        assert progress["history"][1][1]["selection_metrics"] == {"fitness": 0.72}
        assert progress["gene_catalogue"] == source["gene_catalogue"]
        assert progress["token_counts"] == source["token_counts"]
        assert progress["observation_only"] is True
        assert progress["is_running"] is False
        assert client.get("/api/history").json()["items"][0]["can_continue"] is False
    assert source == before


@pytest.mark.parametrize("method,path", [
    ("POST", "/api/start-evolution"), ("POST", "/api/continue-saved-evolution"),
    ("POST", "/api/evolution/actual-run/repair"), ("POST", "/api/save-evolution"),
    ("PUT", "/api/templates/game_design"), ("DELETE", "/api/evolution/actual-run"),
    ("PATCH", "/any-other-path"),
])
def test_all_write_routes_rejected(method, path):
    with TestClient(create_app(export())) as client:
        response = client.request(method, path, json={"malicious": "change"})
        assert response.status_code == 405
        assert "Read-only export preview" in response.json()["message"]


def test_page_uses_real_export_and_does_not_inject_untrusted_names():
    source = export()
    source["name"] = '</script><script>alert("injected")</script>'
    with TestClient(create_app(source)) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert source["name"] not in page.text
        assert "Read-only export preview." in page.text
        assert "loadHistoryItemData" in page.text
        assert client.get("/api/settings/status").json()["has_api_key"] is False
        assert client.get("/api/billing").status_code == 404
        assert client.get("/api/evolution/other-run").status_code == 404


def test_population_only_export_and_invalid_history():
    source = {"population": [{"id": "old", "idea": "Older plain document"}], "status": "paused"}
    result = prepare_export(source, "Older export")
    assert result["history"][0][0]["content"] == "Older plain document"
    assert result["status"] == "paused"
    assert "token_counts" not in result and "gene_catalogue" not in result
    with pytest.raises(ValueError, match="no saved history"):
        prepare_export({}, "empty")
    with pytest.raises(ValueError, match="list of candidate"):
        prepare_export({"history": [["not a candidate"]]}, "invalid")


def test_live_preview_refreshes_valid_checkpoints_and_retains_last_good_on_partial_write(tmp_path):
    path = tmp_path / 'run.json'
    source = export()
    path.write_text(json.dumps(source))
    with TestClient(create_app(source, source_path=path)) as client:
        initial = client.get('/api/preview-status').json()
        updated = copy.deepcopy(source)
        updated['history'].append([updated['population'][1]])
        updated['current_generation'] = 2
        updated['checkpoint_id'] = 'next-checkpoint'
        path.write_text(json.dumps(updated))
        status = client.get('/api/preview-status').json()
        assert status['version'] != initial['version']
        assert status['snapshots'] == 3
        assert status['item']['generations'] == 2
        assert status['checkpoint_id'] == 'next-checkpoint'
        assert len(client.get('/api/evolution/actual-run').json()['data']['history']) == 3
        path.write_text('{partial')
        failed = client.get('/api/preview-status').json()
        assert failed['warning'] and failed['version'] == status['version']
        assert len(client.get('/api/generations').json()) == 3
        path.write_text(json.dumps(updated))
        assert client.get('/api/preview-status').json()['warning'] is None
        assert client.post('/api/start-evolution').status_code == 405
        html = client.get('/').text
        assert 'setInterval(refreshPreview, 15000)' in html
        assert 'getGeneExplorerState.state = explorer' in html
    assert json.loads(path.read_text()) == updated
