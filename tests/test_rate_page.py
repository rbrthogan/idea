"""
Test the rate page functionality including evolution loading and data flow
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock
from contextlib import contextmanager
from fastapi.testclient import TestClient
import pytest

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.viewer import _autorating_pairwise_evaluator, app, require_auth
from idea.auth import UserInfo


@contextmanager
def authenticated_user(uid="test-user"):
    """Temporarily override auth for endpoints that require a logged-in user."""
    async def mock_require_auth():
        return UserInfo(uid=uid, email=f"{uid}@example.com", is_admin=False)

    app.dependency_overrides[require_auth] = mock_require_auth
    try:
        yield
    finally:
        app.dependency_overrides.pop(require_auth, None)


class TestRatePageAPI:
    """Test the API endpoints used by the rate page"""

    def setup_method(self):
        """Set up test client"""
        self.client = TestClient(app)

    def test_rate_page_loads(self):
        """Test that the rate page loads without errors"""
        response = self.client.get("/rate")
        assert response.status_code == 200
        assert "Compare candidates" in response.text
        assert "evolutionSelect" in response.text

    def test_cross_origin_requests_are_not_allowed_by_default(self):
        response = self.client.options(
            "/api/start-evolution",
            headers={
                "Origin": "https://untrusted.example",
                "Access-Control-Request-Method": "POST",
            },
        )

        assert response.headers.get("access-control-allow-origin") is None

    def test_api_generations_empty_when_no_data(self):
        """Test that /api/generations returns empty array when no data is available

        Note: This endpoint now requires authentication. With no auth token,
        it should return 401/403. We test this behavior here.
        """
        # Without authentication, the endpoint should reject the request
        response = self.client.get("/api/generations")
        assert response.status_code in [401, 403]

    def test_api_generations_with_mock_auth(self):
        """Test that /api/generations returns proper data structure with auth

        Since the endpoint now requires authentication and uses user-scoped state,
        we need to mock both the auth dependency and the user state.
        """
        from idea.auth import UserInfo
        from idea.user_state import UserEvolutionState

        mock_user = UserInfo(uid="test_user_123", email="test@example.com", is_admin=False)
        mock_state = UserEvolutionState()
        mock_state.engine = None  # No engine yet
        mock_state.latest_data = [
            [
                {"title": "Idea 1", "content": "Content 1"},
                {"title": "Idea 2", "content": "Content 2"}
            ],
            [
                {"title": "Idea 3", "content": "Content 3"},
                {"title": "Idea 4", "content": "Content 4"}
            ]
        ]

        async def mock_require_auth():
            return mock_user

        async def mock_get_state(user_id):
            return mock_state

        from idea.viewer import app, require_auth
        from idea.user_state import user_states as real_user_states

        # Override the auth dependency
        app.dependency_overrides[require_auth] = mock_require_auth

        # Patch user_states.get
        with patch.object(real_user_states, 'get', mock_get_state):
            response = self.client.get("/api/generations")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert len(data[0]) == 2
            assert len(data[1]) == 2
            assert data[0][0]["title"] == "Idea 1"
            assert data[0][0]["content"] == "Content 1"

        # Clean up dependency override
        app.dependency_overrides.clear()

    def test_api_evolutions_returns_list(self):
        """Test that /api/evolutions returns a list structure"""
        with authenticated_user(), patch("idea.viewer.db.list_evolutions", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = []
            response = self.client.get("/api/evolutions")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)

    def test_api_evolution_not_found(self):
        """Test that requesting non-existent evolution returns error"""
        with authenticated_user(), patch(
            "idea.viewer.db.get_evolution", new=AsyncMock(return_value=None)
        ) as mock_get, patch(
            "idea.viewer.db.get_checkpoint", new=AsyncMock(return_value=None)
        ) as mock_checkpoint:
            response = self.client.get("/api/evolution/nonexistent")
            assert response.status_code == 404
            mock_get.assert_awaited_once_with("test-user", "nonexistent")
            mock_checkpoint.assert_awaited_once_with("test-user", "nonexistent")

    def test_checkpoint_read_and_delete_require_authentication(self):
        assert self.client.get("/api/checkpoints/checkpoint-1").status_code == 401
        assert self.client.delete("/api/checkpoints/checkpoint-1").status_code == 401

    def test_checkpoint_read_is_scoped_to_authenticated_user(self):
        with authenticated_user(), patch(
            "idea.viewer.db.get_checkpoint",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = {"checkpoint_id": "checkpoint-1"}
            response = self.client.get("/api/checkpoints/checkpoint-1")

        assert response.status_code == 200
        mock_get.assert_awaited_once_with("test-user", "checkpoint-1")

    def test_checkpoint_delete_is_scoped_to_authenticated_user(self):
        with authenticated_user(), patch(
            "idea.viewer.db.delete_checkpoint",
            new_callable=AsyncMock,
        ) as mock_delete:
            mock_delete.return_value = True
            response = self.client.delete("/api/checkpoints/checkpoint-1")

        assert response.status_code == 200
        mock_delete.assert_awaited_once_with("test-user", "checkpoint-1")

    def test_submit_rating_missing_evolution(self):
        """Test that submitting rating for non-existent evolution returns error"""
        rating_data = {
            "idea_a_id": "test_id_a",
            "idea_b_id": "test_id_b",
            "outcome": "A",
            "evolution_id": "nonexistent"
        }
        with authenticated_user(), patch(
            "idea.viewer.db.get_evolution",
            new=AsyncMock(return_value=None),
        ):
            response = self.client.post("/api/submit-rating", json=rating_data)
        assert response.status_code == 404

    @pytest.mark.parametrize(
        ("endpoint", "payload"),
        [
            (
                "/api/submit-rating",
                {
                    "idea_a_id": "a",
                    "idea_b_id": "b",
                    "outcome": "A",
                    "evolution_id": "run",
                },
            ),
            ("/api/auto-rate", {"evolutionId": "run"}),
            ("/api/reset-ratings", {"evolutionId": "run", "ratingType": "all"}),
            ("/api/get-efficient-pair", {"evolution_id": "run"}),
        ],
    )
    def test_rating_mutation_endpoints_require_authentication(self, endpoint, payload):
        response = self.client.post(endpoint, json=payload)

        assert response.status_code == 401

    def test_auto_rate_rejects_missing_evolution_id_before_store_lookup(self):
        with authenticated_user(), patch(
            "idea.viewer._artifact_store_for_evolution",
        ) as mock_store_lookup:
            response = self.client.post("/api/auto-rate", json={})

        assert response.status_code == 400
        assert response.json()["message"] == "Evolution ID is required"
        mock_store_lookup.assert_not_called()

    def test_auto_rate_progress_defaults_when_unauthenticated(self):
        response = self.client.get("/api/auto-rate/progress")
        assert response.status_code == 200
        data = response.json()
        assert data["is_running"] is False
        assert data["status"] == "idle"

    def test_auto_rate_progress_returns_user_state(self):
        from idea.viewer import get_current_user
        from idea.auth import UserInfo
        from idea.user_state import UserEvolutionState, user_states as real_user_states

        mock_user = UserInfo(uid="autorate_user", email="autorate@example.com", is_admin=False)
        mock_state = UserEvolutionState()
        mock_state.autorating_status = {
            "is_running": True,
            "status": "in_progress",
            "status_message": "Running Swiss round 1/3...",
            "progress": 42,
            "completed_matches": 7,
            "total_matches": 16,
            "version": 3,
        }

        async def mock_get_current_user():
            return mock_user

        async def mock_get_state(_user_id):
            return mock_state

        app.dependency_overrides[get_current_user] = mock_get_current_user
        try:
            with patch.object(real_user_states, "get", mock_get_state):
                response = self.client.get("/api/auto-rate/progress")
                assert response.status_code == 200
                data = response.json()
                assert data["is_running"] is True
                assert data["status"] == "in_progress"
                assert data["completed_matches"] == 7
                assert data["total_matches"] == 16
                assert data["version"] == 3
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    def test_autorating_uses_saved_pairwise_component_and_canonical_idea_type(self):
        evaluator = _autorating_pairwise_evaluator(
            {
                "config": {
                    "idea_type": "prime_visualization",
                    "fitness_alpha": 0.6,
                    "evaluation": {
                        "components": [
                            {
                                "id": "quality",
                                "kind": "llm_pairwise",
                                "rounds": 7,
                                "metric_name": "quality_elo",
                            }
                        ],
                        "fitness": {
                            "metrics": [
                                {
                                    "component_id": "quality",
                                    "metric_name": "quality_elo",
                                }
                            ]
                        },
                    },
                }
            }
        )

        assert evaluator.config.id == "quality"
        assert evaluator.config.metric_name == "quality_elo"
        assert evaluator.config.rounds is None

    def test_autorating_rejects_runs_without_a_pairwise_component(self):
        with pytest.raises(ValueError, match="does not configure pairwise"):
            _autorating_pairwise_evaluator(
                {
                    "config": {
                        "evaluation": {
                            "enabled": True,
                            "components": [
                                {
                                    "id": "length",
                                    "kind": "computed",
                                    "runner_ref": "builtin.character_count",
                                    "metric_names": ["character_count"],
                                }
                            ],
                            "fitness": {"enabled": False, "metrics": []},
                        }
                    }
                }
            )

    def test_autorating_rejects_disabled_or_artifact_dependent_policies(self):
        with pytest.raises(ValueError, match="evaluation disabled"):
            _autorating_pairwise_evaluator(
                {
                    "config": {
                        "evaluation": {
                            "enabled": False,
                            "components": [
                                {"id": "quality", "kind": "llm_pairwise"}
                            ],
                            "fitness": {"enabled": False, "metrics": []},
                        }
                    }
                }
            )

    def test_auto_rate_endpoint_passes_explicit_stop_policy_to_evaluator(self):
        from idea.user_state import UserEvolutionState

        captured = {}

        class _FakePairwiseEvaluator:
            def evaluate_population(self, _critic, ideas, _idea_type, **kwargs):
                captured.update(kwargs)
                assert len(ideas) == 2
                kwargs["details"].append(
                    {
                        "round": 1,
                        "pairs": [
                            {"a_idx": 0, "b_idx": 1, "winner": "A"}
                        ],
                    }
                )
                kwargs["progress_callback"](1, 1)
                return SimpleNamespace(batch=SimpleNamespace(records=[]))

        evolution_data = {
            "config": {"idea_type": "drabble"},
            "current_generation": 1,
            "history": [
                [
                    {
                        "id": "idea-a",
                        "idea": {"title": "A", "content": "alpha"},
                    },
                    {
                        "id": "idea-b",
                        "idea": {"title": "B", "content": "beta"},
                    },
                ]
            ],
        }

        with authenticated_user(), \
            patch(
                "idea.viewer.user_states.get",
                new=AsyncMock(return_value=UserEvolutionState()),
            ), \
            patch(
                "idea.viewer._load_rating_evolution",
                new=AsyncMock(return_value=(None, evolution_data, "firestore")),
            ), \
            patch(
                "idea.viewer.db.get_user_api_key",
                new=AsyncMock(return_value="test-api-key"),
            ), \
            patch(
                "idea.viewer._autorating_pairwise_evaluator",
                return_value=_FakePairwiseEvaluator(),
            ), \
            patch("idea.viewer.Critic", return_value=object()), \
            patch("idea.viewer._get_autorating_costs", return_value={}):
            response = self.client.post(
                "/api/auto-rate",
                json={
                    "evolutionId": "evolution-1",
                    "numRounds": 1,
                    "skipSave": True,
                },
            )

        assert response.status_code == 200, response.text
        assert callable(captured["should_stop"])
        assert captured["should_stop"]() is False

        with pytest.raises(ValueError, match="authenticated artifact store"):
            _autorating_pairwise_evaluator(
                {
                    "config": {
                        "evaluation": {
                            "components": [
                                {
                                    "id": "render",
                                    "kind": "llm_code_artifact",
                                    "execution_profile": "remote-render",
                                    "provides": ["artifact.primary"],
                                },
                                {
                                    "id": "quality",
                                    "kind": "llm_pairwise",
                                    "requires": ["artifact.primary"],
                                },
                            ],
                            "allow_generated_code_execution": True,
                            "fitness": {
                                "metrics": [
                                    {
                                        "component_id": "quality",
                                        "metric_name": "quality_elo",
                                    }
                                ]
                            },
                        }
                    }
                }
            )

        loader = object()
        evaluator = _autorating_pairwise_evaluator(
            {
                "config": {
                    "evaluation": {
                        "components": [
                            {
                                "id": "render",
                                "kind": "llm_code_artifact",
                                "execution_profile": "remote-render",
                                "provides": ["artifact.primary"],
                            },
                            {
                                "id": "quality",
                                "kind": "llm_pairwise",
                                "requires": ["artifact.primary"],
                            },
                        ],
                        "allow_generated_code_execution": True,
                        "fitness": {
                            "metrics": [
                                {
                                    "component_id": "quality",
                                    "metric_name": "quality_elo",
                                }
                            ]
                        },
                    }
                }
            },
            artifact_loader=loader,
        )
        assert evaluator.artifact_loader is loader

    def test_start_evolution_logs_field_count_without_names_or_values(self, caplog):
        from idea.user_state import UserEvolutionState

        secret_marker = "do-not-log-this-secret"
        state = UserEvolutionState()

        with authenticated_user(), \
            patch("idea.viewer.user_states.get", new=AsyncMock(return_value=state)), \
            patch("idea.viewer.db.get_user_api_key", new=AsyncMock(return_value="test-api-key")), \
            caplog.at_level(logging.INFO, logger="idea.viewer"):
            response = self.client.post(
                "/api/start-evolution",
                json={"apiKey": secret_marker, "popSize": "invalid"},
            )

        assert response.status_code == 422
        records = [
            record
            for record in caplog.records
            if record.getMessage() == "Received start-evolution request"
        ]
        assert len(records) == 1
        assert records[0].request_field_count == 2
        assert "apiKey" not in caplog.text
        assert secret_marker not in caplog.text

    def test_start_evolution_rejects_non_object_before_logging(self, caplog):
        from idea.user_state import UserEvolutionState

        with authenticated_user(), \
            patch("idea.viewer.user_states.get", new=AsyncMock(return_value=UserEvolutionState())), \
            patch("idea.viewer.db.get_user_api_key", new=AsyncMock(return_value="test-api-key")), \
            caplog.at_level(logging.INFO, logger="idea.viewer"):
            response = self.client.post("/api/start-evolution", json=["secret-value"])

        assert response.status_code == 422
        assert "Received start-evolution request" not in caplog.text
        assert "secret-value" not in caplog.text


class TestEvolutionDataFlow:
    """Test the evolution data loading and processing flow"""

    def setup_method(self):
        """Set up test client and temporary data directory"""
        self.client = TestClient(app)
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_local_save_requires_authentication(self):
        payload = {
            "data": {"history": []},
            "filename": "unauthenticated.json",
        }

        with patch("idea.viewer.DATA_DIR", self.temp_path):
            response = self.client.post("/api/save-evolution", json=payload)

        assert response.status_code == 401
        assert not (self.temp_path / "unauthenticated.json").exists()

    def test_local_save_requires_json_object(self):
        with patch("idea.viewer.DATA_DIR", self.temp_path), authenticated_user():
            response = self.client.post("/api/save-evolution", json=[])

        assert response.status_code == 400

    def test_local_save_rejects_paths_outside_data_directory(self):
        outside_path = self.temp_path.parent / f"{self.temp_path.name}-outside.json"
        assert not outside_path.exists()

        filenames = [
            "../escape.json",
            "nested/escape.json",
            r"..\escape.json",
            str(outside_path),
            "not-json.txt",
        ]

        with patch("idea.viewer.DATA_DIR", self.temp_path), authenticated_user():
            for filename in filenames:
                response = self.client.post(
                    "/api/save-evolution",
                    json={"data": {"history": []}, "filename": filename},
                )
                assert response.status_code == 400, filename

        assert not outside_path.exists()
        assert list(self.temp_path.iterdir()) == []

    def test_local_save_atomically_replaces_symlink_without_touching_target(self):
        outside_path = self.temp_path.parent / f"{self.temp_path.name}-symlink-target.json"
        outside_path.write_text('{"outside": "unchanged"}')

        try:
            with patch("idea.viewer.DATA_DIR", self.temp_path), authenticated_user():
                bootstrap = self.client.post(
                    "/api/save-evolution",
                    json={"data": {"history": []}, "filename": "bootstrap.json"},
                )
                assert bootstrap.status_code == 200
                link_path = Path(bootstrap.json()["file_path"]).with_name("redirect.json")
                link_path.symlink_to(outside_path)

                response = self.client.post(
                    "/api/save-evolution",
                    json={
                        "data": {"history": [[{"id": "safe"}]]},
                        "filename": link_path.name,
                    },
                )

            assert response.status_code == 200
            assert not link_path.is_symlink()
            assert json.loads(link_path.read_text())["history"][0][0]["id"] == "safe"
            assert outside_path.read_text() == '{"outside": "unchanged"}'
        finally:
            outside_path.unlink(missing_ok=True)

    def test_rating_read_refuses_user_file_symlink(self):
        outside_path = self.temp_path.parent / f"{self.temp_path.name}-rating-symlink.json"
        outside_path.write_text(json.dumps({
            "history": [[
                {"id": "a", "title": "Outside A", "content": "A"},
                {"id": "b", "title": "Outside B", "content": "B"},
            ]],
        }))

        try:
            with patch("idea.viewer.DATA_DIR", self.temp_path), authenticated_user(), patch(
                "idea.viewer.db.get_evolution",
                new=AsyncMock(return_value=None),
            ):
                bootstrap = self.client.post(
                    "/api/save-evolution",
                    json={"data": {"history": []}, "filename": "bootstrap.json"},
                )
                local_path = Path(bootstrap.json()["file_path"]).with_name("linked.json")
                local_path.symlink_to(outside_path)

                response = self.client.post(
                    "/api/get-efficient-pair",
                    json={"evolution_id": "linked"},
                )

            assert response.status_code == 400
            assert "Outside A" in outside_path.read_text()
        finally:
            outside_path.unlink(missing_ok=True)

    def test_local_saves_are_isolated_by_authenticated_user(self):
        run_a = {
            "history": [[
                {"id": "a1", "title": "User A 1", "content": "A1"},
                {"id": "a2", "title": "User A 2", "content": "A2"},
            ]],
        }
        run_b = {
            "history": [[
                {"id": "b1", "title": "User B 1", "content": "B1"},
                {"id": "b2", "title": "User B 2", "content": "B2"},
            ]],
        }

        with patch("idea.viewer.DATA_DIR", self.temp_path):
            with authenticated_user("user-a"):
                saved_a = self.client.post(
                    "/api/save-evolution",
                    json={"data": run_a, "filename": "shared.json"},
                )
            with authenticated_user("user-b"):
                saved_b = self.client.post(
                    "/api/save-evolution",
                    json={"data": run_b, "filename": "shared.json"},
                )

            path_a = Path(saved_a.json()["file_path"])
            path_b = Path(saved_b.json()["file_path"])
            assert path_a != path_b
            assert json.loads(path_a.read_text()) == run_a
            assert json.loads(path_b.read_text()) == run_b

            with patch("idea.viewer.db.get_evolution", new=AsyncMock(return_value=None)):
                with authenticated_user("user-a"):
                    pair_a = self.client.post(
                        "/api/get-efficient-pair",
                        json={"evolution_id": "shared"},
                    )
                with authenticated_user("user-b"):
                    pair_b = self.client.post(
                        "/api/get-efficient-pair",
                        json={"evolution_id": "shared"},
                    )

        assert pair_a.status_code == 200
        assert pair_b.status_code == 200
        assert {pair_a.json()["idea_a"]["id"], pair_a.json()["idea_b"]["id"]} == {"a1", "a2"}
        assert {pair_b.json()["idea_a"]["id"], pair_b.json()["idea_b"]["id"]} == {"b1", "b2"}

    def test_history_scans_only_authenticated_user_namespace_by_default(self):
        from idea.user_state import UserEvolutionState

        saved_data = {"history": [[{"id": "mine"}]], "config": {"idea_type": "test"}}
        with patch("idea.viewer.DATA_DIR", self.temp_path):
            with authenticated_user("history-user"):
                mine = self.client.post(
                    "/api/save-evolution",
                    json={"data": saved_data, "filename": "mine.json"},
                )
            with authenticated_user("other-user"):
                other = self.client.post(
                    "/api/save-evolution",
                    json={"data": saved_data, "filename": "other.json"},
                )
            assert mine.status_code == 200
            assert other.status_code == 200
            (self.temp_path / "root-legacy.json").write_text(json.dumps(saved_data))

            with authenticated_user("history-user"), patch(
                "idea.viewer.user_states.get",
                new=AsyncMock(return_value=UserEvolutionState()),
            ), patch(
                "idea.viewer.EvolutionEngine.list_evolutions_for_user",
                new=AsyncMock(return_value=[]),
            ), patch(
                "idea.viewer.EvolutionEngine.list_checkpoints_for_user",
                new=AsyncMock(return_value=[]),
            ), patch.dict(
                os.environ,
                {"IDEA_ALLOW_UNSCOPED_LEGACY_LOCAL_FILES": ""},
            ):
                response = self.client.get("/api/history")

        assert response.status_code == 200
        ids = {item["id"] for item in response.json()["items"]}
        assert "mine" in ids
        assert "other" not in ids
        assert "root-legacy" not in ids

    def test_rating_fallback_rejects_traversal_before_reading_outside_file(self):
        outside_path = self.temp_path.parent / f"{self.temp_path.name}-rating-read.json"
        outside_path.write_text('{"history": "outside-sentinel"}')

        try:
            relative_id = f"../{outside_path.stem}"
            with patch("idea.viewer.DATA_DIR", self.temp_path), authenticated_user(), patch(
                "idea.viewer.db.get_evolution",
                new=AsyncMock(return_value=None),
            ):
                response = self.client.post(
                    "/api/get-efficient-pair",
                    json={"evolution_id": relative_id},
                )

            assert response.status_code == 400
            assert outside_path.read_text() == '{"history": "outside-sentinel"}'
        finally:
            outside_path.unlink(missing_ok=True)

    def test_rating_fallback_rejects_absolute_path_before_rewrite(self):
        outside_path = self.temp_path.parent / f"{self.temp_path.name}-rating-write.json"
        original = json.dumps({
            "history": [[{
                "id": "outside",
                "title": "Outside",
                "content": "must not be rewritten",
                "ratings": {"auto": 1900, "manual": 1800},
            }]],
        })
        outside_path.write_text(original)

        try:
            absolute_id = str(outside_path.with_suffix(""))
            with patch("idea.viewer.DATA_DIR", self.temp_path), authenticated_user(), patch(
                "idea.viewer.db.get_evolution",
                new=AsyncMock(return_value=None),
            ):
                response = self.client.post(
                    "/api/reset-ratings",
                    json={"evolutionId": absolute_id, "ratingType": "all"},
                )

            assert response.status_code == 400
            assert outside_path.read_text() == original
        finally:
            outside_path.unlink(missing_ok=True)

    def test_unscoped_legacy_rating_files_are_disabled_by_default(self):
        evolution_id = "legacy-run_2.5=demo"
        local_path = self.temp_path / f"{evolution_id}.json"
        local_path.write_text(json.dumps({
            "history": [[
                {"id": "a", "title": "A", "content": "Alpha"},
                {"id": "b", "title": "B", "content": "Beta"},
            ]],
        }))

        with patch("idea.viewer.DATA_DIR", self.temp_path), authenticated_user(), patch(
            "idea.viewer.db.get_evolution",
            new=AsyncMock(return_value=None),
        ), patch.dict(os.environ, {"IDEA_ALLOW_UNSCOPED_LEGACY_LOCAL_FILES": ""}):
            response = self.client.post(
                "/api/get-efficient-pair",
                json={"evolution_id": evolution_id},
            )

        assert response.status_code == 404

    def test_opt_in_legacy_rating_migrates_only_when_user_votes(self):
        evolution_id = "legacy-run_2.5=demo"
        local_path = self.temp_path / f"{evolution_id}.json"
        original = {
            "_save_revision": "revision-retained-in-legacy-export",
            "history": [[
                {"id": "a", "title": "A", "content": "Alpha"},
                {"id": "b", "title": "B", "content": "Beta"},
            ]],
        }
        local_path.write_text(json.dumps(original))

        with patch("idea.viewer.DATA_DIR", self.temp_path), authenticated_user(), patch(
            "idea.viewer.db.get_evolution",
            new=AsyncMock(return_value=None),
        ), patch.dict(
            os.environ,
            {"IDEA_ALLOW_UNSCOPED_LEGACY_LOCAL_FILES": "1"},
        ):
            response = self.client.post(
                "/api/get-efficient-pair",
                json={"evolution_id": evolution_id},
            )

            assert response.status_code == 200
            assert not list((self.temp_path / "local-users").glob(f"user-*/{evolution_id}.json"))
            pair = response.json()
            voted = self.client.post("/api/submit-rating", json={
                "evolution_id": evolution_id,
                "idea_a_id": pair["idea_a"]["id"],
                "idea_b_id": pair["idea_b"]["id"],
                "pair_token": pair["pair_token"],
                "outcome": "A",
            })
            replayed = self.client.post("/api/submit-rating", json={
                "evolution_id": evolution_id,
                "idea_a_id": pair["idea_a"]["id"],
                "idea_b_id": pair["idea_b"]["id"],
                "pair_token": pair["pair_token"],
                "outcome": "A",
            })

        assert response.status_code == 200
        assert voted.status_code == 200
        assert replayed.status_code == 409
        assert {response.json()["idea_a"]["id"], response.json()["idea_b"]["id"]} == {"a", "b"}
        assert json.loads(local_path.read_text()) == original
        migrated = list((self.temp_path / "local-users").glob(f"user-*/{evolution_id}.json"))
        assert len(migrated) == 1

    def test_evolution_save_and_load_flow(self):
        """Test the complete flow of saving and loading evolution data"""
        # Create mock evolution data
        evolution_data = {
            "history": [
                [
                    {
                        "id": "idea_1",
                        "title": "Test Idea 1",
                        "content": "This is test content 1",
                        "elo": 1500,
                        "ratings": {"auto": 1500, "manual": 1500}
                    },
                    {
                        "id": "idea_2",
                        "title": "Test Idea 2",
                        "content": "This is test content 2",
                        "elo": 1520,
                        "ratings": {"auto": 1520, "manual": 1480}
                    }
                ],
                [
                    {
                        "id": "idea_3",
                        "title": "Test Idea 3",
                        "content": "This is test content 3",
                        "elo": 1540,
                        "ratings": {"auto": 1540, "manual": 1510}
                    }
                ]
            ]
        }

        # Patch the DATA_DIR to use our temporary directory
        with patch('idea.viewer.DATA_DIR', self.temp_path):
            # Save evolution data
            save_data = {
                "data": evolution_data,
                "filename": "test_evolution.json"
            }
            with authenticated_user():
                response = self.client.post("/api/save-evolution", json=save_data)
            assert response.status_code == 200

            # Verify file was created
            file_path = Path(response.json()["file_path"])
            assert file_path.exists()
            assert file_path.parent.parent.name == "local-users"

            with authenticated_user(), \
                patch("idea.viewer.db.list_evolutions", new_callable=AsyncMock) as mock_list, \
                patch("idea.viewer.db.get_evolution", new_callable=AsyncMock) as mock_get:
                mock_list.return_value = [
                    {
                        "id": "test_evolution",
                        "updated_at": "2026-01-01T00:00:00Z",
                        "name": "test_evolution",
                    }
                ]
                mock_get.return_value = evolution_data

                # Test loading evolutions list
                response = self.client.get("/api/evolutions")
                assert response.status_code == 200
                evolutions = response.json()
                assert len(evolutions) >= 1
                assert any(e["id"] == "test_evolution" for e in evolutions)

                # Test loading specific evolution
                response = self.client.get("/api/evolution/test_evolution")
                assert response.status_code == 200
                loaded_data = response.json()
                assert "data" in loaded_data
                assert "history" in loaded_data["data"]
                assert len(loaded_data["data"]["history"]) == 2

    def test_evolution_data_structure_validation(self):
        """Test that evolution data has the expected structure for the rate page"""
        evolution_data = {
            "history": [
                [
                    {
                        "id": "idea_1",
                        "title": "Test Idea 1",
                        "content": "This is test content 1",
                        "elo": 1500
                    }
                ]
            ]
        }

        with patch('idea.viewer.DATA_DIR', self.temp_path):
            # Save evolution data
            save_data = {
                "data": evolution_data,
                "filename": "test_evolution_structure.json"
            }
            with authenticated_user():
                response = self.client.post("/api/save-evolution", json=save_data)
            assert response.status_code == 200

            with authenticated_user(), patch("idea.viewer.db.get_evolution", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = evolution_data

                # Load and verify structure
                response = self.client.get("/api/evolution/test_evolution_structure")
                assert response.status_code == 200
                loaded_data = response.json()

                # Verify essential fields for rate page
                assert "id" in loaded_data
                assert "timestamp" in loaded_data
                assert "data" in loaded_data
                assert "history" in loaded_data["data"]

                # Verify ideas structure
                ideas = loaded_data["data"]["history"][0]
                assert len(ideas) == 1
                idea = ideas[0]
                assert "id" in idea
                assert "title" in idea
                assert "content" in idea


class TestRatePageEdgeCases:
    """Test edge cases and error conditions for the rate page"""

    def setup_method(self):
        """Set up test client"""
        self.client = TestClient(app)

    def test_empty_evolution_handling(self):
        """Test handling of evolution with no ideas"""
        evolution_data = {"history": []}

        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)

        try:
            with patch('idea.viewer.DATA_DIR', temp_path):
                # Save empty evolution
                save_data = {
                    "data": evolution_data,
                    "filename": "empty_evolution.json"
                }
                with authenticated_user():
                    response = self.client.post("/api/save-evolution", json=save_data)
                assert response.status_code == 200

                with authenticated_user(), patch("idea.viewer.db.get_evolution", new_callable=AsyncMock) as mock_get:
                    mock_get.return_value = evolution_data
                    # Load empty evolution
                    response = self.client.get("/api/evolution/empty_evolution")
                    assert response.status_code == 200
                    loaded_data = response.json()
                    assert loaded_data["data"]["history"] == []
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_malformed_evolution_data(self):
        """Test handling of malformed evolution data"""
        # Test with missing history
        evolution_data = {"not_history": []}

        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)

        try:
            with patch('idea.viewer.DATA_DIR', temp_path):
                save_data = {
                    "data": evolution_data,
                    "filename": "malformed_evolution.json"
                }
                with authenticated_user():
                    response = self.client.post("/api/save-evolution", json=save_data)
                assert response.status_code == 200

                with authenticated_user(), patch("idea.viewer.db.get_evolution", new_callable=AsyncMock) as mock_get:
                    mock_get.return_value = evolution_data
                    # Load malformed evolution - should not crash
                    response = self.client.get("/api/evolution/malformed_evolution")
                    assert response.status_code == 200
                    loaded_data = response.json()
                    assert "data" in loaded_data
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
