import asyncio
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from idea.evolution import EvolutionEngine
from idea.run_requests import parse_run_request, parse_additional_generations


@pytest.mark.parametrize('value', ['bad', 0, -1, 1.5, 1000000])
def test_rejects_invalid_population_counts(value):
    with pytest.raises(ValueError):
        parse_run_request({'popSize': value})


@pytest.mark.parametrize('field,value', [('creativeTemp', 9), ('topP', -0.4),
    ('mutationRate', 2), ('maxBudget', float('nan')), ('islandCount', 100),
    ('tournamentCount', 100000), ('generations', 1000000)])
def test_rejects_invalid_request_settings(field, value):
    with pytest.raises(ValueError):
        parse_run_request({field: value})


def test_aliases_preserve_config_and_tournament_semantics():
    config = parse_run_request({'popSize': 8, 'generations': 4,
                               'tournamentCount': 0.5, 'randomSeed': 42})
    assert config.pop_size == 8
    assert config.tournament_rounds == 2
    assert config.random_seed == 42
    assert config == type(config).from_stored_mapping(config.to_persistence_dict())


def test_request_seed_numbers_are_exact_in_browser_json():
    assert parse_run_request({'randomSeed': 2**53 - 1}).random_seed == 2**53 - 1
    with pytest.raises(ValueError, match='exact decimal string'):
        parse_run_request({'randomSeed': 2**53})
    # Historical seeds can still be supplied without browser-number rounding.
    assert parse_run_request({'random_seed': str(2**64 - 1)}).random_seed == 2**64 - 1


def test_new_root_seed_is_browser_safe_and_legacy_seed_stays_exact():
    with patch('idea.evolution.Ideator'), patch('idea.evolution.Critic'), \
         patch('idea.evolution.Breeder'), patch('idea.evolution.Oracle'), \
         patch('idea.evolution.Formatter'), patch('idea.evolution.DiversityCalculator'), \
         patch('idea.evolution.secrets.randbits', side_effect=lambda bits: 2**bits - 1):
        fresh = EvolutionEngine()
        legacy = EvolutionEngine(random_seed=2**64 - 1)
    assert fresh.config_state.random_seed == 2**53 - 1
    assert legacy.config_state.random_seed == 2**64 - 1


@pytest.mark.parametrize('value', ['bad', 0, -1, 1.5, 1000000])
def test_continuation_is_bounded(value):
    with pytest.raises(ValueError):
        parse_additional_generations({'additionalGenerations': value})


def test_invalid_start_preserves_previous_state():
    from idea.auth import UserInfo
    from idea.viewer import app, require_auth, user_states
    from idea.user_state import UserEvolutionState

    state = UserEvolutionState(latest_data=[[{'id': 'previous'}]])
    async def auth():
        return UserInfo(uid='validation-test')
    async def get_state(_):
        return state
    app.dependency_overrides[require_auth] = auth
    try:
        with patch.object(user_states, 'get', side_effect=get_state):
            response = TestClient(app).post('/api/start-evolution', json={'popSize': 'bad'})
        assert response.status_code == 422
        assert state.latest_data == [[{'id': 'previous'}]]
        assert state.engine is None
    finally:
        app.dependency_overrides.pop(require_auth, None)


def test_constructor_uses_normalized_config_before_providers():
    with patch('idea.evolution.Ideator') as ideator, patch('idea.evolution.Critic'), \
         patch('idea.evolution.Breeder') as breeder, patch('idea.evolution.Oracle'), \
         patch('idea.evolution.Formatter'), patch('idea.evolution.DiversityCalculator'):
        engine = EvolutionEngine(idea_type='airesearch', pop_size=-5, generations=0,
                                 creative_temp=9, top_p=-0.4, mutation_rate=2)
    assert engine.pop_size == engine.config_state.pop_size == 1
    assert engine.generations == engine.config_state.generations == 1
    assert engine.creative_temp == engine.config_state.creative_temp == 2
    assert ideator.call_args.kwargs['temperature'] == 2
    creative_breeder = next(call for call in breeder.call_args_list if 'mutation_rate' in call.kwargs)
    assert creative_breeder.kwargs['mutation_rate'] == engine.mutation_rate == 1


@pytest.mark.parametrize('id_field', ['checkpoint_id', 'checkpointId'])
def test_resume_accepts_history_and_legacy_checkpoint_fields(id_field):
    """Both clients must reach restoration, not fail request validation."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from idea import viewer
    from idea.auth import UserInfo
    from idea.user_state import UserEvolutionState

    async def check():
        state = UserEvolutionState()
        engine = SimpleNamespace(
            unsupported_evaluation_components=[], evolution_id='saved-run',
            evolution_name='Arcade run', generations=5, current_generation=2,
            tournament_count=1.0, full_tournament_rounds=19, tournament_rounds=19,
            history=[], contexts=[], specific_prompts=[], breeding_prompts=[],
        )
        request = SimpleNamespace(json=AsyncMock(return_value={id_field: 'saved-run'}))
        with patch.object(viewer.user_states, 'get', AsyncMock(return_value=state)), \
             patch.object(viewer, '_user_execution_mode', AsyncMock(return_value=None)), \
             patch.object(viewer, '_stored_evolution_artifact_context', AsyncMock(return_value=('google_generative_ai', False))), \
             patch.object(viewer, '_resolve_user_provider_key', AsyncMock(return_value='test-key')), \
             patch.object(viewer.EvolutionEngine, 'load_evolution_for_user', AsyncMock(return_value=engine)) as load, \
             patch.object(viewer, '_claim_run_slot', AsyncMock(return_value=(True, {}, 'user'))), \
             patch.object(viewer.db, 'update_active_run', AsyncMock()), \
             patch.object(viewer, '_heartbeat_loop', AsyncMock()), \
             patch.object(viewer, '_launch_run', AsyncMock()) as launch:
            response = await viewer.resume_evolution(request, UserInfo(uid='resume-user'))
            assert response.status_code == 200
            load.assert_awaited_once_with(
                'resume-user', 'saved-run', api_key='test-key',
                evaluation_runners={}, artifact_store=None,
            )
            launch.assert_awaited_once_with(state, mode='resume')
            assert state.status['current_generation'] == 2
            assert state.status['total_generations'] == 5
            assert state.engine is engine
            # Drain the mocked heartbeat; no worker or provider call is made.
            await asyncio.sleep(0)
    asyncio.run(check())


def test_stopped_batch_cancels_and_drains_remaining_work():
    async def exercise():
        engine = object.__new__(EvolutionEngine)
        engine.semaphore = asyncio.Semaphore(2)
        engine.stop_requested = False
        engine.random_uniform = lambda *args: 0
        cancelled = asyncio.Event()
        slow_started = asyncio.Event()
        async def fast():
            await slow_started.wait()
            return 'fast'
        async def slow():
            slow_started.set()
            try:
                await asyncio.sleep(20)
            finally:
                cancelled.set()
        async def progress(_):
            engine.stop_requested = True
        result = await engine._run_batch_with_progress([fast, slow], progress, {}, 0, 2, '{completed}/{total}')
        assert result == ['fast', None]
        assert cancelled.is_set()
    asyncio.run(exercise())
