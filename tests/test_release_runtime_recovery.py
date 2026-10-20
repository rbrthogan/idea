"""Release regressions for recovery and request ownership; no provider calls."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from idea import auth, viewer
from idea.evolution_orchestrator import EvolutionOrchestrator
from idea.evolution_persistence import EvolutionRepository, EvolutionSerializer
from idea.run_service import RunService
from idea.user_state import UserEvolutionState
from tests.test_evolution_orchestrator import _FakeEngine
from tests.test_runtime_reliability import fake_db


@pytest.mark.parametrize('handler_name,body', [
    ('resume_evolution', {'checkpoint_id': 'saved'}),
    ('continue_evolution', {'evolution_id': 'saved', 'additional_generations': 1}),
    ('continue_saved_evolution', {'evolution_id': 'saved', 'additional_generations': 1}),
])
def test_losing_restore_request_preserves_winning_run(monkeypatch, handler_name, body):
    async def check():
        prior = SimpleNamespace(evolution_id='prior')
        winner = SimpleNamespace(evolution_id='winning')
        state = UserEvolutionState(engine=prior, latest_data=[[{'id': 'prior'}]])
        candidate = SimpleNamespace(
            evolution_id='saved', evolution_name='Saved', generations=3,
            tournament_count=1, full_tournament_rounds=1, tournament_rounds=1,
            unsupported_evaluation_components=[],
        )
        monkeypatch.setattr(viewer.user_states, 'get', AsyncMock(return_value=state))
        monkeypatch.setattr(viewer, '_stored_evolution_artifact_context',
                            AsyncMock(return_value=('google_generative_ai', False)))
        monkeypatch.setattr(viewer, '_resolve_user_provider_key', AsyncMock(return_value='test'))
        monkeypatch.setattr(viewer.EvolutionEngine, 'load_evolution_for_user', AsyncMock(return_value=candidate))
        close = Mock()
        monkeypatch.setattr(viewer, 'close_engine_clients', close)
        async def claim(**kwargs):
            # Another request completed its claim while this request was loading.
            assert state.engine is prior
            state.engine = winner
            state.run_token = 'winning-token'
            state.run_owner_id = 'winning-owner'
            state.status['is_running'] = True
            return False, {'run_token': 'winning-token'}, 'user'
        monkeypatch.setattr(viewer, '_claim_run_slot', claim)
        response = await getattr(viewer, handler_name)(
            SimpleNamespace(json=AsyncMock(return_value=body)), auth.UserInfo('u'))
        assert response.status_code == 409
        assert state.engine is winner
        assert state.run_token == 'winning-token'
        assert state.run_owner_id == 'winning-owner'
        assert state.status['is_running'] is True
        assert state.latest_data == [[{'id': 'prior'}]]
        close.assert_called_once_with(candidate)
    asyncio.run(check())


@pytest.mark.parametrize('body', [[], None, 'invalid'])
def test_resume_rejects_non_object_json_before_storage(monkeypatch, body):
    async def check():
        monkeypatch.setattr(viewer.user_states, 'get', AsyncMock(return_value=UserEvolutionState()))
        read = AsyncMock()
        monkeypatch.setattr(viewer, '_stored_evolution_artifact_context', read)
        response = await viewer.resume_evolution(SimpleNamespace(json=AsyncMock(return_value=body)), auth.UserInfo('u'))
        assert response.status_code == 422
        read.assert_not_called()
    asyncio.run(check())


def test_resume_rejects_malformed_json(monkeypatch):
    async def check():
        monkeypatch.setattr(viewer.user_states, 'get', AsyncMock(return_value=UserEvolutionState()))
        response = await viewer.resume_evolution(
            SimpleNamespace(json=AsyncMock(side_effect=ValueError('malformed'))), auth.UserInfo('u'))
        assert response.status_code == 422
    asyncio.run(check())


def test_failed_checkpoint_is_resumable_in_history_but_archived_execution_is_not():
    recoverable = {'status': 'error', 'generations': 1, 'idea_type': 'drabble'}
    viewer._apply_history_capabilities(recoverable)
    assert recoverable['can_resume'] is True
    archived = {**recoverable, 'idea_type': 'prime_visualization', 'can_use_config': True}
    viewer._apply_history_capabilities(archived)
    assert archived['can_resume'] is False
    assert archived['can_continue'] is False
    assert archived['can_use_config'] is False
    assert archived['execution_archived'] is True


def _reset_stop(engine):
    engine.stop_requested = False
    engine.is_stopped = False


def test_graceful_stop_during_seeding_saves_partial_population(monkeypatch):
    async def check():
        engine = _FakeEngine()
        engine.reset_stop_state = lambda: _reset_stop(engine)
        updates = []
        async def publish(update): updates.append(update)
        orchestrator = EvolutionOrchestrator(engine, publish)
        async def partial_seed(**kwargs):
            engine.population = [{'id': 'finished-seed', 'idea': 'A complete seed'}]
            engine.stop_requested = True
        monkeypatch.setattr(orchestrator, '_seed_initial_population', partial_seed)
        await orchestrator.run(start_generation=0, mode='new')
        assert engine.last_checkpoint_status == 'paused'
        assert engine.history == [engine.population]
        assert updates[-1]['is_resumable'] is True
        assert updates[-1]['durably_saved'] is True
        assert updates[-1]['current_generation'] == 0
        assert updates[-1]['stop_reason'] == 'requested'
        assert updates[-1]['status_message'] == 'Evolution paused. Progress saved; you can resume.'
    asyncio.run(check())


def test_graceful_stop_during_breeding_keeps_consistent_generation():
    async def check():
        engine = _FakeEngine()
        engine.reset_stop_state = lambda: _reset_stop(engine)
        engine._stop_on_breeding = True
        updates = []
        async def publish(update): updates.append(update)
        await EvolutionOrchestrator(engine, publish).run(start_generation=0, mode='resume')
        assert engine.last_checkpoint_status == 'paused'
        assert len(engine.history) == engine.current_generation + 1 == 1
        assert updates[-1]['is_resumable'] is True
        assert updates[-1]['current_generation'] == 0
    asyncio.run(check())


def test_pause_save_failure_does_not_advertise_durable_recovery(monkeypatch):
    async def check():
        engine = _FakeEngine()
        engine.stop_requested = True
        engine.save_checkpoint = AsyncMock(side_effect=ConnectionError('offline'))
        updates = []
        async def publish(update): updates.append(update)
        await EvolutionOrchestrator(engine, publish).run(start_generation=0, mode='new')
        assert updates[-1]['is_resumable'] is False
        assert updates[-1]['durably_saved'] is False
        assert updates[-1]['save_error'] == 'offline'
        assert updates[-1]['status_message'] == 'Evolution failed. The latest progress could not be saved.'
    asyncio.run(check())


def test_force_stop_save_failure_is_not_reported_resumable():
    async def check():
        engine = SimpleNamespace(user_id='u', checkpoint_id='allocated-but-unsaved',
                                 save_checkpoint=AsyncMock(side_effect=ConnectionError('offline')))
        state = UserEvolutionState(engine=engine)
        service = RunService()
        async def operation(): await asyncio.Event().wait()
        task = service.start(state, operation, AsyncMock())
        await asyncio.sleep(0)
        service.request_cancel(state)
        await task
        assert state.status['is_resumable'] is False
        assert state.status['durably_saved'] is False
        assert state.status['save_failed'] is True
    asyncio.run(check())


def test_force_stop_reports_saved_pause_after_checkpoint_success():
    async def check():
        engine = SimpleNamespace(user_id='u', checkpoint_id='saved', save_checkpoint=AsyncMock())
        state = UserEvolutionState(engine=engine)
        service = RunService()
        async def operation(): await asyncio.Event().wait()
        task = service.start(state, operation, AsyncMock())
        await asyncio.sleep(0)
        service.request_cancel(state)
        await task
        assert state.status['is_resumable'] is True
        assert state.status['status_message'] == 'Evolution paused. Progress saved; you can resume.'
        assert state.status['stop_reason'] == 'requested'
    asyncio.run(check())


def test_unexpected_worker_failure_saves_recovery_checkpoint():
    async def check():
        engine = SimpleNamespace(user_id='u', checkpoint_id='saved', save_checkpoint=AsyncMock())
        state = UserEvolutionState(engine=engine)
        service = RunService()
        async def operation(): raise RuntimeError('unexpected failure')
        await service.start(state, operation, AsyncMock())
        engine.save_checkpoint.assert_awaited_once_with(status='error')
        assert state.status['is_resumable'] is True
        assert state.status['error'] == 'unexpected failure'
    asyncio.run(check())


def test_budget_pause_replaces_last_work_message(monkeypatch):
    async def check():
        engine = _FakeEngine()
        engine.reset_stop_state = lambda: _reset_stop(engine)
        budget_reached = False
        engine.check_budget = lambda: budget_reached
        updates = []
        async def publish(update): updates.append(update)
        orchestrator = EvolutionOrchestrator(engine, publish)
        async def reach_budget(**kwargs):
            nonlocal budget_reached
            await publish({'status_message': 'Breeding and refining idea 1/1...'})
            budget_reached = True
            engine.stop_requested = True
        monkeypatch.setattr(orchestrator, '_run_resume', reach_budget)
        await orchestrator.run(start_generation=0, mode='resume')
        final = updates[-1]
        assert engine.last_checkpoint_status == 'paused'
        assert final['stop_reason'] == 'budget'
        assert 'cumulative budget limit was reached' in final['status_message']
        assert 'Use config to start a new experiment with a larger budget' in final['status_message']
        assert final['stop_message'] == final['status_message']
        assert final['is_running'] is False
        assert final['is_resumable'] is True
        assert final['durably_saved'] is True
    asyncio.run(check())


@pytest.mark.parametrize('island_count', [1, 2])
def test_completion_replaces_final_judging_message_and_waits_to_save_complete(monkeypatch, island_count):
    async def check():
        engine = _FakeEngine()
        engine.reset_stop_state = lambda: _reset_stop(engine)
        engine.island_count = island_count
        engine.oracle = None
        status = {'is_resumable': True, 'is_stopped': True, 'stop_reason': 'requested'}
        async def publish(update): status.update(update)
        orchestrator = EvolutionOrchestrator(engine, publish)
        score_final = orchestrator._score_final_generation_for_continuation
        async def checked_final_score():
            assert engine.last_checkpoint_status == 'in_progress'
            await score_final()
        monkeypatch.setattr(orchestrator, '_score_final_generation_for_continuation', checked_final_score)
        await orchestrator.run(start_generation=0, mode='resume')
        assert engine.last_checkpoint_status == 'complete'
        assert status['status_message'] == 'Evolution complete. All results saved.'
        assert status['is_running'] is False
        assert status['is_stopped'] is False
        assert status['is_resumable'] is False
        assert status['stop_reason'] is None
        assert status['stop_message'] is None
        assert status['durably_saved'] is True
    asyncio.run(check())


@pytest.mark.parametrize('status,budget,reason', [
    ('paused', True, 'budget'), ('paused', False, 'requested'), ('complete', False, None),
])
def test_terminal_message_survives_checkpoint_and_cross_instance_progress(monkeypatch, fake_db, status, budget, reason):
    from idea import database
    async def check():
        engine = SimpleNamespace(user_id='u', evolution_id='e', evolution_name='Test',
                                 checkpoint_id='checkpoint', created_at='t', check_budget=lambda: budget)
        save = AsyncMock(return_value='revision')
        monkeypatch.setattr(EvolutionSerializer, 'to_checkpoint_state', Mock(return_value={}))
        monkeypatch.setattr(database, 'save_evolution', save)
        await EvolutionRepository.save_checkpoint(engine, status)
        saved = save.await_args.args[2]
        assert saved['stop_reason'] == reason
        assert saved['status_message']

        claim = await database.claim_active_run('u', {'evolution_id': 'e'}, 30, 'owner')
        state = UserEvolutionState(engine=engine, run_owner_id='owner', run_token=claim['data']['run_token'])
        state.status['status_message'] = 'Running final Swiss tournament...'
        terminal = {key: saved[key] for key in ('status_message', 'stop_message', 'stop_reason')}
        terminal.update({'is_running': False, 'is_stopped': status == 'paused',
                         'is_resumable': status == 'paused', 'durably_saved': True,
                         'checkpoint_id': engine.checkpoint_id})
        await viewer._refresh_run_state('u', state, terminal, force=True)
        # A new process has no in-memory engine and must read durable run state.
        monkeypatch.setattr(viewer.user_states, 'get', AsyncMock(return_value=UserEvolutionState()))
        response = await viewer.get_progress(SimpleNamespace(query_params={}), auth.UserInfo('u'))
        progress = json.loads(response.body)
        assert progress['status_message'] == saved['status_message']
        assert progress['stop_message'] == saved['stop_message']
        assert progress['stop_reason'] == reason
        assert progress['durably_saved'] is True
        assert progress['is_running'] is False
        assert 'disconnected' not in progress['status_message']
    asyncio.run(check())


def test_disconnected_worker_message_replaces_last_work_without_losing_recovery(monkeypatch, fake_db):
    from idea import database
    async def check():
        claim = await database.claim_active_run('u', {'evolution_id': 'e'}, 30, 'owner')
        await database.update_active_run('u', {
            'status': 'in_progress', 'is_resumable': None,
            'status_message': 'Breeding and refining idea 1/1...', 'lease_expires_at_ms': 999,
        }, owner_id='owner', run_token=claim['data']['run_token'])
        monkeypatch.setattr(viewer.user_states, 'get', AsyncMock(return_value=UserEvolutionState()))
        response = await viewer.get_progress(SimpleNamespace(query_params={}), auth.UserInfo('u'))
        progress = json.loads(response.body)
        assert progress['stop_reason'] == 'disconnected'
        assert 'worker disconnected' in progress['status_message']
        assert progress['is_running'] is False
        assert progress['is_resumable'] is True
        assert progress['durably_saved'] is False
    asyncio.run(check())


@pytest.mark.parametrize('count,key', [(1, 'test-key'), (2, None)])
def test_autorating_validation_failure_releases_running_state(monkeypatch, count, key):
    async def check():
        state = UserEvolutionState()
        evolution = {'config': {'idea_type': 'drabble'}, 'history': [[
            {'id': str(i), 'idea': f'Story {i}'} for i in range(count)
        ]]}
        monkeypatch.setattr(viewer.user_states, 'get', AsyncMock(return_value=state))
        monkeypatch.setattr(viewer, '_load_rating_evolution', AsyncMock(return_value=(None, evolution, 'firestore')))
        monkeypatch.setattr(viewer, '_live_or_persisted_artifact_store', lambda *a, **kw: None)
        monkeypatch.setattr(viewer, '_resolve_user_provider_key', AsyncMock(return_value=key))
        critic = Mock()
        monkeypatch.setattr(viewer, 'Critic', critic)
        request = SimpleNamespace(json=AsyncMock(return_value={'evolutionId': 'e', 'numRounds': 1}))
        first = await viewer.auto_rate(request, auth.UserInfo('u'))
        assert first.status_code == 400
        assert state.autorating_status['is_running'] is False
        assert state.autorating_status['status'] == 'error'
        assert state.autorating_task is None
        # A corrected request may try again; it must not hit a stale 409 lock.
        second = await viewer.auto_rate(request, auth.UserInfo('u'))
        assert second.status_code == 400
        critic.assert_not_called()
    asyncio.run(check())


def test_autorating_uses_saved_custom_template_and_type(monkeypatch):
    from tests.test_autorating_evidence import judge
    from idea.prompts.loader import get_builtin_template_snapshot
    async def check():
        state = UserEvolutionState()
        snapshot = get_builtin_template_snapshot('drabble')
        evolution = {'config': {'idea_type': 'saved_custom_template'}, 'template_data': snapshot,
                     'history': [[{'id': 'a', 'idea': 'alpha'}, {'id': 'b', 'idea': 'beta'}]]}
        critic = judge()
        def compare(a, b, idea_type):
            assert idea_type == 'saved_custom_template'
            assert dict(critic._iter_custom_templates())[idea_type] == snapshot
            assert critic._resolve_prompts(idea_type).name == snapshot['name']
            return 'A'
        monkeypatch.setattr(critic, 'compare_ideas', compare)
        monkeypatch.setattr(viewer.user_states, 'get', AsyncMock(return_value=state))
        monkeypatch.setattr(viewer, '_load_rating_evolution', AsyncMock(return_value=(None, evolution, 'firestore')))
        monkeypatch.setattr(viewer, '_live_or_persisted_artifact_store', lambda *a, **kw: None)
        monkeypatch.setattr(viewer, '_resolve_user_provider_key', AsyncMock(return_value='fake-key'))
        monkeypatch.setattr(viewer, '_save_rating_evolution', AsyncMock())
        monkeypatch.setattr(viewer.db, 'save_billing_event', AsyncMock())
        monkeypatch.setattr(viewer, 'Critic', lambda **kwargs: critic)
        request = SimpleNamespace(json=AsyncMock(return_value={
            'evolutionId': 'e', 'numRounds': 1, 'ideaType': 'unrelated-current-ui-selection',
        }))
        response = await viewer.auto_rate(request, auth.UserInfo('u'))
        assert response.status_code == 200, response.body.decode()
        assert state.autorating_status['is_running'] is False
    asyncio.run(check())


@pytest.mark.parametrize('status', ['paused', 'complete', 'error'])
@pytest.mark.parametrize('flag', ['active', 'is_running'])
def test_terminal_progress_retains_live_claim_until_finalizer_acknowledges(fake_db, status, flag):
    from idea import database
    async def check():
        first = await database.claim_active_run('u', {'evolution_id': 'first'}, 30, 'first-owner')
        token = first['data']['run_token']
        await database.update_active_run('u', {
            'status': status, 'active': False, 'is_running': False, flag: True,
        }, owner_id='first-owner', run_token=token)
        second = await database.claim_active_run('u', {'evolution_id': 'second'}, 30, 'second-owner')
        assert second['ok'] is False
        assert second['data']['run_token'] == token
        # Only the owner's cleanup acknowledgement makes the terminal claim free.
        await database.update_active_run('u', {'active': False, 'is_running': False},
                                         owner_id='first-owner', run_token=token)
        next_run = await database.claim_active_run('u', {'evolution_id': 'second'}, 30, 'second-owner')
        assert next_run['ok'] is True
        assert next_run['data']['run_token'] != token
    asyncio.run(check())
