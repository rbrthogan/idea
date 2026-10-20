import asyncio
import json
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from idea import database, viewer
from idea.job_dispatch import CloudJobDispatcher, parse_job_payload
from idea.run_service import RunJob
from idea.user_state import UserEvolutionState
from tests.test_runtime_reliability import fake_db


def test_dispatch_payload_contains_no_key_and_uses_job_override():
    calls = []
    class Session:
        def post(self, url, **kwargs):
            calls.append((url, kwargs))
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'name': 'operations/run-1'})
    job = RunJob('u', 'e', 'token', 'continue', 2)
    dispatcher = CloudJobDispatcher('projects/p/locations/europe-west1/jobs/idea-worker', Session())
    assert asyncio.run(dispatcher.dispatch(job)) == 'operations/run-1'
    url, kwargs = calls[0]
    assert url.endswith('/jobs/idea-worker:run')
    args = kwargs['json']['overrides']['containerOverrides'][0]['args']
    assert args[0] == '--payload'
    assert json.loads(args[1]) == asdict(job)
    assert kwargs['timeout'] == 20


@pytest.mark.parametrize('change', [{'api_key': 'secret'}, {'mode': 'invalid'}, {'additional_generations': -1}, {'user_id': '../u'}])
def test_job_payload_rejects_invalid_or_secret_fields(change):
    payload = {**asdict(RunJob('u', 'e', 'token')), **change}
    with pytest.raises(ValueError): parse_job_payload(json.dumps(payload))


def test_worker_adopts_exact_queued_delivery_once(fake_db):
    async def check():
        await database.claim_active_run('u', {'evolution_id': 'e', 'run_token': 't', 'status': 'queued'}, 30, 'web')
        assert not await database.adopt_active_run('u', 'e', 'other', 'worker', 30)
        assert await database.adopt_active_run('u', 'e', 't', 'worker', 30)
        assert not await database.adopt_active_run('u', 'e', 't', 'duplicate', 30)
        record = await database.get_active_run('u')
        assert record['owner_id'] == 'worker'
        assert not await database.update_active_run('u', {'status': 'error'}, 0, 'web', run_token='t')
    asyncio.run(check())


def test_worker_adoption_transfers_global_lease_atomically(fake_db):
    async def check():
        await database.claim_active_run('u', {'evolution_id': 'e', 'run_token': 't', 'status': 'queued'}, 30, 'web')
        await database.claim_global_run_lock('web', 'u', 'e', 30, run_token='t')
        assert await database.adopt_active_run('u', 'e', 't', 'worker', 30, transfer_global_lock=True)
        assert not await database.release_global_run_lock('web', run_token='t')
        assert await database.refresh_global_run_lock('worker', 30, run_token='t')
    asyncio.run(check())


def test_cloud_launch_saves_before_dispatch_and_detaches(fake_db, monkeypatch):
    async def check():
        monkeypatch.setenv('IDEA_RUN_JOB', 'projects/p/locations/europe-west1/jobs/worker')
        await database.claim_active_run('u', {'evolution_id': 'e', 'run_token': 't'}, 30, viewer.INSTANCE_ID)
        saved = []
        engine = SimpleNamespace(user_id='u', evolution_id='e', _run_token='t', pre_judge_evaluators=[])
        async def save(**kwargs): saved.append(kwargs['status'])
        engine.save_checkpoint = save
        state = UserEvolutionState(engine=engine, run_owner_id=viewer.INSTANCE_ID, run_token='t')
        async def dispatch(self, job):
            assert saved == ['queued']
            record = await database.get_active_run('u')
            assert record['status'] == 'queued'
            assert record['lease_expires_at_ms'] == 601000
            return 'operations/test'
        monkeypatch.setattr(CloudJobDispatcher, 'dispatch', dispatch)
        await viewer._launch_run(state)
        assert state.engine is engine  # response can still serialize the engine
        await asyncio.sleep(0)
        assert state.engine is None
        assert state.status['is_running'] is False
        assert (await database.get_active_run('u'))['job_operation'] == 'operations/test'
    asyncio.run(check())


def test_queued_cancel_prevents_worker_adoption(fake_db, monkeypatch):
    async def check():
        await database.claim_active_run('u', {'evolution_id': 'e', 'run_token': 't', 'status': 'queued'}, 30, 'web')
        monkeypatch.setattr(viewer.user_states, 'get', AsyncMock(return_value=UserEvolutionState()))
        response = await viewer.force_stop_evolution(SimpleNamespace(uid='u'))
        assert response.status_code == 200
        assert not await database.adopt_active_run('u', 'e', 't', 'worker', 30)
    asyncio.run(check())


def test_worker_loads_encrypted_key_and_completes_through_supervisor(fake_db, monkeypatch):
    from idea import worker
    async def check():
        await database.claim_active_run('u', {'evolution_id': 'e', 'run_token': 't', 'status': 'queued'}, 30, 'web')
        engine = SimpleNamespace(user_id='u', evolution_id='e', evolution_name='Run', unsupported_evaluation_components=[], pre_judge_evaluators=[])
        monkeypatch.setattr(database, 'get_evolution', AsyncMock(return_value={'config': {'model_type': 'gemini-2.5-flash'}}))
        key_lookup = AsyncMock(return_value='fake-key')
        monkeypatch.setattr(database, 'get_user_api_key', key_lookup)
        monkeypatch.setattr(viewer, '_stored_evolution_artifact_context', AsyncMock(return_value=('google_generative_ai', False)))
        monkeypatch.setattr(worker.EvolutionEngine, 'load_evolution_for_user', AsyncMock(return_value=engine))
        async def run(state): state.status.update({'is_running': False, 'progress': 100})
        monkeypatch.setattr(viewer, 'run_evolution_task', run)
        assert await worker.execute_job(RunJob('u', 'e', 't'))
        key_lookup.assert_awaited_once_with('u', 'google_generative_ai')
        record = await database.get_active_run('u')
        assert record['status'] == 'complete'
        assert record['cancel_acknowledged'] is True
        assert record['is_running'] is False
    asyncio.run(check())


def test_bootstrap_lease_loss_cancels_initialization_before_provider_work(fake_db, monkeypatch):
    from idea import worker
    async def check():
        await database.claim_active_run('u', {'evolution_id': 'e', 'run_token': 't', 'status': 'queued'}, 30, 'web')
        monkeypatch.setattr(worker, 'RUN_HEARTBEAT_SECONDS', .001)
        cancelled = []
        async def blocked_read(*args):
            try: await asyncio.Event().wait()
            finally: cancelled.append(True)
        monkeypatch.setattr(database, 'get_evolution', blocked_read)
        monkeypatch.setattr(database, 'update_active_run', AsyncMock(return_value=False))
        load = AsyncMock()
        monkeypatch.setattr(worker.EvolutionEngine, 'load_evolution_for_user', load)
        with pytest.raises(database.RunOwnershipLost, match='initialization'):
            await asyncio.wait_for(worker.execute_job(RunJob('u', 'e', 't')), 1)
        assert cancelled == [True]
        load.assert_not_called()
    asyncio.run(check())


def test_termination_allows_checkpoint_cleanup():
    from idea.worker import _stop_on_termination
    async def check():
        cleaned = []
        async def operation():
            try: await asyncio.Event().wait()
            finally: cleaned.append('checkpoint')
        task = asyncio.create_task(operation())
        await asyncio.sleep(0)
        exits = []
        await _stop_on_termination(task, grace_seconds=.1, hard_exit=exits.append)
        assert cleaned == ['checkpoint']
        assert not exits
    asyncio.run(check())


def test_termination_deadline_is_bounded_when_cleanup_stalls():
    from idea.worker import _stop_on_termination
    async def check():
        release = asyncio.Event()
        async def operation():
            try: await asyncio.Event().wait()
            except asyncio.CancelledError: await release.wait()
        task = asyncio.create_task(operation())
        await asyncio.sleep(0)
        exits = []
        await _stop_on_termination(task, grace_seconds=.01, hard_exit=exits.append)
        assert exits == [143]
        release.set()
        await task
    asyncio.run(check())


def test_sigterm_signals_provider_guard_before_task_cancellation():
    import threading
    from idea.provider_execution import await_provider_call
    from idea.worker import _stop_on_termination
    async def check():
        started, release = threading.Event(), threading.Event()
        stopped = []
        retries = []
        def provider():
            started.set()
            assert release.wait(timeout=1)
            if not stopped: retries.append('paid retry')
        async def operation(): await await_provider_call(provider)
        task = asyncio.create_task(operation())
        while not started.is_set(): await asyncio.sleep(.001)
        terminating = asyncio.create_task(_stop_on_termination(task, grace_seconds=1,
                                          request_stop=lambda: stopped.append(True)))
        await asyncio.sleep(0)
        assert stopped == [True]
        release.set()
        await terminating
        assert not retries
    asyncio.run(check())
