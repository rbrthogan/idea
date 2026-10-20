"""Offline regressions for worker claims, cancellation, and blocking SDK boundaries."""
import asyncio
import copy
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from idea import database, viewer, auth
from idea.evolution_persistence import EvolutionRepository, EvolutionSerializer, EvolutionPersistenceError
from idea.run_service import RunService
from idea.user_state import UserEvolutionState

REAL_GET_ACTIVE_RUN = database.get_active_run


class Snapshot:
    def __init__(self, value):
        self.exists = value is not None
        self.value = copy.deepcopy(value)
    def to_dict(self): return copy.deepcopy(self.value)


class Ref:
    def __init__(self, db, path=()): self.db, self.path = db, path
    def collection(self, value): return Ref(self.db, self.path + (value,))
    def document(self, value): return Ref(self.db, self.path + (value,))
    def get(self, transaction=None): return Snapshot(self.db.docs.get(self.path))
    def set(self, value, merge=False):
        self.db.docs[self.path] = {**(self.db.docs.get(self.path, {}) if merge else {}), **copy.deepcopy(value)}


class FakeDB(Ref):
    def __init__(self):
        self.docs = {}
        super().__init__(self)
    def transaction(self): return self
    def set(self, ref, value, merge=False): ref.set(value, merge)


@pytest.fixture
def fake_db(monkeypatch):
    db = FakeDB()
    monkeypatch.setattr(database, 'get_db', lambda: db)
    monkeypatch.setattr(database, 'get_active_run', REAL_GET_ACTIVE_RUN)
    monkeypatch.setattr(database.firestore, 'transactional', lambda f: f)
    monkeypatch.setattr(database, '_now_ms', lambda: 1000)
    monkeypatch.setattr(viewer, '_now_ms', lambda: 1000)
    monkeypatch.setattr(viewer, 'GLOBAL_MAX_ACTIVE_RUNS', 0)
    return db


def test_same_owner_new_run_rejects_stale_token_and_expired_renewal(fake_db):
    async def check():
        first = await database.claim_active_run('u', {'evolution_id': 'e'}, 20, 'owner')
        token = first['data']['run_token']
        ref = database._run_doc_ref(fake_db, 'u')
        ref.set({'lease_expires_at_ms': 999}, merge=True)
        assert not await database.update_active_run('u', {}, 20, 'owner', run_token=token, require_live_lease=True)
        second = await database.claim_active_run('u', {'evolution_id': 'e2'}, 20, 'owner')
        assert second['data']['run_token'] != token
        before = ref.get().to_dict()
        assert not await database.update_active_run('u', {'status': 'complete'}, 0, 'owner', run_token=token)
        assert ref.get().to_dict() == before
    asyncio.run(check())


def test_active_repair_and_autorating_are_rejected_before_reads_or_paid_calls(fake_db):
    async def check():
        await database.claim_active_run('u', {'evolution_id': 'e'}, 20, 'owner')
        repair = SimpleNamespace(json=AsyncMock(return_value={"data": {
            "evolution_id": "e", "history": [[{"idea": "cached partial state"}]], "current_generation": 1}}))
        rating = SimpleNamespace(json=AsyncMock(return_value={"evolutionId": "e", "numRounds": 1}))
        read = AsyncMock()
        save = AsyncMock()
        with patch.object(database, 'get_evolution', read), patch.object(database, 'save_evolution', save), \
             patch.object(viewer, 'Critic') as critic:
            user = auth.UserInfo(uid='u')
            repaired = await viewer.repair_evolution_endpoint(repair, 'e', user)
            rated = await viewer.auto_rate(rating, user)
        assert repaired.status_code == rated.status_code == 409
        read.assert_not_awaited()
        save.assert_not_awaited()
        critic.assert_not_called()
    asyncio.run(check())


def test_local_graceful_stop_publishes_with_the_owner_token(fake_db):
    async def check():
        claimed = await database.claim_active_run('u', {'evolution_id': 'e'}, 20, 'owner')
        stopped = []
        state = UserEvolutionState(run_owner_id='owner', run_token=claimed['data']['run_token'],
                                   engine=SimpleNamespace(stop_evolution=lambda: stopped.append(True)))
        with patch.object(viewer.user_states, 'get', AsyncMock(return_value=state)):
            response = await viewer.stop_evolution(auth.UserInfo(uid='u'))
        assert response.status_code == 200
        assert stopped == [True]
        claim = database._run_doc_ref(fake_db, 'u').get().to_dict()
        assert claim['stop_requested'] is True
        assert claim['status'] == 'stopping'
    asyncio.run(check())


def test_global_release_and_refresh_are_fenced(fake_db):
    async def check():
        await database.claim_global_run_lock('owner', 'u', 'e', 20, run_token='new')
        assert not await database.release_global_run_lock('owner', run_token='old')
        assert not await database.refresh_global_run_lock('owner', 20, run_token='old')
        assert await database.release_global_run_lock('owner', run_token='new')
    asyncio.run(check())


def test_progress_cannot_erase_remote_stop_and_stale_clear_cannot_expire_live_claim(fake_db):
    async def check():
        await database.claim_active_run('u', {'run_token': 't'}, 20, 'owner')
        assert await database.update_active_run('u', {'status': 'stopping', 'stop_requested': True}, owner_id='owner', run_token='t')
        assert await database.update_active_run('u', {'status': 'in_progress'}, 20, 'owner', run_token='t', require_live_lease=True)
        ref = database._run_doc_ref(fake_db, 'u')
        assert ref.get().to_dict()['status'] == 'stopping'
        assert not await database.update_active_run('u', {'status': 'paused'}, 0, 'owner', run_token='t', require_expired_lease=True)
    asyncio.run(check())


def test_remote_force_stop_waits_for_worker_acknowledgement(fake_db, monkeypatch):
    async def check():
        service = RunService()
        monkeypatch.setattr(viewer, 'run_service', service)
        monkeypatch.setattr(viewer, 'RUN_HEARTBEAT_SECONDS', .001)
        await database.claim_active_run('u', {'run_token': 't', 'evolution_id': 'e'}, 20, 'owner')
        remote = UserEvolutionState()
        monkeypatch.setattr(viewer.user_states, 'get', AsyncMock(return_value=remote))
        owner = UserEvolutionState(run_owner_id='owner', run_token='t')
        engine = SimpleNamespace(user_id='u', checkpoint_id='e', stop_requested=False, is_stopped=False,
                                 _run_token='t', save_checkpoint=AsyncMock())
        engine.stop_evolution = lambda: setattr(engine, 'stop_requested', True)
        owner.engine = engine
        owner.status['is_running'] = True
        async def operation(): await asyncio.Event().wait()
        task = service.start(owner, operation, viewer._finalize_run_state)
        await asyncio.sleep(0)
        response = await viewer.force_stop_evolution(auth.UserInfo('u'))
        assert response.status_code == 202
        current = await database.get_active_run('u')
        assert current['status'] == 'stopping'
        assert current['lease_expires_at_ms'] > 1000
        heartbeat = asyncio.create_task(viewer._heartbeat_loop('u', owner))
        await asyncio.wait_for(task, 1)
        await heartbeat
        current = await database.get_active_run('u')
        assert current['cancel_acknowledged'] is True
        assert current['is_running'] is False
        assert current['lease_expires_at_ms'] == 1000
        engine.save_checkpoint.assert_awaited_once_with(status='paused')
    asyncio.run(check())


def test_database_sdk_does_not_block_event_loop(fake_db, monkeypatch):
    original = Ref.get
    def slow_get(self, **kwargs):
        time.sleep(.04)
        return original(self, **kwargs)
    monkeypatch.setattr(Ref, 'get', slow_get)
    async def check():
        started = time.monotonic()
        task = asyncio.create_task(database.get_active_run('u'))
        await asyncio.sleep(.005)
        assert time.monotonic() - started < .03
        assert not task.done()
        await task
    asyncio.run(check())


def test_checkpoint_failure_is_explicit_and_preserves_revision():
    engine = SimpleNamespace(user_id='u', evolution_id='e', evolution_name='Test', created_at='t', persistence_revision='old')
    with patch.object(EvolutionSerializer, 'to_checkpoint_state', return_value={}), patch.object(database, 'save_evolution', AsyncMock(side_effect=ConnectionError('offline'))):
        with pytest.raises(EvolutionPersistenceError, match='durably saved'):
            asyncio.run(EvolutionRepository.save_checkpoint(engine, 'complete'))
    assert engine.persistence_revision == 'old'
    assert engine.last_save_error


def test_supervisor_observes_failure_and_prevents_second_worker():
    async def check():
        service = RunService()
        state = UserEvolutionState(engine=SimpleNamespace(user_id='u'), run_token='t')
        finalizer = AsyncMock()
        async def operation():
            await asyncio.sleep(.01)
            raise RuntimeError('worker failed')
        task = service.start(state, operation, finalizer)
        with pytest.raises(RuntimeError, match='still active'):
            service.start(state, operation, finalizer)
        await task
        assert state.status['error'] == 'worker failed'
        assert state.status['is_running'] is False
        assert state.run_task is None
        finalizer.assert_awaited_once()
    asyncio.run(check())


def test_auto_rate_duplicate_request_preserves_original_running_state(monkeypatch):
    async def check():
        state = UserEvolutionState()
        state.autorating_status.update({'is_running': True, 'status': 'in_progress', 'progress': 42})
        monkeypatch.setattr(viewer.user_states, 'get', AsyncMock(return_value=state))
        load = AsyncMock()
        monkeypatch.setattr(viewer, '_load_rating_evolution', load)
        request = SimpleNamespace(json=AsyncMock(return_value={'evolutionId': 'e'}))
        response = await viewer.auto_rate(request, auth.UserInfo('u'))
        assert response.status_code == 409
        assert state.autorating_status['is_running'] is True
        assert state.autorating_status['progress'] == 42
        load.assert_not_called()
    asyncio.run(check())


def test_replacing_engine_closes_shared_transports_once():
    from unittest.mock import Mock
    client = SimpleNamespace(close=Mock())
    prior = SimpleNamespace(critic=SimpleNamespace(client=client), ideator=SimpleNamespace(client=client))
    state = UserEvolutionState(engine=prior)
    viewer._replace_state_engine(state, SimpleNamespace())
    client.close.assert_called_once()


@pytest.mark.parametrize("encoder_kind", ["manager_alias", "separate_encoder"])
def test_replacing_engine_closes_gene_analysis_transports_once(encoder_kind):
    from unittest.mock import Mock

    manager_client = SimpleNamespace(close=Mock())
    encoder_client = (manager_client if encoder_kind == "manager_alias"
                      else SimpleNamespace(close=Mock()))
    manager = SimpleNamespace(client=manager_client)
    encoder = (manager if encoder_kind == "manager_alias"
               else SimpleNamespace(_client=encoder_client))
    prior = SimpleNamespace(
        gene_manager=manager,
        breeder=SimpleNamespace(gene_encoder=encoder),
    )
    state = UserEvolutionState(engine=prior)

    viewer._replace_state_engine(state, SimpleNamespace())

    manager_client.close.assert_called_once()
    encoder_client.close.assert_called_once()


def test_renewal_rechecks_clock_after_slow_transaction_read(fake_db, monkeypatch):
    async def check():
        await database.claim_active_run('u', {'run_token': 't'}, 1, 'owner')
        clock = [1000]
        original_get = Ref.get
        def advancing_get(ref, **kwargs):
            snapshot = original_get(ref, **kwargs)
            clock[0] = 2001
            return snapshot
        monkeypatch.setattr(database, '_now_ms', lambda: clock[0])
        monkeypatch.setattr(Ref, 'get', advancing_get)
        assert not await database.update_active_run('u', {}, 30, 'owner', run_token='t', require_live_lease=True)
    asyncio.run(check())


def test_claim_lease_uses_clock_after_transaction_read(fake_db, monkeypatch):
    async def check():
        clock = [1000]
        original_get = Ref.get
        def advancing_get(ref, **kwargs):
            snapshot = original_get(ref, **kwargs)
            clock[0] = 5000
            return snapshot
        monkeypatch.setattr(database, '_now_ms', lambda: clock[0])
        monkeypatch.setattr(Ref, 'get', advancing_get)
        result = await database.claim_active_run('u', {'run_token': 't'}, 1, 'owner')
        assert result['data']['lease_expires_at_ms'] == 6000
    asyncio.run(check())


def test_service_shutdown_signals_provider_guard_before_draining_thread():
    import threading
    from idea.provider_execution import await_provider_call
    async def check():
        started, release = threading.Event(), threading.Event()
        retries = []
        engine = SimpleNamespace(user_id='u', checkpoint_id='e', stop_requested=False,
                                 save_checkpoint=AsyncMock())
        engine.stop_evolution = lambda: setattr(engine, 'stop_requested', True)
        state = UserEvolutionState(engine=engine)
        service = RunService()
        def provider():
            started.set()
            assert release.wait(timeout=1)
            if not engine.stop_requested: retries.append('paid retry')
        async def operation(): await await_provider_call(provider)
        service.start(state, operation, AsyncMock())
        while not started.is_set(): await asyncio.sleep(.001)
        closing = asyncio.create_task(service.close())
        await asyncio.sleep(0)
        assert engine.stop_requested is True
        release.set()
        await closing
        assert retries == []
        engine.save_checkpoint.assert_awaited_once_with(status='paused')
    asyncio.run(check())


def test_service_shutdown_before_supervisor_start_prevents_operation():
    async def check():
        engine = SimpleNamespace(user_id='u', checkpoint_id='e', stop_requested=False, save_checkpoint=AsyncMock())
        state = UserEvolutionState(engine=engine)
        service = RunService()
        operation = AsyncMock()
        service.start(state, operation, AsyncMock())
        await service.close()
        operation.assert_not_awaited()
        engine.save_checkpoint.assert_awaited_once_with(status='paused')
    asyncio.run(check())
