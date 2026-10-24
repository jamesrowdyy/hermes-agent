"""A manual model pick cancels a quota wait, but never an ordinary active turn."""
import asyncio
from types import SimpleNamespace

import pytest

from tests.gateway.test_session_mutation_model_busy import _owner, _mutate, _running_model
from tests.waiting import async_wait_until


@pytest.mark.asyncio
@pytest.mark.parametrize('quota_wait', [True, False])
async def test_model_pick_cancels_only_a_quota_wait(tmp_path, monkeypatch, quota_wait):
    from gateway.session_contract import Submission
    import hermes_state_runtime as rt

    started, release = asyncio.Event(), asyncio.Event()
    interrupted = []

    async def handle(event):
        started.set()
        await release.wait()
        return 'stopped' if interrupted else 'finished'

    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    agent = SimpleNamespace(_quota_wait_active=quota_wait)
    monkeypatch.setattr(authority, 'agent', lambda _ref: agent)

    async def interrupt(actor, target, generation):
        assert target == ref
        assert generation == db.get_session(ref.session_id)['runtime_generation']
        interrupted.append(generation)
        release.set()
    monkeypatch.setattr(authority, 'interrupt', interrupt)
    try:
        turn = await authority.submit(owner.actor, Submission('run', ref, {'text': 'RUN'}, 'queue'))
        await asyncio.wait_for(started.wait(), 10)
        reply = await _mutate(owner, db, ref, 'pick', 'picked')
        assert reply['result']['status'] == 'deferred'
        assert bool(interrupted) is quota_wait
        release.set()
        await async_wait_until(lambda: rt.get_session_admission(db, admission_id=turn.admission_id)['status'] == 'terminal', timeout=10)
        await async_wait_until(lambda: _running_model(authority, ref) == 'picked', timeout=10)
    finally:
        release.set()
        await owner.close()
        store.close_all_db_handles()
