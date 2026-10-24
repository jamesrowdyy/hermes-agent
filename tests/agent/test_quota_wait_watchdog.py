"""Five-hour virtual waits exercise real backoff heartbeats and the gateway watchdog."""
import threading
from types import SimpleNamespace

from tests.agent.test_turn_quota_wait import _Agent, _QuotaErr
from agent.turn_retry_state import TurnRetryState


def test_five_hour_quota_wait_keeps_the_real_inactivity_watchdog_alive(monkeypatch):
    import agent.turn_quota_wait as quota
    import agent.turn_recovery as recovery
    import agent.session_activity as activity
    import gateway.run as gateway

    now, last_touch, samples = [1000.0], [1000.0], []
    agent = _Agent()
    agent._touch_activity = lambda *a, **kw: last_touch.__setitem__(0, now[0])

    def sleep(seconds):
        now[0] += seconds
        samples.append((now[0], now[0] - last_touch[0]))

    clock = SimpleNamespace(time=lambda: now[0], monotonic=lambda: now[0], sleep=sleep)
    monkeypatch.setattr(quota, 'time', clock)
    monkeypatch.setattr(recovery, 'time', clock)
    monkeypatch.setattr(quota.random, 'uniform', lambda *a: 5)
    error = _QuotaErr()
    error.quota_reset_at = now[0] + 5 * 3600
    result = quota.wait_for_quota_reset(agent, error, TurnRetryState(), messages=[], conversation_history=None, api_call_count=1)
    assert result == {'action': 'continue', 'result': None}
    assert now[0] >= 1000 + 5 * 3600
    assert max(idle for _, idle in samples) < 31
    assert agent._quota_wait_active is False

    # Feed the observed backoff heartbeat history into the real watchdog, not a
    # guessed timeout formula. There is no wall-clock five-hour sleep or provider I/O.
    readings = iter(samples[::25])
    idle, fired = [0.0], []
    now[0] = 1000
    monkeypatch.setattr(agent, 'get_activity_summary', lambda: {'seconds_since_activity': idle[0]}, raising=False)

    class Done(threading.Event):
        def wait(self, timeout=None):
            try:
                now[0], idle[0] = next(readings)
                return False
            except StopIteration:
                return True

    monkeypatch.setattr(gateway, 'time', clock)
    monkeypatch.setattr(activity, 'time', clock)
    monkeypatch.setattr(gateway, '_abandon_timed_out_gateway_turn', lambda **kw: fired.append(kw))
    gateway._watch_gateway_turn_inactivity(agent_holder=[agent], task_id='quota-test', process_baseline=None,
        timeout=1800, worker_done=Done(), timeout_fired=threading.Event(), cleanup_lock=threading.Lock())
    assert fired == []
