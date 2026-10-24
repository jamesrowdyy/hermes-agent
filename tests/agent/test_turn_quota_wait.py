"""Per-model-step quota wait-and-resume: a provider error that carries typed reset metadata
(``quota_reset_at``, epoch seconds) parks only the pending LLM request until the reset, then
re-fires that same request. No fallback/model switch, no replayed tools, Stop cancels."""
import time
from types import SimpleNamespace
from unittest.mock import patch

from agent.error_classifier import classify_api_error
from agent.turn_recovery import route_classified_error
from agent.turn_retry_state import TurnRetryState


class _QuotaErr(Exception):
    def __init__(self, reset_in=300.0, resumable=True):
        super().__init__("Claude plan limit reached on every configured seat; resets in 5min")
        self.status_code = 429
        self.response = SimpleNamespace(headers={})
        self.body = None
        if resumable:
            self.quota_reset_at = time.time() + reset_in
            self.safe_to_resume = True


class _Agent:
    log_prefix = ""
    verbose = False
    provider = "claude-subscription-directsdk-experimental"
    platform = "cli"
    _credential_pool = None
    _current_streamed_assistant_text = ""
    _interrupt_requested = False
    _quota_wait_max_s = 6 * 3600.0
    _quota_wait_cycles = 3

    def __init__(self, fallback_left=True):
        self.statuses, self.waits, self.activated = [], [], []
        self._fallback_chain = [object()] if fallback_left else []
        self._fallback_index = 0

    def _has_pending_fallback(self):
        return bool(self._fallback_chain)

    def _try_activate_fallback(self, **kw):
        self.activated.append(kw)
        return True

    def _has_content_after_think_block(self, text):
        return bool(text.strip())

    def _emit_diagnostic_status(self, t):
        self.statuses.append(str(t))

    def _emit_diagnostic_wait(self, t):
        self.waits.append(str(t))

    def _buffer_diagnostic_status(self, t):
        pass

    def _touch_activity(self, *a, **k):
        pass

    def clear_interrupt(self, **kw):
        return False

    def _client_log_context(self):
        return ""

    def __getattr__(self, name):
        return lambda *a, **k: None


def _route(agent, err, retry=None, sleep=lambda a, w, *x, **k: None, retry_count=1):
    retry = retry or TurnRetryState(primary_recovery_attempted=True)
    classified = classify_api_error(err, provider=agent.provider)
    with patch("agent.turn_recovery.interruptible_backoff_sleep", sleep), \
            patch("agent.conversation_loop._arm_fallback_restart", lambda a, m, s, r: s):
        return route_classified_error(
            agent, err, classified, retry, error_msg=str(err).lower(), error_context={},
            recovered_with_pool=False, base_url="x", model="m", messages=[], api_messages=[],
            system_message=None, active_system_prompt=None, conversation_history=None,
            retry_count=retry_count, max_retries=3, compression_attempts=0,
            max_compression_attempts=3, api_call_count=1, effective_task_id=None,
        ), retry


def test_quota_error_waits_until_reset_and_reenters_without_fallback():
    agent, waited = _Agent(fallback_left=True), []
    verdict, retry = _route(agent, _QuotaErr(300), sleep=lambda a, w, *x, **k: waited.append(w))
    assert verdict.action == "continue" and verdict.result is None
    assert 299 <= waited[0] <= 310          # earliest reset (+ small margin)
    assert agent.activated == []            # no silent model/provider switch
    assert retry.quota_waits_used == 1
    assert verdict.retry_count == 1         # the wait does not burn a retry
    assert any("Claude" in s or "quota" in s.lower() or "reset" in s.lower() for s in agent.statuses)
    assert "Stop" in agent.statuses[0] or "stop" in agent.statuses[0]


def test_stop_during_quota_wait_cancels_pending_retry():
    agent = _Agent()
    agent._interrupt_requested = True
    agent._drop_trailing_empty_response_scaffolding = lambda m: None
    agent._persist_session = lambda *a, **k: None
    agent.clear_interrupt = lambda **kw: False
    # real interruptible_backoff_sleep: honours Stop at once
    classified_err = _QuotaErr(3600)
    retry = TurnRetryState(primary_recovery_attempted=True)
    classified = classify_api_error(classified_err, provider=agent.provider)
    t0 = time.time()
    with patch("agent.conversation_loop._arm_fallback_restart", lambda a, m, s, r: s):
        verdict = route_classified_error(
            agent, classified_err, classified, retry, error_msg="x", error_context={},
            recovered_with_pool=False, base_url="x", model="m", messages=[], api_messages=[],
            system_message=None, active_system_prompt=None, conversation_history=None,
            retry_count=1, max_retries=3, compression_attempts=0, max_compression_attempts=3,
            api_call_count=1, effective_task_id=None)
    assert time.time() - t0 < 2
    assert verdict.action == "return" and verdict.result["interrupted"] is True


def test_redirect_during_quota_wait_rebuilds_instead_of_refiring():
    agent = _Agent()

    def sleep(a, w, retry, **k):
        retry.restart_with_redirected_messages = True
    verdict, _ = _route(agent, _QuotaErr(300), sleep=sleep)
    assert verdict.action == "break"


def test_unknown_error_without_typed_metadata_is_not_waited_on():
    agent, waited = _Agent(fallback_left=True), []
    verdict, retry = _route(agent, _QuotaErr(resumable=False), sleep=lambda a, w, *x, **k: waited.append(w))
    assert waited == [] and retry.quota_waits_used == 0
    assert verdict.action in ("break", "fallthrough")  # existing behaviour untouched


def test_delivered_partial_text_is_never_resumed():
    agent, waited = _Agent(), []
    agent._current_streamed_assistant_text = "half an answer"
    _route(agent, _QuotaErr(300), sleep=lambda a, w, *x, **k: waited.append(w))
    assert waited == []


def test_reset_beyond_cap_or_disabled_or_spent_does_not_wait():
    for mutate, err in (
        (lambda a: None, _QuotaErr(7 * 3600)),
        (lambda a: setattr(a, "_quota_wait_max_s", 0), _QuotaErr(300)),
        (lambda a: setattr(a, "_quota_wait_cycles", 0), _QuotaErr(300)),
    ):
        agent, waited = _Agent(), []
        mutate(agent)
        _route(agent, err, sleep=lambda a, w, *x, **k: waited.append(w))
        assert waited == []
    agent, waited = _Agent(), []
    spent = TurnRetryState(primary_recovery_attempted=True, quota_waits_used=3)
    _route(agent, _QuotaErr(300), retry=spent, sleep=lambda a, w, *x, **k: waited.append(w))
    assert waited == []


def test_reset_time_without_safe_flag_is_not_waited_on():
    agent, waited = _Agent(), []
    err = _QuotaErr(300)
    err.safe_to_resume = False
    _route(agent, err, sleep=lambda a, w, *x, **k: waited.append(w))
    assert waited == []
