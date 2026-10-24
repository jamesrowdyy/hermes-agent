"""Per-model-step quota wait-and-resume.

A provider that is *certain* nothing was executed and names when its quota reopens marks the
error with two attributes: ``quota_reset_at`` (epoch seconds) and ``safe_to_resume = True``.
The pending model request alone is parked until then (Stop/redirect cancel it) and re-fired;
never a fallback/model switch, tool replay or user-prompt replay. Anything unmarked is untouched.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger("agent.conversation_loop")
_MARGIN_S = 5.0


def quota_wait_seconds(agent: Any, api_error: Any, _retry: Any) -> Optional[float]:
    """Seconds to wait, or None when the error is not an explicitly resumable quota wait."""
    if getattr(api_error, "safe_to_resume", False) is not True:
        return None
    reset_at = getattr(api_error, "quota_reset_at", None)
    if not isinstance(reset_at, (int, float)) or isinstance(reset_at, bool):
        return None
    cycles = int(getattr(agent, "_quota_wait_cycles", 0) or 0)
    cap = float(getattr(agent, "_quota_wait_max_s", 0) or 0)
    if cycles <= 0 or cap <= 0 or int(getattr(_retry, "quota_waits_used", 0)) >= cycles:
        return None
    if agent._has_content_after_think_block(getattr(agent, "_current_streamed_assistant_text", "") or ""):
        return None  # delivered text is never replayed
    wait = float(reset_at) - time.time() + _MARGIN_S
    return wait if 0 < wait <= cap else None


def wait_for_quota_reset(agent: Any, api_error: Any, _retry: Any, *, messages: Any,
                         conversation_history: Any, api_call_count: int) -> Optional[dict[str, Any]]:
    """``None`` when not applicable; else ``{"action": "continue"|"break"|"return", "result": ...}``."""
    from agent.turn_recovery import interruptible_backoff_sleep
    wait = quota_wait_seconds(agent, api_error, _retry)
    if wait is None:
        return None
    _retry.quota_waits_used = int(getattr(_retry, "quota_waits_used", 0)) + 1
    minutes = max(1, int(wait // 60))
    notice = (f"⏳ Plan quota exhausted — resuming this request automatically in ~{minutes} min "
              f"(same model); press Esc / send /stop to cancel")
    agent._emit_diagnostic_status(notice)
    agent._emit_diagnostic_wait(notice)
    logger.warning("%sQuota wait %.0fs before re-firing pending request %s",
                   agent.log_prefix, wait, agent._client_log_context())
    interrupted = interruptible_backoff_sleep(
        agent, wait, _retry, messages=messages, conversation_history=conversation_history,
        api_call_count=api_call_count,
        abort_message="Interrupt detected during quota wait, aborting.",
        interrupt_text="Operation interrupted: waiting for plan quota to reset.",
        activity_label="quota reset wait",
    )
    if interrupted is not None:
        return {"action": "return", "result": interrupted}
    if _retry.restart_with_redirected_messages:
        return {"action": "break", "result": None}
    return {"action": "continue", "result": None}
