#!/usr/bin/env python3
"""
Persistent IPython kernel with Hermes tool access (RLM-style).

Like execute_code but state persists across calls. The kernel stays alive
for the session, so imports, variables, loaded data, and tool results
survive between turns.

Architecture:
  1. On first call, start a Jupyter kernel + UDS RPC socket (same as execute_code)
  2. Inject hermes_tools stubs into kernel namespace as a proper sys.modules entry
  3. Subsequent calls reuse the same kernel — state persists
  4. Per-kernel lock serializes concurrent access
  5. Idle timeout + atexit + AIAgent.close() cleanup kills kernels
  6. Output: truncated, ANSI-stripped, secret-redacted (same as execute_code)
"""
import atexit
import json
import logging
import os
import platform
import secrets
import socket
import threading
import time
from typing import Any, Dict, List, Optional

from tools.registry import registry, tool_error
from tools.code_execution_tool import (
    generate_hermes_tools_module,
    SANDBOX_ALLOWED_TOOLS,
    _TERMINAL_BLOCKED_PARAMS,
    _truncate_stdout_text,
    MAX_STDERR_BYTES,
)
from tools.ansi_strip import strip_ansi
from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__)
_IS_WINDOWS = platform.system() == "Windows"

# ---------------------------------------------------------------------------
# Session-scoped kernel registry
# ---------------------------------------------------------------------------

_kernel_sessions: Dict[str, dict] = {}
_kernel_lock = threading.Lock()
_cleanup_running = True
_cleanup_thread: Optional[threading.Thread] = None
_KERNEL_IDLE_TIMEOUT = 600
_KERNEL_MEMORY_LIMIT_MB = 500


def _cleanup_kernel(task_id: str):
    """Kill and remove a single kernel session. Best-effort."""
    with _kernel_lock:
        session = _kernel_sessions.pop(task_id, None)
    if session is None:
        return
    for closer in (
        lambda: session["kc"].stop_channels(),
        lambda: session["km"].shutdown_kernel(now=True),
    ):
        try:
            closer()
        except Exception:
            pass
    sock = session.get("server_sock")
    if sock is not None:
        try:
            sock.close()
        except Exception:
            pass
    import shutil
    tmpdir = session.get("tmpdir")
    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)
    logger.info("Kernel for task %s cleaned up", task_id)


def _cleanup_inactive_kernels(timeout_seconds: int = _KERNEL_IDLE_TIMEOUT):
    """Kill kernels idle for longer than timeout."""
    now = time.time()
    stale = []
    with _kernel_lock:
        for task_id, session in list(_kernel_sessions.items()):
            if now - session["last_used"] > timeout_seconds:
                stale.append(task_id)
    for task_id in stale:
        logger.info("Kernel for task %s idle >%ds, shutting down", task_id, timeout_seconds)
        _cleanup_kernel(task_id)


def _cleanup_thread_worker():
    while _cleanup_running:
        try:
            _cleanup_inactive_kernels()
        except Exception as e:
            logger.warning("Kernel cleanup thread error: %s", e)
        for _ in range(60):
            if not _cleanup_running:
                break
            time.sleep(1)


def _ensure_cleanup_thread():
    global _cleanup_thread
    if _cleanup_thread is None or not _cleanup_thread.is_alive():
        _cleanup_thread = threading.Thread(target=_cleanup_thread_worker, daemon=True)
        _cleanup_thread.start()


def _atexit_cleanup():
    global _cleanup_running
    _cleanup_running = False
    with _kernel_lock:
        task_ids = list(_kernel_sessions.keys())
    for task_id in task_ids:
        _cleanup_kernel(task_id)


atexit.register(_atexit_cleanup)


# ---------------------------------------------------------------------------
# RPC server (same protocol as execute_code, single persistent connection)
# ---------------------------------------------------------------------------

def _rpc_server_loop(
    server_sock, task_id, tool_call_counter, max_tool_calls,
    allowed_tools, stop_event, rpc_token, tool_call_log,
):
    """Accept one client and dispatch tool calls until disconnect or stop."""
    from model_tools import handle_function_call
    from agent.thread_scoped_output import thread_scoped_silence

    conn = None
    try:
        server_sock.settimeout(0.05)
        while not stop_event.is_set():
            try:
                conn, _ = server_sock.accept()
                break
            except socket.timeout:
                continue
        if conn is None:
            return
        conn.settimeout(300)
        buf = b""
        while True:
            try:
                chunk = conn.recv(65536)
            except socket.timeout:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    req = json.loads(line.decode())
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    conn.sendall((tool_error(f"Invalid RPC request: {exc}") + "\n").encode())
                    continue
                if not rpc_token or not secrets.compare_digest(
                    str(req.get("token") or "").encode(), rpc_token.encode()
                ):
                    conn.sendall((tool_error("Unauthorized RPC request") + "\n").encode())
                    continue
                tool_name = req.get("tool", "")
                tool_args = req.get("args", {})
                if tool_name not in allowed_tools:
                    conn.sendall((tool_error(
                        f"Tool '{tool_name}' not available in kernel."
                    ) + "\n").encode())
                    continue
                if tool_call_counter[0] >= max_tool_calls:
                    conn.sendall((tool_error(
                        "Tool call limit reached in kernel."
                    ) + "\n").encode())
                    continue
                if tool_name == "terminal" and isinstance(tool_args, dict):
                    for param in _TERMINAL_BLOCKED_PARAMS:
                        tool_args.pop(param, None)
                tool_call_counter[0] += 1
                call_start = time.monotonic()
                try:
                    with thread_scoped_silence():
                        result = handle_function_call(
                            tool_name, tool_args, task_id=task_id
                        )
                except Exception as exc:
                    logger.error("Kernel RPC tool call failed: %s", exc, exc_info=True)
                    result = tool_error(str(exc))
                # Audit log: record tool name + args preview + duration.
                # Surfaced in the kernel_execute result so transcript review
                # (Codex, audit) can see exactly what the kernel did.
                tool_call_log.append({
                    "tool": tool_name,
                    "args_preview": str(tool_args)[:80],
                    "duration": round(time.monotonic() - call_start, 2),
                })
                conn.sendall((result + "\n").encode())
    except Exception as e:
        logger.debug("Kernel RPC server error: %s", e)
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Kernel creation + stub injection
# ---------------------------------------------------------------------------

def _check_kernel_memory(km) -> bool:
    """Return True if kernel RSS is under the memory limit, False to restart."""
    try:
        import psutil
        info = km.get_connection_info()
        pid = info.get("pid") if isinstance(info, dict) else None
        if pid:
            rss_mb = psutil.Process(pid).memory_info().rss / (1024 * 1024)
            return rss_mb <= _KERNEL_MEMORY_LIMIT_MB
    except Exception:
        pass
    return True  # can't check — assume OK


def _get_or_create_kernel(task_id, enabled_tools):
    """Get existing kernel for this session, or create one."""
    with _kernel_lock:
        if task_id in _kernel_sessions:
            session = _kernel_sessions[task_id]
            session["last_used"] = time.time()
            # Auto-restart if the kernel has grown past the memory limit
            if not _check_kernel_memory(session["km"]):
                logger.warning(
                    "Kernel for %s exceeded %dMB, restarting",
                    task_id, _KERNEL_MEMORY_LIMIT_MB,
                )
                _cleanup_kernel(task_id)
            else:
                return session
        _ensure_cleanup_thread()

    from jupyter_client import KernelManager
    import tempfile

    km = KernelManager()
    km.start_kernel()
    kc = km.client()
    kc.start_channels()
    kc.wait_for_ready(timeout=30)

    # Set up UDS RPC socket
    tmpdir = tempfile.mkdtemp(prefix="kernel_rpc_")
    rpc_token = secrets.token_hex(16)

    if _IS_WINDOWS:
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.bind(("127.0.0.1", 0))
        rpc_endpoint = f"tcp://127.0.0.1:{server_sock.getsockname()[1]}"
    else:
        sock_path = os.path.join(tmpdir, "rpc.sock")
        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_sock.bind(sock_path)
        rpc_endpoint = sock_path

    server_sock.listen(1)

    # Generate + inject stubs
    stubs_code = generate_hermes_tools_module(
        sorted(SANDBOX_ALLOWED_TOOLS & set(enabled_tools)),
        transport="uds",
    )
    stubs_modified = stubs_code.replace(
        'os.environ["HERMES_RPC_SOCKET"]', f'"{rpc_endpoint}"',
    ).replace(
        'os.environ.get("HERMES_RPC_TOKEN", "")', f'"{rpc_token}"',
    )

    # Register as proper importable module in kernel namespace
    injection_code = (
        "import sys, types\n"
        "_ht = types.ModuleType('hermes_tools')\n"
        f"exec({repr(stubs_modified)}, _ht.__dict__)\n"
        "sys.modules['hermes_tools'] = _ht\n"
    )
    kc.execute(injection_code, silent=True)
    while True:
        try:
            msg = kc.get_iopub_msg(timeout=10)
        except Exception:
            break
        if msg.get("msg_type") == "status" and msg.get("content", {}).get("execution_state") == "idle":
            break
    try:
        kc.get_shell_msg(timeout=5)
    except Exception:
        pass

    # Start RPC server thread
    stop_event = threading.Event()
    tool_call_counter = [0]
    tool_call_log = []  # audit trail of every tool call the kernel made
    rpc_thread = threading.Thread(
        target=_rpc_server_loop,
        args=(server_sock, task_id, tool_call_counter, 50,
              frozenset(SANDBOX_ALLOWED_TOOLS & set(enabled_tools)),
              stop_event, rpc_token, tool_call_log),
        daemon=True,
    )
    rpc_thread.start()

    session = {
        "km": km, "kc": kc, "lock": threading.Lock(),
        "last_used": time.time(), "task_id": task_id,
        "server_sock": server_sock, "rpc_thread": rpc_thread,
        "stop_event": stop_event, "tool_call_counter": tool_call_counter,
        "tool_call_log": tool_call_log,
        "tmpdir": tmpdir,
    }
    with _kernel_lock:
        _kernel_sessions[task_id] = session
    logger.info("Created persistent kernel for task %s", task_id)
    return session


# ---------------------------------------------------------------------------
# kernel_execute — the tool handler
# ---------------------------------------------------------------------------

def kernel_execute(code, task_id=None, enabled_tools=None, timeout=120, restart=False):
    """Execute Python in the persistent kernel. State survives across calls.
    
    If restart=True, kill and recreate the kernel first (drops all state).
    """
    if not task_id:
        return tool_error("kernel_execute requires a session context (task_id)")

    try:
        if restart:
            _cleanup_kernel(task_id)
        session = _get_or_create_kernel(task_id, enabled_tools or list(SANDBOX_ALLOWED_TOOLS))
    except ImportError:
        return tool_error(
            "kernel_execute requires jupyter_client + ipykernel. "
            "Install with: pip install jupyter_client ipykernel"
        )
    except Exception as exc:
        return tool_error(f"Failed to start kernel: {exc}")

    kc = session["kc"]
    km = session["km"]
    exec_start = time.monotonic()

    try:
        # Interrupt check setup
        try:
            from tools.interrupt import is_interrupted as _is_interrupted
            from tools.interrupt import clear_current_thread_interrupt
        except ImportError:
            _is_interrupted = lambda: False
            clear_current_thread_interrupt = None

        if clear_current_thread_interrupt:
            try:
                clear_current_thread_interrupt()
            except Exception:
                pass

        # Activity touch (keep gateway alive)
        try:
            from tools.environments.base import touch_activity_if_due
        except Exception:
            touch_activity_if_due = None

        # Serialize access to this kernel
        with session["lock"]:
            msg_id = kc.execute(code, store_history=False)

            stdout_chunks, stderr_chunks = [], []
            got_reply = False
            status = "success"
            _activity_state = {"last_touch": exec_start, "start": exec_start}
            deadline = time.time() + timeout

            while time.time() < deadline:
                if _is_interrupted():
                    try:
                        km.interrupt_kernel()
                    except Exception:
                        pass
                    status = "interrupted"
                    break
                if touch_activity_if_due is not None:
                    try:
                        touch_activity_if_due(_activity_state, "kernel_execute running")
                    except Exception:
                        pass
                try:
                    msg = kc.get_iopub_msg(timeout=0.5)
                except Exception:
                    if not got_reply:
                        try:
                            reply = kc.get_shell_msg(timeout=0.5)
                            if reply and reply.get("parent_header", {}).get("msg_id") == msg_id:
                                got_reply = True
                        except Exception:
                            pass
                    if got_reply:
                        break
                    continue

                mt = msg.get("msg_type", "")
                ct = msg.get("content", {})
                if msg.get("parent_header", {}).get("msg_id") != msg_id:
                    continue
                if mt == "stream":
                    if ct.get("name") == "stderr":
                        stderr_chunks.append(ct.get("text", ""))
                    else:
                        stdout_chunks.append(ct.get("text", ""))
                elif mt == "execute_result":
                    stdout_chunks.append(ct.get("data", {}).get("text/plain", ""))
                elif mt == "error":
                    stderr_chunks.append("\n".join(ct.get("traceback", [])))
                elif mt == "status" and ct.get("execution_state") == "idle":
                    if not got_reply:
                        try:
                            kc.get_shell_msg(timeout=2)
                        except Exception:
                            pass
                    break

            if status != "interrupted" and time.time() >= deadline:
                try:
                    km.interrupt_kernel()
                except Exception:
                    pass
                status = "timeout"

            session["last_used"] = time.time()
            duration = round(time.monotonic() - exec_start, 2)

            # Assemble + process output
            stdout_text = strip_ansi("".join(stdout_chunks))
            stderr_text = strip_ansi("".join(stderr_chunks))
            stdout_text = redact_sensitive_text(stdout_text, code_file=True)
            stderr_text = redact_sensitive_text(stderr_text, code_file=True)
            stdout_text, stdout_metadata = _truncate_stdout_text(stdout_text)
            stderr_text = stderr_text[:MAX_STDERR_BYTES]

            result = {
                "status": status,
                "output": stdout_text,
                "stderr": stderr_text,
                "tool_calls_made": session["tool_call_counter"][0],
                "tool_call_log": list(session["tool_call_log"]),
                "duration_seconds": duration,
            }
            result.update(stdout_metadata)

            if status == "timeout":
                msg = f"Kernel execution timed out after {timeout}s and was interrupted."
                result["error"] = msg
                result["output"] = (stdout_text + f"\n\n⏰ {msg}") if stdout_text else f"⏰ {msg}"
            elif status == "interrupted":
                result["output"] = stdout_text + "\n[execution interrupted — user sent a new message]"

            return json.dumps(result, ensure_ascii=False)

    except Exception as exc:
        duration = round(time.monotonic() - exec_start, 2)
        logger.error("kernel_execute failed after %ss: %s", duration, exc, exc_info=True)
        return json.dumps({
            "status": "error",
            "error": str(exc),
            "duration_seconds": duration,
        }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Public API for cleanup (called from AIAgent.close())
# ---------------------------------------------------------------------------

def cleanup_all_kernels():
    """Kill all active kernels. Called from atexit."""
    with _kernel_lock:
        task_ids = list(_kernel_sessions.keys())
    for task_id in task_ids:
        _cleanup_kernel(task_id)


def cleanup_session_kernel(task_id: str):
    """Kill the kernel for a specific session. Called from AIAgent.close()."""
    _cleanup_kernel(task_id)


def get_kernel_status():
    """Return status of all active kernels (for /kernel command)."""
    with _kernel_lock:
        return [
            {
                "task_id": task_id,
                "last_used": session["last_used"],
                "tool_calls": session["tool_call_counter"][0],
                "alive": session["km"].is_alive(),
            }
            for task_id, session in _kernel_sessions.items()
        ]


def restart_kernel(task_id, enabled_tools=None):
    """Restart the kernel for a session (drops all state)."""
    _cleanup_kernel(task_id)
    if enabled_tools:
        _get_or_create_kernel(task_id, enabled_tools)
    return {"status": "restarted", "task_id": task_id}


def list_kernel_variables(task_id):
    """List variables in the kernel namespace (for /kernel command)."""
    with _kernel_lock:
        session = _kernel_sessions.get(task_id)
    if not session:
        return {"error": "No active kernel for this session"}
    kc = session["kc"]
    with session["lock"]:
        msg_id = kc.execute(
            "import sys, json\n"
            "_vars = {k: type(v).__name__ for k, v in globals().items()\n"
            "         if not k.startswith('_') and k not in ('In', 'Out', 'exit', 'quit', 'get_ipython')}\n"
            "print(json.dumps(_vars))\n",
            silent=False,
        )
        while True:
            try:
                msg = kc.get_iopub_msg(timeout=5)
            except Exception:
                break
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            mt = msg.get("msg_type", "")
            ct = msg.get("content", {})
            if mt == "stream":
                import json as _json
                text = ct.get("text", "").strip()
                if text:
                    try:
                        return _json.loads(text)
                    except _json.JSONDecodeError:
                        pass
            elif mt == "execute_result":
                import json as _json
                text = ct.get("data", {}).get("text/plain", "").strip()
                if text:
                    try:
                        return _json.loads(text)
                    except _json.JSONDecodeError:
                        pass
            elif mt == "status" and ct.get("execution_state") == "idle":
                break
    return {}


# ---------------------------------------------------------------------------
# Requirements check + schema + registration
# ---------------------------------------------------------------------------

def check_kernel_requirements():
    try:
        import jupyter_client  # noqa: F401
        import ipykernel  # noqa: F401
        return True
    except ImportError:
        # Try lazy-install before declaring unavailable
        try:
            from tools.lazy_deps import ensure
            ensure("exec.kernel", prompt=False)
        except Exception:
            pass
        try:
            import jupyter_client  # noqa: F401
            import ipykernel  # noqa: F401
            return True
        except ImportError:
            return False


_SCHEMA_DESCRIPTION = (
    "Execute Python in a PERSISTENT IPython kernel with Hermes tool access. "
    "State (variables, imports, loaded data, tool results) survives across calls — "
    "unlike execute_code which is stateless. Use this for multi-step work where "
    "you need to build up state incrementally, or when you need to compose tools "
    "with loops, branches, and conditional logic across multiple calls. "
    "Available tools via `from hermes_tools import ...`: web_search, web_extract, "
    "read_file, write_file, search_files, patch, terminal. "
    "Print results to stdout. "
    "Limits: 120s timeout, 50KB stdout cap, max 50 tool calls per session. "
    "Use /kernel to inspect, clear, or restart the kernel."
)

registry.register(
    name="kernel_execute",
    toolset="code_execution",
    schema={
        "name": "kernel_execute",
        "description": _SCHEMA_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute in the persistent kernel.",
                },
                "restart": {
                    "type": "boolean",
                    "description": "If true, restart the kernel first (drops all state). Use when previous state is stale or causing errors.",
                    "default": False,
                },
                "timeout": {
                    "type": "integer",
                    "description": "Max seconds to wait (default 120).",
                    "default": 120,
                },
            },
            "required": ["code"],
        },
    },
    handler=lambda args, **kw: kernel_execute(
        code=args.get("code", ""),
        task_id=kw.get("task_id"),
        enabled_tools=kw.get("enabled_tools"),
        timeout=args.get("timeout", 120),
        restart=args.get("restart", False),
    ),
    check_fn=check_kernel_requirements,
    emoji="🔬",
    max_result_size_chars=100_000,
)
