"""Starts and supervises the Reachy Mini MuJoCo simulator daemon.

Why not just `ReachyMini(spawn_daemon=True, use_sim=True)`?
  * The SDK launches the daemon and connects immediately; on Windows MuJoCo often needs longer
    than the client timeout, so the connection fails while the daemon keeps running orphaned.
  * The daemon gives its MuJoCo backend only 2 s to become ready. A cold start (viewer window +
    physics warm-up) regularly misses that, leaving the daemon in an ERROR state (the red
    "error occurred" dot on the http://127.0.0.1:8000 dashboard). A second start is usually fast.
  * The daemon's uvicorn logs would flood this app's console.

So we launch `reachy-mini-daemon --sim` ourselves with its logs going to logs/daemon.log, poll
its /api/daemon/status endpoint, and kill + relaunch it if it lands in ERROR.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

STATUS_URL = "http://127.0.0.1:8000/api/daemon/status"
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


def daemon_status(timeout: float = 1.0) -> dict | None:
    """The local daemon's status JSON, or None if no daemon is listening."""
    try:
        with urllib.request.urlopen(STATUS_URL, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _daemon_command(headless: bool) -> list[str]:
    exe_name = "reachy-mini-daemon.exe" if os.name == "nt" else "reachy-mini-daemon"
    candidates = [Path(sys.executable).parent / exe_name, Path(sys.executable).parent / "Scripts" / exe_name]
    exe = next((str(p) for p in candidates if p.exists()), None) or shutil.which("reachy-mini-daemon")
    cmd = [exe] if exe else [sys.executable, "-m", "reachy_mini.daemon.app.main"]
    cmd.append("--sim")
    if headless:
        cmd.append("--headless")
    return cmd


def kill_local_daemons() -> int:
    """Kill any reachy-mini daemon processes on this machine (used to recover from ERROR)."""
    try:
        import psutil  # dependency of reachy_mini
    except ImportError:
        return 0
    killed = 0
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info.get("cmdline") or [])
            if proc.pid != os.getpid() and ("reachy-mini-daemon" in cmdline or "reachy_mini.daemon.app.main" in cmdline):
                for child in proc.children(recursive=True):
                    child.kill()
                proc.kill()
                killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    if killed:
        time.sleep(2.0)  # let the OS free port 8000
    return killed


class SimDaemon:
    def __init__(self, headless: bool = False, start_timeout_s: float = 60.0, attempts: int = 3) -> None:
        self.headless = headless
        self.start_timeout_s = start_timeout_s
        self.attempts = attempts
        self._proc: subprocess.Popen | None = None
        self._log_file = None

    def ensure_running(self) -> bool:
        """Make sure a healthy simulation daemon is serving on localhost:8000."""
        for attempt in range(1, self.attempts + 1):
            st = daemon_status()
            if st is not None and st.get("state") == "running" and st.get("simulation_enabled"):
                log.info("Simulation daemon is running.")
                return True
            if st is not None:
                # A daemon is there but unhealthy (ERROR) or not a simulator: replace it.
                log.warning("Existing daemon state=%s sim=%s error=%s -> restarting it.",
                            st.get("state"), st.get("simulation_enabled"), st.get("error"))
                self.stop()
                kill_local_daemons()

            self._launch()
            outcome = self._wait_until_ready()
            if outcome == "running":
                return True
            log.warning("Simulator attempt %d/%d failed (%s). Retrying...", attempt, self.attempts, outcome)
            self.stop()
            kill_local_daemons()
        log.error("Simulator did not start. See %s", LOG_DIR / "daemon.log")
        return False

    def _launch(self) -> None:
        LOG_DIR.mkdir(exist_ok=True)
        cmd = _daemon_command(self.headless)
        log.info("Launching simulator: %s  (logs: logs/daemon.log)", " ".join(cmd))
        self._log_file = open(LOG_DIR / "daemon.log", "a", encoding="utf-8", errors="replace")
        self._log_file.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)} =====\n")
        self._log_file.flush()
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0  # Ctrl+C here doesn't hit it
        self._proc = subprocess.Popen(
            cmd, stdout=self._log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            creationflags=flags, start_new_session=(os.name != "nt"),
        )

    def _wait_until_ready(self) -> str:
        deadline = time.monotonic() + self.start_timeout_s
        last_state = None
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                return f"daemon process exited with code {self._proc.returncode}"
            st = daemon_status()
            state = st.get("state") if st else None
            if state != last_state and state:
                log.info("Simulator daemon state: %s", state)
                last_state = state
            if state == "running":
                return "running"
            if state == "error":
                return f"daemon error: {st.get('error') or 'backend not ready in time'}"
            time.sleep(0.5)
        return "timed out"

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            log.info("Stopping simulator daemon...")
            try:
                import psutil

                parent = psutil.Process(self._proc.pid)
                for child in parent.children(recursive=True):
                    child.kill()
            except Exception:
                pass
            self._proc.kill()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        self._proc = None
        if self._log_file:
            self._log_file.close()
            self._log_file = None
