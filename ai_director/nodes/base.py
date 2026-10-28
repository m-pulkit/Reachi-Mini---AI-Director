"""Base class for every node: a named daemon thread with a shared stop event and crash isolation."""

from __future__ import annotations

import logging
import threading
import time


class BaseNode(threading.Thread):
    #: seconds to wait before restarting `run_loop` after an unexpected exception
    RESTART_BACKOFF_S = 1.0

    def __init__(self, name: str, stop_event: threading.Event) -> None:
        super().__init__(name=name, daemon=True)
        self.stop_event = stop_event
        self.log = logging.getLogger(f"node.{name}")

    # Subclasses implement these -------------------------------------------------
    def setup(self) -> None:
        """Runs once in the node's own thread before the loop."""

    def run_loop(self) -> None:
        """Main loop. Must return promptly once `self.stop_event` is set."""
        raise NotImplementedError

    def teardown(self) -> None:
        """Runs once in the node's own thread after the loop exits."""

    def wake(self) -> None:
        """Called by the supervisor during shutdown to unblock queue waits (e.g. poison pill)."""

    # ---------------------------------------------------------------------------
    def run(self) -> None:
        self.log.info("started")
        try:
            self.setup()
            while not self.stop_event.is_set():
                try:
                    self.run_loop()
                    break  # run_loop returned normally -> stop requested
                except Exception:
                    self.log.exception("crashed; restarting in %.1fs", self.RESTART_BACKOFF_S)
                    self.on_crash()
                    time.sleep(self.RESTART_BACKOFF_S)
        except Exception:
            self.log.exception("setup failed; node is down")
        finally:
            try:
                self.teardown()
            except Exception:
                self.log.exception("teardown failed")
            self.log.info("stopped")

    def on_crash(self) -> None:
        """Hook for recovery after an exception in run_loop (e.g. reset the FSM)."""
