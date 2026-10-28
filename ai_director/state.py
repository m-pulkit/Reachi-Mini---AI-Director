"""Thread-safe finite state machine shared by every node.

    IDLE ──"rolling"──▶ RECORDING ──"cut"──▶ PROCESSING ──critique──▶ FEEDBACK ──done──▶ IDLE

Nodes never set the state directly: they request a transition, and the machine rejects
illegal ones (e.g. "cut" while IDLE). Nodes can block on `wait_for` instead of busy-polling.
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Callable, Iterable

log = logging.getLogger(__name__)


class SystemState(str, Enum):
    IDLE = "IDLE"
    RECORDING = "RECORDING"
    PROCESSING = "PROCESSING"
    FEEDBACK = "FEEDBACK"


_ALLOWED: dict[SystemState, set[SystemState]] = {
    SystemState.IDLE: {SystemState.RECORDING},
    SystemState.RECORDING: {SystemState.PROCESSING, SystemState.IDLE},  # IDLE = aborted take
    SystemState.PROCESSING: {SystemState.FEEDBACK, SystemState.IDLE},
    SystemState.FEEDBACK: {SystemState.IDLE},
}

Listener = Callable[[SystemState, SystemState], None]


class StateMachine:
    def __init__(self) -> None:
        self._state = SystemState.IDLE
        self._cond = threading.Condition()
        self._listeners: list[Listener] = []
        self._entered_at = time.monotonic()
        self.take_id = 0  # incremented every time a new take starts
        self.take_started_at = 0.0  # time.monotonic() at the last IDLE -> RECORDING
        self.idle_since = 0.0  # time.monotonic() at the last return to IDLE

    @property
    def state(self) -> SystemState:
        with self._cond:
            return self._state

    @property
    def seconds_in_state(self) -> float:
        with self._cond:
            return time.monotonic() - self._entered_at

    def add_listener(self, fn: Listener) -> None:
        """Register a callback(old, new). Callbacks run on the transitioning thread: keep them fast."""
        with self._cond:
            self._listeners.append(fn)

    def transition(self, new: SystemState, *, expected: Iterable[SystemState] | None = None) -> bool:
        """Atomically move to `new`. Returns False (no-op) if the move is illegal or
        the current state is not in `expected`."""
        with self._cond:
            old = self._state
            if expected is not None and old not in set(expected):
                return False
            if new not in _ALLOWED[old]:
                log.debug("Rejected transition %s -> %s", old.value, new.value)
                return False
            self._state = new
            self._entered_at = time.monotonic()
            if new is SystemState.RECORDING:
                self.take_id += 1
                self.take_started_at = self._entered_at
            elif new is SystemState.IDLE:
                self.idle_since = self._entered_at
            listeners = list(self._listeners)
            self._cond.notify_all()
        log.info("STATE  %s -> %s", old.value, new.value)
        for fn in listeners:
            try:
                fn(old, new)
            except Exception:  # a buggy listener must never break the FSM
                log.exception("State listener failed")
        return True

    def force_idle(self) -> None:
        """Recovery path used after unexpected errors."""
        with self._cond:
            old = self._state
            self._state = SystemState.IDLE
            self._entered_at = self.idle_since = time.monotonic()
            listeners = list(self._listeners)
            self._cond.notify_all()
        if old is not SystemState.IDLE:
            log.warning("STATE  %s -> IDLE (forced)", old.value)
            for fn in listeners:
                try:
                    fn(old, SystemState.IDLE)
                except Exception:
                    log.exception("State listener failed")

    def wait_for(self, states: Iterable[SystemState], timeout: float | None = None) -> bool:
        targets = set(states)
        with self._cond:
            return self._cond.wait_for(lambda: self._state in targets, timeout=timeout)
