"""Supervisor: wires queues, HAL and nodes together, runs the preview UI on the main thread,
and guarantees a clean shutdown on Ctrl+C.

                ┌────────────── StateMachine (shared, thread-safe) ──────────────┐
                │  IDLE → RECORDING → PROCESSING → FEEDBACK → IDLE                │
                └─────────────────────────────────────────────────────────────────┘
   mic ─▶ AudioNode ──TakeEnded──▶ brain_queue ─▶ BrainNode ──CritiqueCommand──▶ action_queue ─▶ ActionNode
              │                                     ▲   │                                         │
              └──PoseCommand('attention')───────────┼───┼──PoseCommand('thinking')───────────────▶│
                                                    │   │                                         ▼
   camera ─▶ VisionNode (rolling take buffer) ──export_take()                           RobotHAL (TTS ∥ motion)
"""

from __future__ import annotations

import logging
import os
import queue
import signal
import sys
import threading
import time

import numpy as np

from .config import Settings
from .hal import RobotHAL
from .nodes.action_node import ActionNode
from .nodes.audio_node import AudioNode
from .nodes.base import BaseNode
from .nodes.brain_node import BrainNode
from .nodes.vision_node import VisionNode
from .state import StateMachine, SystemState

log = logging.getLogger("director")

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


class DirectorApp:
    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or Settings()
        self.stop_event = threading.Event()
        self.fsm = StateMachine()
        self.hal = RobotHAL(self.s.robot)
        self.brain_q: queue.Queue = queue.Queue()
        self.action_q: queue.Queue = queue.Queue()

        self.vision = VisionNode(self.stop_event, self.fsm, self.hal, self.s.vision)
        self.audio = AudioNode(self.stop_event, self.fsm, self.s.audio, self.brain_q, self.action_q)
        self.brain = BrainNode(self.stop_event, self.fsm, self.s.brain, self.vision,
                               self.brain_q, self.action_q, self.s.save_takes_dir)
        self.action = ActionNode(self.stop_event, self.fsm, self.hal, self.s.action, self.action_q)
        self.nodes: list[BaseNode] = [self.vision, self.audio, self.brain, self.action]
        self._preview_ok = self.s.show_preview and cv2 is not None

    # ------------------------------------------------------------------ lifecycle
    def run(self) -> None:
        mode = self.hal.connect()
        print(f"\n  ALGORITHMIC AI DIRECTOR  |  robot: {mode.upper()}  |  brain: {self.s.brain.provider}\n"
              f"  Say 'Reachy, rolling!' to start, 'Cut!' to stop.\n"
              f"  Keyboard: R = rolling, C = cut, Q = quit  (press them in THIS terminal"
              f"{' or in the preview window' if self._preview_ok else ''})\n")
        self.fsm.add_listener(self._on_state)
        for n in self.nodes:
            n.start()
        threading.Thread(target=self._console_keys, name="console-keys", daemon=True).start()

        try:
            # Ctrl+C anywhere -> stop_event. Main thread stays responsive (short waits only).
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGINT, lambda *_: self.stop_event.set())
                if hasattr(signal, "SIGTERM"):
                    signal.signal(signal.SIGTERM, lambda *_: self.stop_event.set())
            while not self.stop_event.is_set():
                if self._preview_ok:
                    self._preview_tick()
                else:
                    self.stop_event.wait(0.2)
                self._watchdog()
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        log.info("Shutting down...")
        self.stop_event.set()
        for n in self.nodes:
            n.wake()
        for n in self.nodes:
            n.join(timeout=5.0)
            if n.is_alive():
                log.warning("Node %s did not stop in time.", n.name)
        if self._preview_ok:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        self.hal.close()
        log.info("Goodbye. Stay viral.")

    # ------------------------------------------------------------------ helpers
    def _watchdog(self) -> None:
        """Recover from a wedged pipeline (e.g. a node died mid-take)."""
        st, age = self.fsm.state, self.fsm.seconds_in_state
        limit = {
            SystemState.RECORDING: self.s.vision.max_take_seconds * 3,
            SystemState.PROCESSING: (self.s.brain.request_timeout_s + 10) * (2 + self.s.brain.max_retries),
            SystemState.FEEDBACK: self.s.action.tts_timeout_s + 30,
        }.get(st)
        if limit and age > limit:
            log.error("Watchdog: stuck in %s for %.0fs, resetting to IDLE.", st.value, age)
            self.fsm.force_idle()

    def _handle_key(self, key: str) -> None:
        key = key.lower()
        if key == "r":
            if not self.audio.trigger_rolling():
                print(f"  (can't start: state is {self.fsm.state.value})", flush=True)
        elif key == "c":
            if not self.audio.trigger_cut():
                print(f"  (can't cut: state is {self.fsm.state.value})", flush=True)
        elif key in ("q", "\x1b"):
            self.stop_event.set()

    def _console_keys(self) -> None:
        """R / C / Q typed in the terminal window (works even without the preview window)."""
        if os.name == "nt":
            import msvcrt

            while not self.stop_event.is_set():
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch not in ("\x00", "\xe0"):
                        self._handle_key(ch)
                else:
                    self.stop_event.wait(0.05)
        else:  # POSIX: type the letter then Enter
            while not self.stop_event.is_set():
                line = sys.stdin.readline()
                if not line:
                    return
                if line.strip():
                    self._handle_key(line.strip()[0])

    def _on_state(self, old: SystemState, new: SystemState) -> None:
        banners = {
            SystemState.RECORDING: ("[REC] RECORDING  -  perform now, say 'Cut!' (or press C) to stop", 1200),
            SystemState.PROCESSING: ("[...] PROCESSING  -  the Director is judging you", 800),
            SystemState.IDLE: ("[ ] IDLE  -  say 'Reachy, rolling!' (or press R) for the next take", None),
        }
        if new in banners:
            text, freq = banners[new]
            print(f"\n>>> {text}\n", flush=True)
            if freq and os.name == "nt":
                import winsound

                threading.Thread(target=winsound.Beep, args=(freq, 180), daemon=True).start()

    def _preview_tick(self) -> None:
        frame = self.vision.latest_frame()
        if frame is None:  # keep the window (and its keyboard focus) alive without a camera
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "NO CAMERA FRAMES", (150, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255), 2, cv2.LINE_AA)
        if frame is not None:
            st = self.fsm.state
            colors = {SystemState.IDLE: (200, 200, 200), SystemState.RECORDING: (0, 0, 255),
                      SystemState.PROCESSING: (0, 200, 255), SystemState.FEEDBACK: (0, 220, 0)}
            label = st.value
            if st is SystemState.RECORDING:
                label = f"REC  {self.fsm.seconds_in_state:5.1f}s"
                if int(time.monotonic() * 2) % 2:
                    cv2.circle(frame, (24, 28), 9, (0, 0, 255), -1)
            cv2.putText(frame, label, (42, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[st], 2, cv2.LINE_AA)
            cv2.putText(frame, f"robot:{self.hal.mode}  mic:{'on' if self.audio.mic_ok else 'off'}  "
                        "[R]oll [C]ut [Q]uit", (12, frame.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            try:
                cv2.imshow("AI Director", frame)
            except cv2.error as e:
                log.warning("Preview disabled (OpenCV has no GUI support: %s)", e)
                self._preview_ok = False
                return
        try:
            key = cv2.waitKey(30) & 0xFF
        except cv2.error:
            self._preview_ok = False
            return
        if key != 255:
            self._handle_key(chr(key))
