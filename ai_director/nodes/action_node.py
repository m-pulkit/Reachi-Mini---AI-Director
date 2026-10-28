"""Action Node: the robot's body and voice.

Consumes `action_queue`:
  * PoseCommand    -> silent gesture ('attention' at "rolling", looping 'thinking' while PROCESSING)
  * SayCommand     -> speak a line + gesture
  * CritiqueCommand-> speak `spoken_critique` WHILE performing `reachy_emotion` (in parallel),
                      then return the FSM to IDLE.

Speech runs in a subprocess (see tts_worker.py) driven from a helper thread, and motion runs on
this node's thread, so voice and body are truly simultaneous and neither can freeze the other.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
from pathlib import Path

from ..config import ActionSettings
from ..hal import RobotHAL
from ..messages import CritiqueCommand, CritiqueResult, PoseCommand, SayCommand, Shutdown
from ..state import StateMachine, SystemState
from .base import BaseNode

_TTS_SCRIPT = Path(__file__).resolve().parent.parent / "tts_worker.py"


class Speaker:
    """Non-reentrant TTS wrapper around the tts_worker subprocess."""

    def __init__(self, cfg: ActionSettings) -> None:
        self.cfg = cfg
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def say(self, text: str) -> None:
        """Blocking until speech is finished (or killed / timed out)."""
        if not text.strip():
            return
        cmd = [sys.executable, str(_TTS_SCRIPT), "--rate", str(self.cfg.tts_rate),
               "--volume", str(self.cfg.tts_volume), "--voice", self.cfg.tts_voice_hint, text]
        try:
            with self._lock:
                self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            _, err = self._proc.communicate(timeout=self.cfg.tts_timeout_s)
            if self._proc.returncode not in (0, None) and err:
                print(f"[TTS] {err.decode(errors='ignore').strip()}", file=sys.stderr)
        except subprocess.TimeoutExpired:
            self.stop()
        except OSError as e:
            print(f"[TTS unavailable] {e}. Robot would say: {text}", file=sys.stderr)
        finally:
            with self._lock:
                self._proc = None

    def stop(self) -> None:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._proc.kill()


class ActionNode(BaseNode):
    def __init__(
        self,
        stop_event: threading.Event,
        fsm: StateMachine,
        hal: RobotHAL,
        cfg: ActionSettings,
        action_queue: queue.Queue,
    ) -> None:
        super().__init__("action", stop_event)
        self.fsm, self.hal, self.cfg = fsm, hal, cfg
        self.action_q = action_queue
        self.speaker = Speaker(cfg)

    def wake(self) -> None:
        self.speaker.stop()
        self.hal.cancel_motion()
        self.action_q.put(Shutdown())

    def on_crash(self) -> None:
        self.fsm.force_idle()

    def run_loop(self) -> None:
        while not self.stop_event.is_set():
            msg = self.action_q.get()
            if isinstance(msg, Shutdown):
                return
            if isinstance(msg, PoseCommand):
                self._pose(msg.name)
            elif isinstance(msg, SayCommand):
                self._perform(msg.text, msg.emotion)
                if msg.return_to_idle:
                    self.fsm.transition(SystemState.IDLE)
            elif isinstance(msg, CritiqueCommand):
                self._critique(msg)

    # ------------------------------------------------------------------ behaviours
    def _pose(self, name: str) -> None:
        if name == "thinking":
            # Sway until the verdict arrives (new message queued) or the state moves on.
            while (not self.stop_event.is_set() and self.fsm.state is SystemState.PROCESSING
                   and self.action_q.empty()):
                self.hal.play_pose("thinking")
            if self.fsm.state is not SystemState.PROCESSING:
                return
            self.hal.go_neutral(0.5)
        else:
            self.hal.play_pose(name)

    def _perform(self, text: str, emotion: str) -> None:
        """Speak and move at the same time; return when both are done."""
        voice = threading.Thread(target=self.speaker.say, args=(text,), name="tts", daemon=True)
        voice.start()
        try:
            self.hal.play_emotion(emotion)
        finally:
            voice.join(timeout=self.cfg.tts_timeout_s + 1)

    def _critique(self, cmd: CritiqueCommand) -> None:
        r = cmd.result
        emotion = r.reachy_emotion
        # A FAIL always gets the "swipe away" unless the model asked for something stronger.
        if r.overall_verdict == "FAIL" and emotion in ("neutral", "joy"):
            emotion = "disappointment"
        self._print_scorecard(cmd.take_id, r, cmd.brain_offline)
        try:
            self._perform(r.spoken_critique, emotion)
        finally:
            self.fsm.transition(SystemState.IDLE, expected=[SystemState.FEEDBACK])
            if not self.stop_event.is_set():
                self.log.info("Ready for the next take. Say 'Reachy, rolling!'")

    @staticmethod
    def _print_scorecard(take_id: int, r: CritiqueResult, offline: bool) -> None:
        bar = lambda s: "#" * s + "." * (10 - s)  # noqa: E731
        print(
            "\n" + "=" * 62 +
            f"\n  TAKE {take_id:03d}  VERDICT: {r.overall_verdict}{'  (BRAIN OFFLINE)' if offline else ''}"
            f"\n  Hook    [{bar(r.hook_score)}] {r.hook_score}/10"
            f"\n  Pacing  [{bar(r.pacing_score)}] {r.pacing_score}/10"
            f"\n  Visuals [{bar(r.visuals_score)}] {r.visuals_score}/10"
            f"\n  Emotion: {r.reachy_emotion}"
            f"\n  \"{r.spoken_critique}\"\n" + "=" * 62 + "\n",
            flush=True,
        )
