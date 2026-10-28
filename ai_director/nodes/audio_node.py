"""Audio Node: always listening.

Two internal threads so the microphone is never deaf while a phrase is being transcribed:

    [mic capture thread = this node] --AudioData--> _phrase_q --> [recognizer worker]
                                                                      |
                     IDLE + "rolling"  -> FSM: RECORDING, Action: 'attention' pose
                     RECORDING + text  -> append timestamped transcript segment
                     RECORDING + "cut" -> FSM: PROCESSING, Brain: TakeEnded(transcript)

`trigger_rolling()` / `trigger_cut()` are also exposed so the preview window's keyboard
shortcuts (R / C) go through exactly the same logic.
"""

from __future__ import annotations

import queue
import re
import threading
import time

from ..config import AudioSettings
from ..messages import PoseCommand, Shutdown, TakeEnded, TranscriptSegment
from ..state import StateMachine, SystemState
from .base import BaseNode

try:
    import speech_recognition as sr
except ImportError:  # pragma: no cover
    sr = None  # type: ignore[assignment]

# Speech engines routinely mis-hear "Reachy".
_NAME = r"(?:reachy|reachie|ritchie|richie|reechy|reach me|ricky|rachey|peachy)"
_ROLLING = re.compile(r"\b(?:rolling|roll camera)\b", re.I)
_ROLLING_WITH_NAME = re.compile(rf"\b{_NAME}\W*(?:\w+\W+)?(?:rolling|roll)\b", re.I)
# "Cut" must END the phrase (optionally "cut it", "and cut", "cut cut") or be a very short phrase,
# so "I cut the onions" inside a cooking video doesn't end the take.
_CUT_AT_END = re.compile(r"\b(?:and\s+)?cut(?:\s+(?:it|that|cut))?\W*$", re.I)
_CUT_ANY = re.compile(r"\bcut\b", re.I)


class AudioNode(BaseNode):
    def __init__(
        self,
        stop_event: threading.Event,
        fsm: StateMachine,
        cfg: AudioSettings,
        brain_queue: queue.Queue,
        action_queue: queue.Queue,
    ) -> None:
        super().__init__("audio", stop_event)
        self.fsm, self.cfg = fsm, cfg
        self.brain_q, self.action_q = brain_queue, action_queue
        self._phrase_q: queue.Queue = queue.Queue(maxsize=50)
        self._transcript: list[TranscriptSegment] = []
        self._cmd_lock = threading.Lock()  # serialises voice + keyboard triggers
        self._recognizer = sr.Recognizer() if sr else None
        self._worker: threading.Thread | None = None
        self.mic_ok = False

    # ------------------------------------------------------------------ external triggers
    def trigger_rolling(self, leftover_text: str = "") -> bool:
        with self._cmd_lock:
            if not self.fsm.transition(SystemState.RECORDING, expected=[SystemState.IDLE]):
                return False
            self._transcript = []
            if leftover_text.strip():
                self._transcript.append(TranscriptSegment(0.0, leftover_text.strip()))
        self.action_q.put(PoseCommand("attention"))
        return True

    def trigger_cut(self, leftover_text: str = "", at: float | None = None) -> bool:
        with self._cmd_lock:
            if self.fsm.state is not SystemState.RECORDING:
                return False
            take_id = self.fsm.take_id
            duration = (at or time.monotonic()) - self.fsm.take_started_at
            if leftover_text.strip():
                self._transcript.append(TranscriptSegment(round(max(duration - 1.0, 0), 1), leftover_text.strip()))
            transcript, self._transcript = self._transcript, []
            if not self.fsm.transition(SystemState.PROCESSING, expected=[SystemState.RECORDING]):
                return False
        self.log.info("CUT. Take %d: %.1fs, %d transcript segments", take_id, duration, len(transcript))
        self.brain_q.put(TakeEnded(take_id=take_id, duration_s=round(duration, 1), transcript=transcript))
        return True

    # ------------------------------------------------------------------ lifecycle
    def setup(self) -> None:
        self._worker = threading.Thread(target=self._recognize_loop, name="audio-recognizer", daemon=True)
        self._worker.start()

    def wake(self) -> None:
        try:
            self._phrase_q.put_nowait(Shutdown())
        except queue.Full:
            pass

    def teardown(self) -> None:
        self.wake()
        if self._worker:
            self._worker.join(timeout=3)

    def run_loop(self) -> None:
        if sr is None:
            self.log.error("speech_recognition not installed: voice control disabled (use R / C keys).")
            self.stop_event.wait()
            return
        try:
            mic = sr.Microphone(device_index=self._pick_device())
            with mic as source:
                self.log.info("Calibrating for ambient noise (stay quiet ~1s)...")
                self._recognizer.adjust_for_ambient_noise(source, duration=1.0)
                self._recognizer.dynamic_energy_threshold = True
                self._recognizer.pause_threshold = self.cfg.pause_threshold_s
                self.mic_ok = True
                self.log.info("Listening. Say 'Reachy, rolling!' to start a take.")
                while not self.stop_event.is_set():
                    try:
                        audio = self._recognizer.listen(
                            source, timeout=1.0, phrase_time_limit=self.cfg.phrase_time_limit_s
                        )
                    except sr.WaitTimeoutError:
                        continue
                    end = time.monotonic()
                    duration = len(audio.frame_data) / float(audio.sample_rate * audio.sample_width)
                    try:
                        self._phrase_q.put_nowait((end - duration, end, audio))
                    except queue.Full:
                        self.log.warning("Recognizer backlog full; dropping a phrase.")
        except (OSError, AttributeError) as e:
            # No microphone / PyAudio missing. Don't spin: degrade to keyboard control.
            self.mic_ok = False
            self.log.error("Microphone unavailable (%s). Voice control disabled (use R / C keys).", e)
            self.stop_event.wait()

    def _pick_device(self) -> int | None:
        if self.cfg.mic_device_index is not None:
            return self.cfg.mic_device_index
        if self.cfg.mic_name_hint:
            try:
                for i, name in enumerate(sr.Microphone.list_microphone_names()):
                    if self.cfg.mic_name_hint.lower() in name.lower():
                        self.log.info("Using microphone #%d: %s", i, name)
                        return i
            except Exception as e:
                self.log.warning("Could not list microphones: %s", e)
            self.log.warning("No microphone matches '%s'; using system default.", self.cfg.mic_name_hint)
        return None

    # ------------------------------------------------------------------ recognition worker
    def _recognize_loop(self) -> None:
        while not self.stop_event.is_set():
            item = self._phrase_q.get()
            if isinstance(item, Shutdown):
                return
            start, end, audio = item
            state = self.fsm.state
            if state in (SystemState.PROCESSING, SystemState.FEEDBACK):
                continue  # don't transcribe the robot's own voice
            if state is SystemState.IDLE and start < self.fsm.idle_since:
                continue  # phrase was captured while the robot was still talking
            try:
                text = self._recognizer.recognize_google(audio, language=self.cfg.language)
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                self.log.warning("Speech service unreachable: %s", e)
                continue
            except Exception:
                self.log.exception("Recognition failed")
                continue
            text = text.strip()
            if text:
                self.log.info("HEARD [%s] %r", state.value, text)
                self._handle_text(text, start, end)

    def _handle_text(self, text: str, start: float, end: float) -> None:
        state = self.fsm.state
        if state is SystemState.IDLE:
            pattern = _ROLLING_WITH_NAME if self.cfg.require_robot_name else _ROLLING
            m = pattern.search(text)
            if m:
                self.trigger_rolling(leftover_text=text[m.end():])
            return

        if state is SystemState.RECORDING:
            if _CUT_AT_END.search(text) or (len(text.split()) <= 3 and _CUT_ANY.search(text)):
                last_cut = list(_CUT_ANY.finditer(text))[-1]
                before = re.sub(r"\band\W*$", "", text[: last_cut.start()].rstrip(), flags=re.I)
                self.trigger_cut(leftover_text=before, at=end)
                return
            with self._cmd_lock:
                if self.fsm.state is SystemState.RECORDING:
                    t = max(0.0, start - self.fsm.take_started_at)
                    self._transcript.append(TranscriptSegment(round(t, 1), text))
