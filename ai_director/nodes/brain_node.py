"""Brain Node: cognition.

Blocks on `brain_queue` for a `TakeEnded` event (sent by the Audio node on "Cut!"), pulls the
chronological frame sequence from the Vision node, calls the VLM in a worker thread with a hard
timeout, validates the JSON, and hands a `CritiqueCommand` to the Action node.

Never raises into the robot: any failure becomes a "my brain is offline" critique.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import queue
import threading
import time
from pathlib import Path

from ..config import BrainSettings
from ..messages import CritiqueCommand, CritiqueResult, PoseCommand, Shutdown, TakeEnded, TimedFrame
from ..state import StateMachine, SystemState
from ..vlm import VLMClient, VLMError, make_client
from .base import BaseNode
from .vision_node import VisionNode

OFFLINE_LINES = [
    "My brain is offline. The cloud ghosted me harder than your audience did. Do it again.",
    "Brain is offline, bestie. Even the servers swiped away. Try another take.",
    "Algorithm unreachable. I'll pretend that was a ten, but we both know it wasn't.",
]
NOTHING_SEEN_LINE = "I didn't capture a single frame. Was the lens cap on, or was that the point?"


class BrainNode(BaseNode):
    def __init__(
        self,
        stop_event: threading.Event,
        fsm: StateMachine,
        cfg: BrainSettings,
        vision: VisionNode,
        brain_queue: queue.Queue,
        action_queue: queue.Queue,
        save_takes_dir: str = "",
    ) -> None:
        super().__init__("brain", stop_event)
        self.fsm, self.cfg, self.vision = fsm, cfg, vision
        self.brain_q, self.action_q = brain_queue, action_queue
        self.save_dir = Path(save_takes_dir) if save_takes_dir else None
        self._client: VLMClient | None = None
        self._client_error = ""
        self._executor = cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="vlm-call")
        self._offline_idx = 0

    def setup(self) -> None:
        try:
            self._client = make_client(self.cfg)
            self.log.info("VLM provider: %s", self._client.name)
        except VLMError as e:
            self._client_error = str(e)
            self.log.error("VLM unavailable: %s. Robot will report 'brain offline' after each take.", e)

    def wake(self) -> None:
        self.brain_q.put(Shutdown())

    def teardown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def on_crash(self) -> None:
        self.fsm.force_idle()

    def run_loop(self) -> None:
        while not self.stop_event.is_set():
            msg = self.brain_q.get()
            if isinstance(msg, Shutdown):
                return
            if isinstance(msg, TakeEnded):
                self._process(msg)

    # ------------------------------------------------------------------ core
    def _process(self, take: TakeEnded) -> None:
        if self.fsm.state is not SystemState.PROCESSING:
            self.log.warning("Got take %d but state is %s; ignoring.", take.take_id, self.fsm.state.value)
            return
        self.action_q.put(PoseCommand("thinking"))
        time.sleep(0.2)  # let the vision loop notice the transition and freeze the buffer
        frames = self.vision.export_take(take.take_id)
        self.log.info("Analysing take %d: %.1fs, %d frames, %d transcript segments",
                      take.take_id, take.duration_s, len(frames), len(take.transcript))
        self._maybe_save(take, frames)

        if not frames:
            result, offline = self._canned(NOTHING_SEEN_LINE, "anger"), False
        else:
            result, offline = self._call_vlm(take, frames)

        self.log.info("VERDICT %s | hook %d pacing %d visuals %d | %s | %r",
                      result.overall_verdict, result.hook_score, result.pacing_score,
                      result.visuals_score, result.reachy_emotion, result.spoken_critique)
        if self.fsm.transition(SystemState.FEEDBACK, expected=[SystemState.PROCESSING]):
            self.action_q.put(CritiqueCommand(take_id=take.take_id, result=result, brain_offline=offline))

    def _call_vlm(self, take: TakeEnded, frames: list[TimedFrame]) -> tuple[CritiqueResult, bool]:
        if self._client is None:
            self.log.error("No VLM client (%s).", self._client_error)
            return self._offline(), True
        attempts = 1 + max(0, self.cfg.max_retries)
        for attempt in range(1, attempts + 1):
            if self.stop_event.is_set():
                break
            fut = self._executor.submit(self._client.critique, take, frames)
            try:
                return fut.result(timeout=self.cfg.request_timeout_s + 5), False
            except cf.TimeoutError:
                fut.cancel()
                self.log.error("VLM timed out (attempt %d/%d).", attempt, attempts)
            except VLMError as e:
                self.log.error("VLM error (attempt %d/%d): %s", attempt, attempts, e)
            except Exception:
                self.log.exception("Unexpected VLM failure (attempt %d/%d)", attempt, attempts)
            if attempt < attempts:
                self.stop_event.wait(1.5 * attempt)
        return self._offline(), True

    def _offline(self) -> CritiqueResult:
        line = OFFLINE_LINES[self._offline_idx % len(OFFLINE_LINES)]
        self._offline_idx += 1
        return self._canned(line, "disappointment")

    @staticmethod
    def _canned(line: str, emotion: str) -> CritiqueResult:
        return CritiqueResult(hook_score=0, pacing_score=0, visuals_score=0, overall_verdict="FAIL",
                              spoken_critique=line, reachy_emotion=emotion)

    def _maybe_save(self, take: TakeEnded, frames: list[TimedFrame]) -> None:
        if not self.save_dir:
            return
        try:
            d = self.save_dir / f"take_{time.strftime('%Y%m%d_%H%M%S')}_{take.take_id:03d}"
            d.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(frames):
                (d / f"{i:03d}_t{f.t:05.1f}.jpg").write_bytes(f.jpeg)
            (d / "transcript.json").write_text(json.dumps(
                {"duration_s": take.duration_s, "segments": [{"t": s.t, "text": s.text} for s in take.transcript]},
                indent=2), encoding="utf-8")
        except OSError as e:
            self.log.warning("Could not save take: %s", e)
