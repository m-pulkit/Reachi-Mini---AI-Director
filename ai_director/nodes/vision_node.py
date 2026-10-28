"""Vision Node: owns the camera.

* Always grabs frames at CAPTURE_HZ so the live preview never freezes.
* While RECORDING, keeps 1-2 JPEG frames/second in a thread-safe take buffer covering the
  *whole* take (if the take gets very long, the buffer is decimated instead of truncated,
  so the beginning — the hook — is never lost).
* While RECORDING, keeps the creator's face centred: native daemon tracking if the robot
  supports it, otherwise OpenCV Haar-cascade tracking driving `RobotHAL.look_at_pixel`.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from ..config import VisionSettings
from ..hal import RobotHAL
from ..messages import TimedFrame
from ..state import StateMachine, SystemState
from .base import BaseNode


class VisionNode(BaseNode):
    def __init__(self, stop_event: threading.Event, fsm: StateMachine, hal: RobotHAL, cfg: VisionSettings) -> None:
        super().__init__("vision", stop_event)
        self.fsm, self.hal, self.cfg = fsm, hal, cfg

        self._buf_lock = threading.Lock()
        self._take_frames: list[TimedFrame] = []
        self._take_id = -1
        self._sample_period = 1.0 / max(cfg.sample_fps, 0.1)
        self._max_buffer = max(4, int(cfg.max_take_seconds * cfg.sample_fps))

        self._latest_lock = threading.Lock()
        self._latest: np.ndarray | None = None

        self._face_cascade: cv2.CascadeClassifier | None = None
        self._native_tracking = False
        self._smoothed: tuple[float, float] | None = None
        self._last_seen_state = SystemState.IDLE

    # ------------------------------------------------------------------ public API (thread-safe)
    def latest_frame(self) -> np.ndarray | None:
        with self._latest_lock:
            return None if self._latest is None else self._latest.copy()

    def export_take(self, take_id: int) -> list[TimedFrame]:
        """Chronological frames of `take_id`, uniformly subsampled to MAX_FRAMES_TO_VLM.
        Always keeps the first and last frame (hook and CTA)."""
        with self._buf_lock:
            if take_id != self._take_id:
                return []
            frames = list(self._take_frames)
        n, k = len(frames), self.cfg.max_frames_to_vlm
        if n <= k:
            return frames
        idx = np.unique(np.linspace(0, n - 1, k).round().astype(int))
        return [frames[i] for i in idx]

    # ------------------------------------------------------------------ lifecycle
    def setup(self) -> None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(path)
        self._face_cascade = None if cascade.empty() else cascade

    def teardown(self) -> None:
        self.hal.stop_face_tracking()

    def run_loop(self) -> None:
        period = 1.0 / max(self.cfg.capture_hz, 1.0)
        track_period = 1.0 / max(self.cfg.face_tracking_hz, 0.5)
        next_sample = next_track = 0.0

        while not self.stop_event.is_set():
            t0 = time.monotonic()
            state = self.fsm.state
            self._handle_state_change(state)

            frame = self.hal.get_frame()
            if frame is not None:
                with self._latest_lock:
                    self._latest = frame

                if state is SystemState.RECORDING:
                    if t0 >= next_sample:
                        self._append_sample(frame, t0 - self.fsm.take_started_at)
                        next_sample = t0 + self._sample_period
                    if not self._native_tracking and t0 >= next_track:
                        self._software_track(frame)
                        next_track = t0 + track_period
                else:
                    next_sample = 0.0

            self.stop_event.wait(max(0.0, period - (time.monotonic() - t0)))

    # ------------------------------------------------------------------ internals
    def _handle_state_change(self, state: SystemState) -> None:
        if state is self._last_seen_state:
            return
        prev, self._last_seen_state = self._last_seen_state, state
        if state is SystemState.RECORDING:
            with self._buf_lock:
                self._take_frames = []
                self._take_id = self.fsm.take_id
                self._sample_period = 1.0 / max(self.cfg.sample_fps, 0.1)
            self._smoothed = None
            self._native_tracking = self.hal.start_face_tracking()
        elif prev is SystemState.RECORDING:
            self.hal.stop_face_tracking()
            self._native_tracking = False
            with self._buf_lock:
                self.log.info("Take %d frozen: %d frames", self._take_id, len(self._take_frames))

    def _append_sample(self, frame: np.ndarray, t: float) -> None:
        h, w = frame.shape[:2]
        if w > self.cfg.frame_width:
            frame = cv2.resize(frame, (self.cfg.frame_width, int(h * self.cfg.frame_width / w)),
                               interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
        if not ok:
            return
        with self._buf_lock:
            self._take_frames.append(TimedFrame(t=round(t, 2), jpeg=jpg.tobytes()))
            if len(self._take_frames) > self._max_buffer:
                # Over budget: keep every other frame and halve the sampling rate.
                self._take_frames = self._take_frames[::2]
                self._sample_period *= 2
                self.log.info("Long take: decimated buffer, now sampling every %.1fs", self._sample_period)

    def _software_track(self, frame: np.ndarray) -> None:
        if self._face_cascade is None:
            return
        h, w = frame.shape[:2]
        scale = 320.0 / w
        small = cv2.cvtColor(cv2.resize(frame, (320, int(h * scale))), cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(small, scaleFactor=1.15, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        cx, cy = (x + fw / 2) / scale, (y + fh / 2) / scale
        # EMA smoothing + dead-band so the head doesn't jitter.
        if self._smoothed is None:
            self._smoothed = (cx, cy)
        else:
            a = 0.35
            self._smoothed = (a * cx + (1 - a) * self._smoothed[0], a * cy + (1 - a) * self._smoothed[1])
        u, v = self._smoothed
        if abs(u - w / 2) > 0.06 * w or abs(v - h / 2) > 0.06 * h:
            # Proportional step toward the face (gain < 1) => smooth pursuit instead of snapping.
            gain = 0.4
            self.hal.look_at_pixel(int(w / 2 + gain * (u - w / 2)), int(h / 2 + gain * (v - h / 2)))
