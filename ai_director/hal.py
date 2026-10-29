"""Hardware Abstraction Layer for Reachy Mini.

The rest of the codebase only talks to `RobotHAL`. It never imports the SDK and never knows
whether it is driving the physical robot or the MuJoCo simulator.

Connection strategy (REACHY_MODE=auto):
  1. A healthy simulator daemon already on localhost?  -> use it (simulation)
  2. ReachyMini(use_sim=False)                         -> physical robot (local or reachy-mini.local)
  3. Start `reachy-mini-daemon --sim` (see sim_daemon.py: waits for it, retries if MuJoCo
     misses its 2 s start-up window), then ReachyMini(use_sim=True) on localhost -> simulation
  4. If nothing works, a no-op "headless" backend so the app (webcam + TTS + VLM) still runs.
"""

from __future__ import annotations

import inspect
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import RobotSettings
from .sim_daemon import SimDaemon, daemon_status

log = logging.getLogger(__name__)

try:
    from reachy_mini import ReachyMini
    from reachy_mini.utils import create_head_pose

    SDK_AVAILABLE = True
except Exception as e:  # pragma: no cover - SDK missing on this machine
    ReachyMini = None  # type: ignore[assignment]
    create_head_pose = None  # type: ignore[assignment]
    SDK_AVAILABLE = False
    _SDK_IMPORT_ERROR = e

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- choreography
@dataclass(frozen=True)
class Keyframe:
    """One target of an expressive move. Angles in degrees, translations in mm."""

    duration: float
    roll: float = 0.0
    pitch: float = 0.0  # + = look down on Reachy Mini (flip PITCH_SIGN below if your unit differs)
    yaw: float = 0.0  # + = turn to the robot's left
    z: float = 0.0
    antennas: tuple[float, float] = (0.0, 0.0)  # (right, left) degrees
    body_yaw: float = 0.0
    method: str = "minjerk"  # minjerk | linear | ease_in_out | cartoon


PITCH_SIGN = 1.0
NEUTRAL = Keyframe(duration=0.8)

# Hand-made choreographies. Each maps to a function of the robot's "personality".
CHOREOGRAPHIES: dict[str, list[Keyframe]] = {
    # Approving double nod, antennas perked up and wiggling.
    "joy": [
        Keyframe(0.35, pitch=-8, z=8, antennas=(-35, 35), method="ease_in_out"),
        Keyframe(0.25, pitch=12, antennas=(25, -25)),
        Keyframe(0.25, pitch=-8, z=8, antennas=(-35, 35)),
        Keyframe(0.25, pitch=12, antennas=(25, -25)),
        Keyframe(0.4, pitch=-5, z=5, antennas=(-20, 20), method="cartoon"),
    ],
    # "Swipe to the next video": head turns away in disgust, body follows, antennas droop.
    "disappointment": [
        Keyframe(0.5, pitch=10, roll=-6, antennas=(60, -60)),
        Keyframe(0.7, yaw=45, pitch=12, roll=-10, antennas=(80, -80), body_yaw=30, method="ease_in_out"),
        Keyframe(1.4, yaw=48, pitch=14, roll=-10, antennas=(85, -85), body_yaw=32),  # hold the snub
    ],
    # Fast head shake "no", antennas pinned back, then a sharp swipe away.
    "anger": [
        Keyframe(0.25, pitch=5, antennas=(70, -70), z=-5),
        Keyframe(0.18, yaw=-25, pitch=5, antennas=(70, -70), method="linear"),
        Keyframe(0.18, yaw=25, pitch=5, antennas=(70, -70), method="linear"),
        Keyframe(0.18, yaw=-25, pitch=5, antennas=(70, -70), method="linear"),
        Keyframe(0.18, yaw=25, pitch=5, antennas=(70, -70), method="linear"),
        Keyframe(0.5, yaw=-50, pitch=10, roll=8, antennas=(80, -80), body_yaw=-35, method="cartoon"),
        Keyframe(1.0, yaw=-50, pitch=10, roll=8, antennas=(80, -80), body_yaw=-35),
    ],
    # Head jerks back, antennas shoot up.
    "surprise": [
        Keyframe(0.25, pitch=-15, z=12, antennas=(-60, 60), method="cartoon"),
        Keyframe(0.9, pitch=-12, z=10, antennas=(-55, 55)),
    ],
    # Slow sideways tilt, looking down at "the phone", one antenna flops.
    "boredom": [
        Keyframe(1.2, roll=18, pitch=15, yaw=10, antennas=(70, -10), method="ease_in_out"),
        Keyframe(1.0, roll=20, pitch=18, yaw=12, antennas=(75, -5)),
    ],
    # Thoughtful small tilt.
    "neutral": [
        Keyframe(0.6, roll=8, pitch=3, antennas=(15, -15)),
        Keyframe(0.6, roll=0, antennas=(0, 0)),
    ],
    # Used when the take starts: perk up, "I'm watching".
    "attention": [Keyframe(0.5, pitch=-4, z=6, antennas=(-25, 25), method="cartoon")],
    # Slow "thinking" sway during PROCESSING.
    "thinking": [
        Keyframe(0.9, roll=10, pitch=-6, yaw=15, antennas=(-10, 40), method="ease_in_out"),
        Keyframe(0.9, roll=-10, pitch=-6, yaw=-15, antennas=(40, -10), method="ease_in_out"),
    ],
}

# Keywords for picking a clip from Pollen's recorded emotions library (if enabled).
RECORDED_KEYWORDS: dict[str, tuple[str, ...]] = {
    "joy": ("cheerful", "enthusiastic", "proud", "amazed", "happy", "joy"),
    "disappointment": ("disgusted", "contempt", "displeased", "sad", "resigned", "disappoint"),
    "anger": ("furious", "angry", "rage", "frustrated", "irritated"),
    "surprise": ("surprised", "amazed", "shocked"),
    "boredom": ("bored", "boredom", "tired", "indifferent"),
    "neutral": ("thoughtful", "attentive", "curious"),
}


# --------------------------------------------------------------------------- HAL
class RobotHAL:
    """Unified, thread-safe facade over Reachy Mini (physical or simulated)."""

    def __init__(self, settings: RobotSettings) -> None:
        self._s = settings
        self._mini: Any = None
        self._mode = "disconnected"
        self._motion_lock = threading.RLock()
        self._camera_lock = threading.Lock()
        self._webcam: Any = None
        self._webcam_disabled = False
        self._robot_camera_misses = 0
        self._native_tracking = False
        self._recorded_moves: Any = None
        self._recorded_index: dict[str, str] = {}
        self._cancel = threading.Event()
        self._sim: SimDaemon | None = None

    # ------------------------------------------------------------------ lifecycle
    @property
    def mode(self) -> str:
        """'physical', 'simulation' or 'headless'."""
        return self._mode

    def connect(self) -> str:
        if not SDK_AVAILABLE:
            log.error("reachy_mini SDK not importable (%s). Running HEADLESS.", _SDK_IMPORT_ERROR)
            self._mode = "headless"
            return self._mode

        mode = self._s.mode
        existing = daemon_status()
        existing_is_sim = bool(existing and existing.get("simulation_enabled"))

        if mode in ("auto", "physical") and not (mode == "auto" and existing_is_sim):
            try:
                log.info("Connecting to the physical Reachy Mini...")
                self._mini = self._construct(use_sim=False, host=self._s.host, timeout=self._s.connect_timeout_s)
                self._mode = "physical"
            except Exception as e:
                log.warning("Physical robot unavailable: %s", e)
                if mode == "physical":
                    raise
        if self._mini is None and mode in ("auto", "sim"):
            log.info("Falling back to Simulation Mode (MuJoCo)...")
            self._sim = SimDaemon(headless=self._s.sim_headless)
            if self._sim.ensure_running():
                try:
                    self._mini = self._construct(
                        use_sim=True, spawn_daemon=False, connection_mode="localhost_only",
                        localhost_only=True, timeout=max(self._s.connect_timeout_s, 10.0),
                    )
                    self._mode = "simulation"
                except Exception as e:
                    log.error("Simulator is running but the SDK could not connect: %s", e)
        if self._mini is None:
            log.error("No robot backend available. Running HEADLESS (webcam + voice only).")
            self._mode = "headless"
            return self._mode

        log.info("RobotHAL connected in %s mode.", self._mode.upper())
        self._load_recorded_moves()
        self.go_neutral(1.0)
        return self._mode

    @staticmethod
    def _construct(**kwargs: Any) -> Any:
        """Instantiate ReachyMini, dropping kwargs an older/newer SDK doesn't accept."""
        params = inspect.signature(ReachyMini.__init__).parameters
        accepted = {k: v for k, v in kwargs.items() if k in params}
        dropped = set(kwargs) - set(accepted)
        if dropped:
            log.debug("ReachyMini() does not accept %s; ignoring.", dropped)
        return ReachyMini(**accepted)

    def close(self) -> None:
        self._cancel.set()
        self.stop_face_tracking()
        if self._mini is not None:
            try:
                with self._motion_lock:
                    self._goto(NEUTRAL)
            except Exception:
                pass
            try:
                self._mini.__exit__(None, None, None)  # releases media + client connection
            except Exception as e:
                log.debug("ReachyMini close error: %s", e)
            self._mini = None
        if self._sim is not None and not self._s.sim_keep_alive:
            self._sim.stop()
        with self._camera_lock:
            if self._webcam is not None:
                self._webcam.release()
                self._webcam = None
        log.info("RobotHAL closed.")

    # ------------------------------------------------------------------ camera
    def get_frame(self) -> np.ndarray | None:
        """Latest BGR frame from the robot camera, or a webcam fallback. Never raises."""
        with self._camera_lock:
            if self._mini is not None and self._robot_camera_misses < 40:
                try:
                    frame = self._mini.media.get_frame()
                    if frame is not None and getattr(frame, "size", 0):
                        self._robot_camera_misses = 0
                        return frame
                except Exception as e:
                    log.debug("Robot camera error: %s", e)
                self._robot_camera_misses += 1
                if self._robot_camera_misses == 40:
                    log.warning("Robot camera returns no frames; switching to webcam fallback.")
                return None
            return self._read_webcam()

    def _read_webcam(self) -> np.ndarray | None:
        if cv2 is None or self._webcam_disabled or self._s.fallback_camera_index < 0:
            return None
        if self._webcam is None:
            self._webcam = cv2.VideoCapture(self._s.fallback_camera_index)
            if not self._webcam.isOpened():
                log.error("Fallback webcam %d could not be opened.", self._s.fallback_camera_index)
                self._webcam.release()
                self._webcam = None
                self._webcam_disabled = True
                return None
        ok, frame = self._webcam.read()
        return frame if ok else None

    # ------------------------------------------------------------------ face tracking
    def start_face_tracking(self) -> bool:
        """Try daemon-side (native) head tracking. Returns True if the robot handles it itself;
        False means the caller should do software tracking via `look_at_pixel`."""
        if self._mini is None:
            return False
        if hasattr(self._mini, "start_head_tracking"):
            try:
                self._mini.start_head_tracking(weight=1.0)
                self._native_tracking = True
                log.info("Native face tracking enabled.")
                return True
            except Exception as e:
                log.info("Native face tracking unavailable (%s); using OpenCV tracking.", e)
        return False

    def stop_face_tracking(self) -> None:
        if self._mini is not None and self._native_tracking:
            try:
                self._mini.stop_head_tracking()
            except Exception as e:
                log.debug("stop_head_tracking failed: %s", e)
        self._native_tracking = False

    def look_at_pixel(self, u: int, v: int) -> None:
        """Non-blocking: point the head at pixel (u, v) of the camera image."""
        if self._mini is None:
            return
        if not self._motion_lock.acquire(blocking=False):
            return  # an expressive move owns the head; skip this tracking tick
        try:
            pose = self._mini.look_at_image(int(u), int(v), duration=0.0, perform_movement=False)
            self._mini.set_target(head=pose)
        except Exception as e:
            log.debug("look_at_pixel failed: %s", e)
        finally:
            self._motion_lock.release()

    # ------------------------------------------------------------------ expressive motion
    def play_emotion(self, emotion: str) -> None:
        """Blocking: perform the choreography for `emotion`, then return to neutral."""
        if self._mini is None:
            log.info("[headless] would perform emotion: %s", emotion)
            time.sleep(1.0)
            return
        with self._motion_lock:
            self._cancel.clear()
            if self._play_recorded(emotion):
                self._goto(NEUTRAL)
                return
            for kf in CHOREOGRAPHIES.get(emotion, CHOREOGRAPHIES["neutral"]):
                if self._cancel.is_set():
                    break
                self._goto(kf)
            if not self._cancel.is_set():
                self._goto(NEUTRAL)

    def play_pose(self, name: str, return_to_neutral: bool = False) -> None:
        """Blocking: perform a named pose ('attention', 'thinking', ...) without a recorded clip."""
        if self._mini is None:
            return
        with self._motion_lock:
            for kf in CHOREOGRAPHIES.get(name, []):
                if self._cancel.is_set():
                    return
                self._goto(kf)
            if return_to_neutral:
                self._goto(NEUTRAL)

    def go_neutral(self, duration: float = 0.8) -> None:
        if self._mini is None:
            return
        with self._motion_lock:
            self._goto(Keyframe(duration=duration))

    def cancel_motion(self) -> None:
        self._cancel.set()
        if self._mini is not None and hasattr(self._mini, "cancel_move"):
            try:
                self._mini.cancel_move()
            except Exception:
                pass

    def _goto(self, kf: Keyframe) -> None:
        head = create_head_pose(
            z=kf.z, roll=kf.roll, pitch=PITCH_SIGN * kf.pitch, yaw=kf.yaw, mm=True, degrees=True
        )
        try:
            self._mini.goto_target(
                head=head,
                antennas=np.deg2rad(kf.antennas),
                duration=max(kf.duration, 0.05),
                method=kf.method,
                body_yaw=float(np.deg2rad(kf.body_yaw)),
            )
        except TypeError:  # older SDKs: no method/body_yaw kwargs
            self._mini.goto_target(head=head, antennas=np.deg2rad(kf.antennas), duration=max(kf.duration, 0.05))
        except Exception as e:
            log.warning("goto_target failed: %s", e)

    # ------------------------------------------------------------------ recorded moves (optional)
    def _load_recorded_moves(self) -> None:
        if not self._s.use_recorded_emotions:
            return
        try:
            from reachy_mini.motion.recorded_move import RecordedMoves

            self._recorded_moves = RecordedMoves(self._s.emotions_dataset)
            names = self._recorded_moves.list_moves()
            for emotion, keywords in RECORDED_KEYWORDS.items():
                match = next((n for kw in keywords for n in names if kw in n.lower()), None)
                if match:
                    self._recorded_index[emotion] = match
            log.info("Recorded emotions mapped: %s", self._recorded_index)
        except Exception as e:
            log.warning("Recorded emotion library unavailable (%s); using built-in choreographies.", e)
            self._recorded_moves = None

    def _play_recorded(self, emotion: str) -> bool:
        name = self._recorded_index.get(emotion)
        if not (self._recorded_moves and name):
            return False
        try:
            self._mini.play_move(self._recorded_moves.get(name), initial_goto_duration=0.5, sound=False)
            return True
        except Exception as e:
            log.warning("play_move(%s) failed: %s", name, e)
            return False
