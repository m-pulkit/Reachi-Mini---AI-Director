"""Central configuration, loaded from environment variables (and an optional .env file).

Every tunable lives here so the nodes never read os.environ directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # pragma: no cover
    pass


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_int(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw.lstrip("-").isdigit() else None


@dataclass(frozen=True)
class RobotSettings:
    # "auto" = try physical, then fall back to the MuJoCo sim.
    # "physical" = physical only (fail if absent). "sim" = go straight to the simulator.
    mode: str = field(default_factory=lambda: _env_str("REACHY_MODE", "auto").lower())
    host: str = field(default_factory=lambda: _env_str("REACHY_HOST", "reachy-mini.local"))
    connect_timeout_s: float = field(default_factory=lambda: _env_float("REACHY_CONNECT_TIMEOUT", 5.0))
    # Simulator: run MuJoCo without its 3D viewer window (starts faster), keep it running after exit.
    sim_headless: bool = field(default_factory=lambda: _env_bool("SIM_HEADLESS", False))
    sim_keep_alive: bool = field(default_factory=lambda: _env_bool("SIM_KEEP_ALIVE", False))
    # Webcam index used when the robot/sim camera returns no frames (-1 disables).
    fallback_camera_index: int = field(default_factory=lambda: _env_int("FALLBACK_CAMERA_INDEX", 0))
    # Use Pollen's recorded emotion library (HF dataset) when available.
    use_recorded_emotions: bool = field(default_factory=lambda: _env_bool("USE_RECORDED_EMOTIONS", False))
    emotions_dataset: str = field(
        default_factory=lambda: _env_str("EMOTIONS_DATASET", "pollen-robotics/reachy-mini-emotions-library")
    )


@dataclass(frozen=True)
class VisionSettings:
    capture_hz: float = field(default_factory=lambda: _env_float("CAPTURE_HZ", 20.0))  # live camera loop
    sample_fps: float = field(default_factory=lambda: _env_float("SAMPLE_FPS", 2.0))  # frames kept per second of take
    max_take_seconds: float = field(default_factory=lambda: _env_float("MAX_TAKE_SECONDS", 90.0))
    max_frames_to_vlm: int = field(default_factory=lambda: _env_int("MAX_FRAMES_TO_VLM", 60))
    frame_width: int = field(default_factory=lambda: _env_int("FRAME_WIDTH", 640))
    jpeg_quality: int = field(default_factory=lambda: _env_int("JPEG_QUALITY", 80))
    face_tracking_hz: float = field(default_factory=lambda: _env_float("FACE_TRACKING_HZ", 6.0))


@dataclass(frozen=True)
class AudioSettings:
    mic_device_index: int | None = field(default_factory=lambda: _env_optional_int("MIC_DEVICE_INDEX"))
    # Substring used to auto-pick a microphone by name (e.g. "Reachy"); empty = system default.
    mic_name_hint: str = field(default_factory=lambda: _env_str("MIC_NAME_HINT", ""))
    phrase_time_limit_s: float = field(default_factory=lambda: _env_float("PHRASE_TIME_LIMIT", 6.0))
    pause_threshold_s: float = field(default_factory=lambda: _env_float("PAUSE_THRESHOLD", 0.6))
    language: str = field(default_factory=lambda: _env_str("SPEECH_LANGUAGE", "en-US"))
    # If true, "rolling" must be preceded by "Reachy" (or a common mis-hearing of it).
    require_robot_name: bool = field(default_factory=lambda: _env_bool("REQUIRE_ROBOT_NAME", False))


@dataclass(frozen=True)
class BrainSettings:
    provider: str = field(default_factory=lambda: _env_str("VLM_PROVIDER", "gemini").lower())  # gemini|anthropic|mock
    gemini_model: str = field(default_factory=lambda: _env_str("GEMINI_MODEL", "gemini-2.5-flash"))
    anthropic_model: str = field(default_factory=lambda: _env_str("ANTHROPIC_MODEL", "claude-sonnet-4-5"))
    request_timeout_s: float = field(default_factory=lambda: _env_float("VLM_TIMEOUT", 45.0))
    max_retries: int = field(default_factory=lambda: _env_int("VLM_MAX_RETRIES", 1))
    max_output_tokens: int = field(default_factory=lambda: _env_int("VLM_MAX_TOKENS", 1024))


@dataclass(frozen=True)
class ActionSettings:
    tts_rate: int = field(default_factory=lambda: _env_int("TTS_RATE", 185))
    tts_volume: float = field(default_factory=lambda: _env_float("TTS_VOLUME", 1.0))
    tts_voice_hint: str = field(default_factory=lambda: _env_str("TTS_VOICE_HINT", ""))
    tts_timeout_s: float = field(default_factory=lambda: _env_float("TTS_TIMEOUT", 30.0))


@dataclass(frozen=True)
class Settings:
    robot: RobotSettings = field(default_factory=RobotSettings)
    vision: VisionSettings = field(default_factory=VisionSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    brain: BrainSettings = field(default_factory=BrainSettings)
    action: ActionSettings = field(default_factory=ActionSettings)
    show_preview: bool = field(default_factory=lambda: _env_bool("SHOW_PREVIEW", True))
    save_takes_dir: str = field(default_factory=lambda: _env_str("SAVE_TAKES_DIR", ""))  # empty = don't save
    log_level: str = field(default_factory=lambda: _env_str("LOG_LEVEL", "INFO").upper())
