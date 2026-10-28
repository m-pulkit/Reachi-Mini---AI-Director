"""Typed messages that travel through the inter-node queues, plus the VLM output schema."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Emotion = Literal["joy", "disappointment", "anger", "neutral", "surprise", "boredom"]
EMOTIONS: tuple[str, ...] = ("joy", "disappointment", "anger", "neutral", "surprise", "boredom")


# --------------------------------------------------------------------------- VLM output
class CritiqueResult(BaseModel):
    """Strict schema the VLM must return. Lenient *parsing* (e.g. "7/10" -> 7), strict *shape*."""

    model_config = ConfigDict(extra="ignore")

    hook_score: int = Field(..., ge=0, le=10, description="Strength of the first 1-3 seconds, out of 10.")
    pacing_score: int = Field(..., ge=0, le=10, description="Dead air / rhythm / momentum, out of 10.")
    visuals_score: int = Field(..., ge=0, le=10, description="Framing, lighting, kinesthetic energy, out of 10.")
    overall_verdict: Literal["PASS", "FAIL"]
    spoken_critique: str = Field(..., min_length=1, max_length=400, description="1-2 sentences, spoken aloud.")
    reachy_emotion: Emotion

    @field_validator("hook_score", "pacing_score", "visuals_score", mode="before")
    @classmethod
    def _parse_score(cls, v: object) -> object:
        if isinstance(v, str):
            m = re.search(r"\d+(\.\d+)?", v)
            if m:
                return max(0, min(10, round(float(m.group()))))
        if isinstance(v, float):
            return max(0, min(10, round(v)))
        return v

    @field_validator("overall_verdict", mode="before")
    @classmethod
    def _parse_verdict(cls, v: object) -> object:
        return v.strip().upper() if isinstance(v, str) else v

    @field_validator("reachy_emotion", mode="before")
    @classmethod
    def _parse_emotion(cls, v: object) -> object:
        if isinstance(v, str):
            v = v.strip().lower()
            synonyms = {"happy": "joy", "excited": "joy", "sad": "disappointment", "disgust": "disappointment",
                        "angry": "anger", "furious": "anger", "bored": "boredom", "shocked": "surprise"}
            v = synonyms.get(v, v)
            return v if v in EMOTIONS else "neutral"
        return v

    @property
    def average_score(self) -> float:
        return (self.hook_score + self.pacing_score + self.visuals_score) / 3.0


# --------------------------------------------------------------------------- queue messages
@dataclass(frozen=True)
class TimedFrame:
    t: float  # seconds since the take started
    jpeg: bytes


@dataclass(frozen=True)
class TranscriptSegment:
    t: float  # seconds since the take started (approx. start of the phrase)
    text: str


@dataclass(frozen=True)
class TakeEnded:
    """Audio -> Brain: the user said "Cut!". Frames are pulled from the Vision node by the Brain."""

    take_id: int
    duration_s: float
    transcript: list[TranscriptSegment] = field(default_factory=list)


@dataclass(frozen=True)
class SayCommand:
    """Anything -> Action: a short line + gesture outside the critique flow (acks, errors)."""

    text: str
    emotion: Emotion = "neutral"
    return_to_idle: bool = False  # Action node moves FSM to IDLE afterwards


@dataclass(frozen=True)
class PoseCommand:
    """Anything -> Action: a silent gesture (e.g. 'attention' when the take starts)."""

    name: str


@dataclass(frozen=True)
class CritiqueCommand:
    """Brain -> Action: the full verdict."""

    take_id: int
    result: CritiqueResult
    brain_offline: bool = False


class Shutdown:
    """Poison pill pushed into queues on exit."""
