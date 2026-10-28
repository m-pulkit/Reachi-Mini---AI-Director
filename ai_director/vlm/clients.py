"""Vision-Language Model clients. All return a validated `CritiqueResult` or raise `VLMError`.

Each provider is forced into strict JSON:
  * Gemini   -> response_mime_type="application/json" + response_schema (pydantic model)
  * Anthropic-> a single forced tool call whose input_schema is the pydantic JSON schema
  * Mock     -> offline heuristic critic for testing the whole pipeline without an API key
"""

from __future__ import annotations

import base64
import json
import logging
import os
import random
import re
from abc import ABC, abstractmethod

from pydantic import ValidationError

from ..config import BrainSettings
from ..messages import CritiqueResult, TakeEnded, TimedFrame
from ..prompts import FINAL_INSTRUCTION, SYSTEM_PROMPT, build_take_context

log = logging.getLogger(__name__)


class VLMError(RuntimeError):
    """Any failure to obtain a valid critique (network, auth, timeout, bad JSON)."""


def parse_critique(raw: str | dict) -> CritiqueResult:
    """Validate model output; tolerate code fences or chatter around the JSON object."""
    try:
        if isinstance(raw, dict):
            return CritiqueResult.model_validate(raw)
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise VLMError(f"No JSON object in model output: {raw[:200]!r}")
        return CritiqueResult.model_validate(json.loads(text[start : end + 1]))
    except (json.JSONDecodeError, ValidationError) as e:
        raise VLMError(f"Model returned invalid JSON: {e}") from e


class VLMClient(ABC):
    name = "base"

    def __init__(self, cfg: BrainSettings) -> None:
        self.cfg = cfg

    @abstractmethod
    def critique(self, take: TakeEnded, frames: list[TimedFrame]) -> CritiqueResult: ...


# --------------------------------------------------------------------------- Gemini
class GeminiClient(VLMClient):
    name = "gemini"

    def __init__(self, cfg: BrainSettings) -> None:
        super().__init__(cfg)
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise VLMError("pip install google-genai") from e
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise VLMError("GEMINI_API_KEY is not set")
        self._types = types
        self._client = genai.Client(
            api_key=key, http_options=types.HttpOptions(timeout=int(cfg.request_timeout_s * 1000))
        )

    def critique(self, take: TakeEnded, frames: list[TimedFrame]) -> CritiqueResult:
        t = self._types
        parts = [t.Part.from_text(text=build_take_context(take, len(frames)))]
        for f in frames:
            parts.append(t.Part.from_text(text=f"[frame t={f.t:05.1f}s]"))
            parts.append(t.Part.from_bytes(data=f.jpeg, mime_type="image/jpeg"))
        parts.append(t.Part.from_text(text=FINAL_INSTRUCTION))
        try:
            resp = self._client.models.generate_content(
                model=self.cfg.gemini_model,
                contents=[t.Content(role="user", parts=parts)],
                config=t.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=CritiqueResult,
                    temperature=0.9,
                    max_output_tokens=max(self.cfg.max_output_tokens, 2048),  # 2.5 models spend tokens thinking
                ),
            )
        except Exception as e:
            raise VLMError(f"Gemini request failed: {e}") from e
        if getattr(resp, "parsed", None) is not None:
            parsed = resp.parsed
            return parsed if isinstance(parsed, CritiqueResult) else parse_critique(dict(parsed))
        if not resp.text:
            raise VLMError("Gemini returned an empty response (safety block or token limit).")
        return parse_critique(resp.text)


# --------------------------------------------------------------------------- Anthropic
class AnthropicClient(VLMClient):
    name = "anthropic"
    TOOL_NAME = "submit_verdict"

    def __init__(self, cfg: BrainSettings) -> None:
        super().__init__(cfg)
        try:
            import anthropic
        except ImportError as e:
            raise VLMError("pip install anthropic") from e
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise VLMError("ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic(timeout=cfg.request_timeout_s, max_retries=0)
        self._tool = {
            "name": self.TOOL_NAME,
            "description": "Submit the final algorithmic verdict for this take.",
            "input_schema": CritiqueResult.model_json_schema(),
        }

    def critique(self, take: TakeEnded, frames: list[TimedFrame]) -> CritiqueResult:
        content: list[dict] = [{"type": "text", "text": build_take_context(take, len(frames))}]
        for f in frames:
            content.append({"type": "text", "text": f"[frame t={f.t:05.1f}s]"})
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg",
                           "data": base64.standard_b64encode(f.jpeg).decode("ascii")},
            })
        content.append({"type": "text", "text": FINAL_INSTRUCTION + f" Call the {self.TOOL_NAME} tool."})
        try:
            resp = self._client.messages.create(
                model=self.cfg.anthropic_model,
                max_tokens=self.cfg.max_output_tokens,
                system=SYSTEM_PROMPT,
                tools=[self._tool],
                tool_choice={"type": "tool", "name": self.TOOL_NAME},
                messages=[{"role": "user", "content": content}],
            )
        except Exception as e:
            raise VLMError(f"Anthropic request failed: {e}") from e
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                return parse_critique(block.input)
        raise VLMError("Anthropic response contained no tool call.")


# --------------------------------------------------------------------------- Mock
class MockClient(VLMClient):
    """Offline critic: crude heuristics so you can test robot + TTS + state machine for free."""

    name = "mock"

    def critique(self, take: TakeEnded, frames: list[TimedFrame]) -> CritiqueResult:
        words = sum(len(s.text.split()) for s in take.transcript)
        wps = words / max(take.duration_s, 1.0)
        first_line = take.transcript[0] if take.transcript else None
        hook = 7 if first_line and first_line.t < 1.5 else 3  # talking within 1.5 s = some hook
        pacing = int(min(10, max(1, wps * 3.5)))
        visuals = random.randint(4, 8) if frames else 1
        avg = (hook + pacing + visuals) / 3
        verdict = "PASS" if avg >= 6.5 else "FAIL"
        if verdict == "PASS":
            line, emo = "Okay, the hook actually stopped my thumb. Post it before you overthink it.", "joy"
        elif not take.transcript:
            line, emo = "Thirty seconds of silence. Even the algorithm fell asleep.", "boredom"
        elif hook <= 3:
            line, emo = "You lost them at hello. I swiped before you said anything worth saying.", "disappointment"
        else:
            line, emo = "The middle sagged like a Monday morning. Cut the dead air and try again.", "disappointment"
        return CritiqueResult(hook_score=hook, pacing_score=pacing, visuals_score=visuals,
                              overall_verdict=verdict, spoken_critique=line, reachy_emotion=emo)


def make_client(cfg: BrainSettings) -> VLMClient:
    providers = {"gemini": GeminiClient, "anthropic": AnthropicClient, "claude": AnthropicClient, "mock": MockClient}
    cls = providers.get(cfg.provider)
    if cls is None:
        raise VLMError(f"Unknown VLM_PROVIDER '{cfg.provider}'. Use one of {sorted(providers)}.")
    return cls(cfg)
