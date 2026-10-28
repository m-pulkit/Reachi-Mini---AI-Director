"""Standalone pyttsx3 speaker, run as a subprocess by the Action node.

Why a subprocess? pyttsx3 drivers (SAPI5 on Windows, NSSpeech on macOS) are not thread-safe,
often hang on a second `runAndWait()` from a non-main thread, and cannot be interrupted.
A fresh process per utterance is robust, never blocks the robot's threads, and can be killed
instantly on Ctrl+C.

Usage: python tts_worker.py --rate 185 --volume 1.0 --voice "zira" "Text to speak"
"""

import argparse
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("text")
    p.add_argument("--rate", type=int, default=185)
    p.add_argument("--volume", type=float, default=1.0)
    p.add_argument("--voice", default="")
    a = p.parse_args()
    try:
        import pyttsx3
    except ImportError:
        print("pyttsx3 not installed", file=sys.stderr)
        return 2
    engine = pyttsx3.init()
    engine.setProperty("rate", a.rate)
    engine.setProperty("volume", max(0.0, min(1.0, a.volume)))
    if a.voice:
        for v in engine.getProperty("voices"):
            if a.voice.lower() in (v.name or "").lower() or a.voice.lower() in (v.id or "").lower():
                engine.setProperty("voice", v.id)
                break
    engine.say(a.text)
    engine.runAndWait()
    engine.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
