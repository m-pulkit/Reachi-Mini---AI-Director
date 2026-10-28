"""Entry point:  python main.py   (see README.md and .env.example)."""

import logging
import sys

from ai_director.app import DirectorApp
from ai_director.config import Settings


def main() -> int:
    settings = Settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)-14s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "urllib3", "google_genai", "anthropic", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    DirectorApp(settings).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
