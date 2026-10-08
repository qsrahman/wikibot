"""Shared env loading for all entrypoints. Blank string means absent."""

import os


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"error: {name} is not set — fill it in .env (see README.md)")
    return value


def load_config() -> tuple[str, str, str | None]:
    api_key = _require_env("OPENAI_API_KEY")
    model = _require_env("OPENAI_MODEL")
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    return api_key, model, base_url
