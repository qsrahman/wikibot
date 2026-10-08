"""Shared env loading for all entrypoints. Blank string means absent."""

import os
from dataclasses import dataclass


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"error: {name} is not set — fill it in .env (see README.md)")
    return value


@dataclass(frozen=True)
class Config:
    api_key: str
    model: str
    base_url: str | None

    @classmethod
    def from_env(cls) -> "Config":
        api_key = _require_env("OPENAI_API_KEY")
        model = _require_env("OPENAI_MODEL")
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
        return cls(api_key=api_key, model=model, base_url=base_url)