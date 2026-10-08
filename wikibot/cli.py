"""REPL entrypoint: uv run --env-file .env -m wikibot"""

import os
import sys

from openai import OpenAI

from . import agent


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        sys.exit(f"error: {name} is not set — fill it in .env (see README.md)")
    return value


def main() -> None:
    api_key = _require_env("OPENAI_API_KEY")
    model = _require_env("OPENAI_MODEL")
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None

    client = OpenAI(api_key=api_key, base_url=base_url)
    history: list = []

    print("wikibot — answers from English Wikipedia.")
    print("commands: exit | quit | .reset (clear history)")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break
        if user == ".reset":
            history.clear()
            print("history cleared")
            continue
        try:
            answer = agent.run_turn(
                client,
                model,
                history,
                user,
                on_tool=lambda line: print(f"  -> {line}"),
            )
        except Exception as exc:
            print(f"error: {type(exc).__name__}: {exc}")
            continue
        print(f"bot> {answer}")
