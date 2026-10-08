"""REPL entrypoint: uv run --env-file .env -m wikibot"""

from openai import OpenAI

from . import agent
from .config import load_config


def main() -> None:
    api_key, model, base_url = load_config()

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
