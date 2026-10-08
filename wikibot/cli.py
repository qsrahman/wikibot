"""REPL entrypoint: uv run --env-file .env -m wikibot"""

import json

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
            user = input("> ").strip()
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
        streamed = False
        try:
            for event in agent.run_turn(client, model, history, user):
                if event[0] == "tool":
                    if streamed:
                        print()
                        streamed = False
                    print(f"  -> {event[1]}({json.dumps(event[2], ensure_ascii=False)})")
                elif event[0] == "delta":
                    if not streamed:
                        print()
                        streamed = True
                    print(event[1], end="", flush=True)
                elif event[0] == "done" and not streamed:
                    # No deltas: e.g. the tool-round-limit message.
                    print(f"\n{event[1]}", end="")
        except Exception as exc:
            if streamed:
                print()
            print(f"error: {type(exc).__name__}: {exc}")
            continue
        print("\n")
