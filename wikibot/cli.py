"""REPL entrypoint: uv run --env-file .env -m wikibot"""

import json
import os
import sys

from openai import OpenAI

from . import agent
from .config import load_config

_STYLES = {"dim": "\x1b[2m", "amber": "\x1b[33m", "red": "\x1b[31m", "reset": "\x1b[0m"}
_USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _style(text: str, kind: str) -> str:
    if not _USE_COLOR:
        return text
    return f"{_STYLES[kind]}{text}{_STYLES['reset']}"


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
            print(_style("history cleared", "amber"))
            continue
        streamed = False
        history_len = len(history)
        try:
            for event in agent.run_turn(client, model, history, user):
                if event[0] == "tool":
                    if streamed:
                        print()
                        streamed = False
                    args = json.dumps(event[2], ensure_ascii=False)
                    print(_style(f"  -> {event[1]}({args})", "dim"))
                elif event[0] == "delta":
                    if not streamed:
                        print()
                        streamed = True
                    print(event[1], end="", flush=True)
                elif event[0] == "done" and not streamed:
                    # No deltas: e.g. the tool-round-limit message.
                    print(f"\n{event[1]}", end="")
        except KeyboardInterrupt:
            if streamed:
                print()
            del history[history_len:]
            print(_style("aborted — history rolled back to last complete turn", "amber"))
            continue
        except Exception as exc:
            if streamed:
                print()
            print(_style(f"error: {type(exc).__name__}: {exc}", "red"))
            continue
        print("\n")
