"""REPL entrypoint: uv run --env-file .env -m wikibot"""

import json
import os
import sys

from openai import OpenAI

from .agent import Agent
from .config import Config

_STYLES = {"dim": "\x1b[2m", "amber": "\x1b[33m", "red": "\x1b[31m", "reset": "\x1b[0m"}
_USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _style(text: str, kind: str) -> str:
    if not _USE_COLOR:
        return text
    return f"{_STYLES[kind]}{text}{_STYLES['reset']}"


class Repl:
    """Prompt loop owning one conversation history for one Agent."""

    def __init__(self, agent: Agent):
        self.agent = agent
        self.history: list[dict] = []

    def _handle(self, user: str) -> None:
        streamed = False
        history_len = len(self.history)
        try:
            for event in self.agent.run_turn(self.history, user):
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
            del self.history[history_len:]
            print(_style("aborted — history rolled back to last complete turn", "amber"))
            return
        except Exception as exc:
            if streamed:
                print()
            print(_style(f"error: {type(exc).__name__}: {exc}", "red"))
            return
        print("\n")

    def run(self) -> None:
        print("wikibot — answers from English Wikipedia.")
        print("commands: exit | quit | .reset (clear history)")

        while True:
            try:
                user = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not user:
                continue
            if user.lower() in {"exit", "quit"}:
                return
            if user == ".reset":
                self.history.clear()
                print(_style("history cleared", "amber"))
                continue
            self._handle(user)


def main() -> None:
    cfg = Config.from_env()
    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    Repl(Agent(client, cfg.model)).run()