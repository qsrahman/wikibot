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


def _dim(text: str) -> str:
    return _style(text, "dim")


def _amber(text: str) -> str:
    return _style(text, "amber")


def _red(text: str) -> str:
    return _style(text, "red")


class _StreamPrinter:
    """Prints the response stream.

    Tracks the in-progress answer line so a tool call, an error, or Ctrl-C
    can break out of it cleanly instead of printing in the middle of a word.
    """

    def __init__(self) -> None:
        self.streaming = False

    def _break_line(self) -> None:
        if self.streaming:
            print()
            self.streaming = False

    def tool(self, name: str, args: dict) -> None:
        self._break_line()
        print(_dim(f"  -> {name}({json.dumps(args, ensure_ascii=False)})"))

    def delta(self, text: str) -> None:
        if not self.streaming:
            print()
            self.streaming = True
        print(text, end="", flush=True)

    def message(self, text: str) -> None:
        """Standalone final message, shown when nothing was streamed."""
        if not self.streaming:
            print(f"\n{text}", end="")

    def finish(self) -> None:
        self._break_line()


class Repl:
    """Prompt loop owning one conversation history for one Agent."""

    def __init__(self, agent: Agent):
        self.agent = agent
        self.history: list[dict] = []

    def _answer(self, user: str) -> None:
        """Run one message and print its tool calls and streamed answer."""
        history_len = len(self.history)
        printer = _StreamPrinter()
        try:
            for event in self.agent.run_turn(self.history, user):
                match event:
                    case ("tool", name, args):
                        printer.tool(name, args)
                    case ("delta", text):
                        printer.delta(text)
                    case ("done", text):
                        printer.message(text)
            print()
        except KeyboardInterrupt:
            printer.finish()
            del self.history[history_len:]
            print(_amber("aborted — history rolled back to last complete turn"))
        except Exception as exc:  # noqa: BLE001
            printer.finish()
            del self.history[history_len:]
            print(_red(f"error: {type(exc).__name__}: {exc}"))

    def run(self) -> None:
        print("wikibot — answers from English Wikipedia.")
        print("commands: exit | quit | .reset (clear history)")

        while True:
            try:
                user = input("> ").strip()
            except EOFError, KeyboardInterrupt:
                print()
                return
            if not user:
                continue
            if user.lower() in {"exit", "quit"}:
                return
            if user == ".reset":
                self.history.clear()
                print(_amber("history cleared"))
                continue
            self._answer(user)


def main() -> None:
    cfg = Config.from_env()
    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    Repl(Agent(client, cfg.model)).run()
