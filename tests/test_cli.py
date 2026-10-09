import contextlib
import io
import unittest
from unittest import mock

from wikibot import cli


class FakeAgent:
    def __init__(self, *events, error: BaseException | None = None):
        self.events, self.error = events, error

    def run_turn(self, history, user_message):
        history.append({"role": "user", "content": user_message})
        yield from self.events
        if self.error:
            raise self.error
        history.append({"type": "message"})


def answer(agent, history=()):
    repl = cli.Repl(agent)
    repl.history.extend(history)
    out = io.StringIO()
    with mock.patch.object(cli, "_USE_COLOR", False), contextlib.redirect_stdout(out):
        repl._answer("question")
    return out.getvalue(), repl.history


TOOL = ("tool", "search_wikipedia", {"query": "é", "limit": 1})


class Answer(unittest.TestCase):
    def test_tool_lines_then_streamed_answer(self):
        out, history = answer(FakeAgent(TOOL, ("tool_result", "search_wikipedia", "1 results"), ("delta", "Hel"), ("delta", "lo"), ("done", "Hello")))
        self.assertEqual(out, '  -> search_wikipedia({"query": "é", "limit": 1})\n\nHello\n')
        self.assertEqual(len(history), 2)

    def test_tool_call_after_text_starts_on_its_own_line(self):
        out, _ = answer(FakeAgent(("delta", "Checking"), TOOL, ("delta", "Hi"), ("done", "Hi")))
        self.assertEqual(out, '\nChecking\n  -> search_wikipedia({"query": "é", "limit": 1})\n\nHi\n')

    def test_final_message_prints_when_nothing_streamed(self):
        out, _ = answer(FakeAgent(("done", "Stopped: tool call limit reached")))
        self.assertEqual(out, "\nStopped: tool call limit reached\n")

    def test_error_keeps_partial_text_and_rolls_history_back(self):
        earlier = [{"role": "user", "content": "before"}]
        out, history = answer(FakeAgent(("delta", "par"), error=RuntimeError("boom")), earlier)
        self.assertEqual(out, "\npar\nerror: RuntimeError: boom\n")
        self.assertEqual(history, earlier)

    def test_ctrl_c_rolls_history_back_and_keeps_the_session(self):
        out, history = answer(FakeAgent(TOOL, error=KeyboardInterrupt()))
        self.assertIn("aborted", out)
        self.assertEqual(history, [])
