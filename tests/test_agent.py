import json
import unittest
from types import SimpleNamespace

from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from wikibot.agent import Agent


def call(name: str, arguments, call_id: str = "c1") -> ResponseFunctionToolCall:
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return ResponseFunctionToolCall(
        type="function_call", id=f"fc_{call_id}", call_id=call_id, name=name, arguments=arguments, status="completed"
    )


def message(text: str) -> ResponseOutputMessage:
    part = ResponseOutputText(type="output_text", text=text, annotations=[])
    return ResponseOutputMessage(id="msg_1", type="message", role="assistant", status="completed", content=[part])


class FakeStream:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        return iter(self.events)

    def __exit__(self, *exc):
        return False


class FakeClient:
    """Stands in for OpenAI: each responses.create() plays the next scripted round.

    A round is the list of output items the model produced; text of message
    items is streamed as deltas first, one per word.
    """

    def __init__(self, *rounds, complete: bool = True):
        self.rounds = list(rounds)
        self.complete = complete
        self.requests: list[dict] = []
        self.responses = self

    def create(self, **kwargs):
        # history is mutated in place later, so keep what was sent at this point.
        self.requests.append({**kwargs, "input": list(kwargs["input"])})
        items = self.rounds.pop(0) if len(self.rounds) > 1 else self.rounds[0]
        text = "".join(part.text for item in items if item.type == "message" for part in item.content)
        events = [
            SimpleNamespace(type="response.output_text.delta", delta=word)
            for word in text.replace(" ", " \0").split("\0")
            if word
        ]
        if self.complete:
            response = SimpleNamespace(output=items, output_text=text)
            events.append(SimpleNamespace(type="response.completed", response=response))
        return FakeStream(events)


class FakeWiki:
    def search_wikipedia(self, query, limit):
        return {"pages": [{"title": "Ada Lovelace", "snippet": "", "url": "https://en.wikipedia.org/wiki/Ada_Lovelace"}]}

    def page_summary(self, title):
        if title == "Boom":
            raise RuntimeError("wiki down")
        return {"found": True, "title": title, "summary": "A mathematician.", "url": "u"}


def run(client, **agent_kwargs):
    history: list[dict] = []
    agent = Agent(client, "test-model", FakeWiki(), **agent_kwargs)
    return list(agent.run_turn(history, "Who was Ada Lovelace?")), history


def assert_calls_paired(test: unittest.TestCase, history: list[dict]):
    """Every function_call is followed, before the next request, by its output."""
    calls = [h["call_id"] for h in history if h.get("type") == "function_call"]
    outputs = [h["call_id"] for h in history if h.get("type") == "function_call_output"]
    test.assertEqual(calls, outputs)


SEARCH = call("search_wikipedia", {"query": "Ada Lovelace", "limit": 5}, "c1")
SUMMARY = call("get_page_summary", {"title": "Ada Lovelace"}, "c2")


class RunTurn(unittest.TestCase):
    def test_events_run_tool_then_delta_then_done(self):
        events, _ = run(FakeClient([SEARCH], [SUMMARY], [message("She was a mathematician.")]))
        self.assertEqual(
            events,
            [
                ("tool", "search_wikipedia", {"query": "Ada Lovelace", "limit": 5}),
                ("tool_result", "search_wikipedia", "1 results: Ada Lovelace"),
                ("tool", "get_page_summary", {"title": "Ada Lovelace"}),
                ("tool_result", "get_page_summary", "summary: 16 chars"),
                ("delta", "She "),
                ("delta", "was "),
                ("delta", "a "),
                ("delta", "mathematician."),
                ("done", "She was a mathematician."),
            ],
        )

    def test_history_holds_the_whole_turn_in_api_shape(self):
        _, history = run(FakeClient([SEARCH], [message("Done.")]))
        self.assertEqual(
            [h.get("type", h.get("role")) for h in history],
            ["user", "function_call", "function_call_output", "message"],
        )
        self.assertEqual(json.loads(history[2]["output"])["pages"][0]["title"], "Ada Lovelace")
        # Output text goes back as input text, without the output-only annotations.
        self.assertEqual(history[3]["content"], [{"type": "input_text", "text": "Done."}])
        assert_calls_paired(self, history)

    def test_each_request_sends_the_history_so_far(self):
        client = FakeClient([SEARCH], [message("Done.")])
        run(client)
        first, second = client.requests
        self.assertEqual(first["input"], [{"role": "user", "content": "Who was Ada Lovelace?"}])
        self.assertEqual([h.get("type") for h in second["input"][1:]], ["function_call", "function_call_output"])
        self.assertEqual(first["model"], "test-model")
        self.assertIs(first["stream"], True)

    def test_parallel_calls_in_one_round_are_all_answered(self):
        events, history = run(FakeClient([SEARCH, SUMMARY], [message("Done.")]))
        self.assertEqual([e[1] for e in events if e[0] == "tool"], ["search_wikipedia", "get_page_summary"])
        assert_calls_paired(self, history)

    def test_tool_failures_are_data_not_exceptions(self):
        cases = {
            "malformed arguments": (call("get_page_summary", "{not json"), {}, "error: KeyError: 'title'"),
            "unknown tool": (call("delete_wikipedia", {}), {}, "error: unknown tool: delete_wikipedia"),
            "wiki raises": (
                call("get_page_summary", {"title": "Boom"}),
                {"title": "Boom"},
                "error: RuntimeError: wiki down",
            ),
        }
        for label, (bad_call, arguments, digest) in cases.items():
            with self.subTest(label):
                events, history = run(FakeClient([bad_call], [message("Sorry.")]))
                self.assertEqual(events[0], ("tool", bad_call.name, arguments))
                self.assertEqual(events[1], ("tool_result", bad_call.name, digest))
                self.assertEqual(events[-1], ("done", "Sorry."))
                assert_calls_paired(self, history)

    def test_round_limit_stops_with_a_message_and_paired_history(self):
        client = FakeClient([SEARCH])  # the model never stops calling tools
        events, history = run(client, max_tool_rounds=3)
        self.assertEqual(len(client.requests), 3)
        self.assertEqual(events[-1], ("done", "Stopped: tool call limit reached (3 rounds) for this message."))
        assert_calls_paired(self, history)

    def test_stream_that_never_completes_raises(self):
        with self.assertRaisesRegex(RuntimeError, "without completing"):
            run(FakeClient([message("partial")], complete=False))


class Digest(unittest.TestCase):
    def test_one_line_summaries(self):
        pages = [{"title": t} for t in "ABCDE"]
        cases = [
            ("search_wikipedia", {"pages": pages}, "5 results: A, B, C (+2 more)"),
            ("search_wikipedia", {"pages": pages[:2]}, "2 results: A, B"),
            ("search_wikipedia", {"pages": []}, "no results"),
            ("get_page_links", {"found": True, "links": pages[:4]}, "4 links: A, B, C (+1 more)"),
            ("get_random_pages", {"pages": pages[:4]}, "4 random pages: A, B, C"),
            ("get_page_summary", {"found": True, "summary": "x" * 1234}, "summary: 1,234 chars"),
            ("get_page_content", {"found": True, "text": "x" * 8000, "truncated": True}, "content: 8,000 chars, truncated"),
            ("get_page_section", {"found": True, "section": "S", "text": "xy", "truncated": False}, "content: 2 chars"),
            ("get_page_summary", {"found": False, "title": "T"}, "page not found"),
            ("get_page_section", {"found": False, "title": "T", "section": "S"}, "section not found"),
            ("get_page_summary", {"error": "e" * 150}, "error: " + "e" * 100),
        ]
        for name, result, expected in cases:
            with self.subTest(expected):
                self.assertEqual(Agent._digest(name, json.dumps(result)), expected)

    def test_unparseable_output_has_no_digest(self):
        self.assertEqual(Agent._digest("search_wikipedia", "not json"), "")
