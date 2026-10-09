import os
import time
import unittest
from unittest import mock

import httpx
from openai import APIConnectionError

# webui reads its config from the environment when imported.
with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test", "OPENAI_MODEL": "test-model"}):
    from wikibot import webui


class FakeAgent:
    def __init__(self, *events, error: Exception | None = None):
        self.events, self.error = events, error

    def run_turn(self, history, user_message):
        history.append({"role": "user", "content": user_message})
        yield from self.events
        if self.error:
            raise self.error
        history.append({"type": "message", "role": "assistant"})


def app(*events, error=None) -> webui.WikiBotApp:
    wikibot = webui.WikiBotApp(webui.config)
    wikibot.agent = FakeAgent(*events, error=error)
    return wikibot


def empty() -> dict:
    return {"chats": [], "active": None}


def ask(wikibot, browser, text="Who was Ada Lovelace?"):
    """Run respond() to the end; returns every frame it yielded."""
    return list(wikibot.respond(text, browser))


def shown(browser) -> list[dict]:
    return webui.WikiBotApp._active(browser)["display"]


def controls(frame):
    """The parts of a frame a user can act on: input box, chat lists, buttons."""
    return (frame[1], *frame[3:])


def listed(frame) -> dict[str, list[str]]:
    """Chat titles under each visible heading of the chat list."""
    return {
        group: [title for title, _ in chats.choices]
        for group, chats in zip(webui.GROUPS, frame[3:6])
        if chats.visible
    }


SEARCH = ("tool", "search_wikipedia", {"query": "Ada Lovelace", "limit": 5})
SEARCHED = ("tool_result", "search_wikipedia", "1 results: Ada Lovelace")
READ = ("tool", "get_page_summary", {"title": "Ada Lovelace"})
WAS_READ = ("tool_result", "get_page_summary", "summary: 16 chars")


class Respond(unittest.TestCase):
    def test_steps_fold_into_one_message_above_the_answer(self):
        browser = empty()
        ask(app(SEARCH, SEARCHED, READ, WAS_READ, ("delta", "She "), ("delta", "was."), ("done", "She was.")), browser)
        self.assertEqual(
            shown(browser),
            [
                {"role": "user", "content": "Who was Ada Lovelace?"},
                {
                    "role": "assistant",
                    "content": "- Searched for “Ada Lovelace”, 1 results: Ada Lovelace\n- Read the summary of Ada Lovelace\n",
                    "metadata": {"title": "Searched Wikipedia and read 1 article", "status": "done"},
                },
                {"role": "assistant", "content": "She was."},
            ],
        )

    def test_steps_show_as_pending_while_the_lookup_runs(self):
        browser = empty()
        frames = app(SEARCH, SEARCHED, ("done", "ok")).respond("q", browser)
        next(frames)  # the echoed question
        next(frames)  # after the tool call
        self.assertEqual(shown(browser)[1]["metadata"], {"title": "Looking on Wikipedia…", "status": "pending"})
        list(frames)
        self.assertEqual(shown(browser)[1]["metadata"]["status"], "done")

    def test_text_between_tool_calls_starts_a_new_steps_group(self):
        browser = empty()
        ask(app(SEARCH, SEARCHED, ("delta", "Checking."), READ, WAS_READ, ("delta", "Done."), ("done", "Done.")), browser)
        self.assertEqual(
            [m.get("metadata", {}).get("title", m["content"]) for m in shown(browser)[1:]],
            ["Searched Wikipedia", "Checking.", "Read 1 article", "Done."],
        )

    def test_final_text_shows_even_when_nothing_streamed(self):
        browser = empty()
        ask(app(("done", "Stopped: tool call limit reached")), browser)
        self.assertEqual(shown(browser)[-1], {"role": "assistant", "content": "Stopped: tool call limit reached"})

    def test_failure_becomes_an_error_bubble_and_closes_open_steps(self):
        browser = empty()
        error = APIConnectionError(request=httpx.Request("POST", "http://gateway"))
        ask(app(SEARCH, error=error), browser)
        steps, bubble = shown(browser)[1:]
        self.assertEqual(steps["metadata"]["status"], "done")
        self.assertIn("Can't reach the model gateway", bubble["content"])
        self.assertIn("OPENAI_BASE_URL", bubble["content"])

    def test_blank_message_changes_nothing(self):
        browser = empty()
        frames = ask(app(("done", "never")), browser, "   ")
        self.assertEqual(len(frames), 1)
        self.assertEqual(shown(browser), [])

    def test_controls_lock_while_streaming_and_unlock_after(self):
        frames = ask(app(("delta", "Hi"), ("done", "Hi")), empty())
        for frame in frames[:-1]:
            self.assertEqual({c.interactive for c in controls(frame)}, {False})
        self.assertEqual({c.interactive for c in controls(frames[-1])}, {True})

    def test_agent_history_is_kept_per_chat(self):
        wikibot, browser = app(("done", "ok")), empty()
        ask(wikibot, browser, "first")
        wikibot.new_chat(browser)
        ask(wikibot, browser, "second")
        self.assertEqual(
            [chat["agent"][0]["content"] for chat in browser["chats"]], ["second", "first"]
        )


class Chats(unittest.TestCase):
    def test_first_load_starts_one_empty_chat(self):
        browser = empty()
        frame = app().load(browser)
        self.assertEqual(frame[0], [])
        self.assertEqual(len(browser["chats"]), 1)
        self.assertEqual(frame[3].choices, [("New chat", browser["active"])])
        self.assertEqual(frame[3].value, browser["active"])
        self.assertEqual(listed(frame), {"Today": ["New chat"]})

    def test_chats_are_listed_newest_first_under_their_first_question(self):
        wikibot, browser = app(("done", "ok")), empty()
        ask(wikibot, browser, "first question")
        ask(wikibot, browser, "a follow-up")
        wikibot.new_chat(browser)
        self.assertEqual(listed(wikibot._frame(browser)), {"Today": ["New chat", "first question"]})
        self.assertEqual(shown(browser), [])

    def test_asking_in_an_older_chat_moves_it_to_the_top(self):
        wikibot, browser = app(("done", "ok")), empty()
        ask(wikibot, browser, "older")
        older = browser["active"]
        wikibot.new_chat(browser)
        ask(wikibot, browser, "newer")
        wikibot.switch_chat(older, browser)
        frames = ask(wikibot, browser, "again")
        self.assertEqual(listed(frames[-1]), {"Today": ["older", "newer"]})

    def test_chats_are_grouped_by_the_day_they_were_last_used(self):
        day = 86400
        ages = {"just now": 0, "a day ago": day, "last week": 6 * day}
        chats = [
            {"id": title, "display": [{"role": "user", "content": title}], "agent": [], "updated": time.time() - age}
            for title, age in ages.items()
        ]
        chats.append({"id": "old", "display": [{"role": "user", "content": "no timestamp"}], "agent": []})
        browser = {"chats": chats, "active": "a day ago"}
        frame = app()._frame(browser)
        self.assertEqual(
            listed(frame),
            {"Today": ["just now"], "Yesterday": ["a day ago"], "Earlier": ["last week", "no timestamp"]},
        )
        # Only the list holding the open chat has a selection.
        self.assertEqual([chats.value for chats in frame[3:6]], [None, "a day ago", None])

    def test_new_chat_reuses_an_empty_chat(self):
        wikibot, browser = app(), empty()
        wikibot.new_chat(browser)
        wikibot.new_chat(browser)
        self.assertEqual(len(browser["chats"]), 1)

    def test_switch_shows_the_other_chat(self):
        wikibot, browser = app(("done", "ok")), empty()
        ask(wikibot, browser, "first")
        first = browser["active"]
        wikibot.new_chat(browser)
        display = wikibot.switch_chat(first, browser)[0]
        self.assertEqual(display[0]["content"], "first")
        self.assertEqual(browser["active"], first)

    def test_delete_removes_both_histories_and_falls_back(self):
        wikibot, browser = app(("done", "ok")), empty()
        ask(wikibot, browser, "first")
        wikibot.new_chat(browser)
        ask(wikibot, browser, "second")
        wikibot.delete_chat(browser)
        self.assertEqual([c["display"][0]["content"] for c in browser["chats"]], ["first"])
        wikibot.delete_chat(browser)
        wikibot._frame(browser)
        self.assertEqual([(c["display"], c["agent"]) for c in browser["chats"]], [([], [])])


class Load(unittest.TestCase):
    def test_a_cut_off_answer_is_repaired(self):
        call = {"type": "function_call", "call_id": "c2", "name": "search_wikipedia", "arguments": "{}"}
        answered = {"type": "function_call", "call_id": "c1", "name": "search_wikipedia", "arguments": "{}"}
        output = {"type": "function_call_output", "call_id": "c1", "output": "{}"}
        user = {"role": "user", "content": "q"}
        browser = {
            "active": "a",
            "chats": [
                {
                    "id": "a",
                    "display": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "- step\n", "metadata": {"title": "Looking on Wikipedia…", "status": "pending"}},
                    ],
                    "agent": [user, answered, call, output],
                }
            ],
        }
        app().load(browser)
        chat = browser["chats"][0]
        self.assertEqual(chat["display"][1]["metadata"], {"title": "Lookup interrupted", "status": "done"})
        # The call that never got an output would make the API reject the chat.
        self.assertEqual(chat["agent"], [user, answered, output])


class TryAgain(unittest.TestCase):
    def test_last_turn_is_dropped_from_both_histories_and_asked_again(self):
        wikibot, browser = app(("delta", "first answer"), ("done", "first answer")), empty()
        ask(wikibot, browser, "one")
        ask(wikibot, browser, "two")
        wikibot.agent = FakeAgent(("delta", "better"), ("done", "better"))
        frames = list(wikibot.retry(browser))
        self.assertEqual(
            [m["content"] for m in shown(browser)], ["one", "first answer", "two", "better"]
        )
        chat = webui.WikiBotApp._active(browser)
        self.assertEqual(
            [h.get("content", h.get("type")) for h in chat["agent"]], ["one", "message", "two", "message"]
        )
        self.assertEqual({c.interactive for c in controls(frames[-1])}, {True})

    def test_retry_after_a_failed_turn_replaces_the_error(self):
        wikibot, browser = app(error=RuntimeError("boom")), empty()
        ask(wikibot, browser, "one")
        wikibot.agent = FakeAgent(("done", "ok"))
        list(wikibot.retry(browser))
        self.assertEqual([m["content"] for m in shown(browser)], ["one", "ok"])
        self.assertEqual(len(webui.WikiBotApp._active(browser)["agent"]), 2)

    def test_nothing_to_retry_in_an_empty_chat(self):
        browser = empty()
        self.assertEqual(len(list(app().retry(browser))), 1)
        self.assertEqual(shown(browser), [])


class StepWording(unittest.TestCase):
    def test_each_tool_reads_as_a_sentence(self):
        cases = [
            ("search_wikipedia", {"query": "Ada", "limit": 5}, "Searched for “Ada”"),
            ("get_page_summary", {"title": "Ada"}, "Read the summary of Ada"),
            ("get_page_content", {"title": "Ada"}, "Read Ada"),
            ("get_page_section", {"title": "Ada", "section": "Life"}, "Read “Life” in Ada"),
            ("get_page_links", {"title": "Ada", "limit": 5}, "Listed the links in Ada"),
            ("get_random_pages", {"limit": 3}, "Picked random articles"),
            ("get_page_summary", {}, "Read the summary of "),  # malformed arguments arrive as {}
            ("some_new_tool", {"x": 1}, "some_new_tool"),
        ]
        for name, args, expected in cases:
            with self.subTest(name=name, args=args):
                self.assertEqual(webui._step_line(name, args), expected)

    def test_summary_line_counts_distinct_articles(self):
        read = lambda title, tool="get_page_summary": (tool, {"title": title})  # noqa: E731
        cases = [
            ([("search_wikipedia", {})], "Searched Wikipedia"),
            ([read("A")], "Read 1 article"),
            ([("search_wikipedia", {}), read("A"), read("A", "get_page_section"), read("B")], "Searched Wikipedia and read 2 articles"),
            ([("get_random_pages", {})], "Looked on Wikipedia"),
        ]
        for calls, expected in cases:
            with self.subTest(expected):
                self.assertEqual(webui._steps_title(calls), expected)
