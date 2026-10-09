"""Gradio web UI: uv run --env-file .env python -m wikibot.webui

Centered-column layout: a chat list on the left, one reading column, and the
lookup steps of each answer folded into a single line.
"""

import html
import uuid
from collections import defaultdict

import gradio as gr
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    NotFoundError,
    OpenAI,
    RateLimitError,
)

from .agent import Agent
from .config import Config

config = Config.from_env()

_THEME = gr.themes.Soft()

# Layout only: every colour is a theme variable, so light and dark both follow
# the theme.
CSS = """
footer { display: none !important; }
.gradio-container, .gradio-container .main { max-width: none !important; padding: 0 !important; }
#shell { gap: 0; flex-wrap: nowrap; height: 100dvh; }
/* Gradio outlines every output of a running handler; here that is the whole page */
#shell .generating { border: 0 !important; }

/* chat list */
#side { background: var(--background-fill-secondary); border-right: 1px solid var(--border-color-primary); padding: 12px; gap: 6px; flex: none; width: 260px; height: 100%; font-size: 14px; }
#side .brand { font-weight: 650; font-size: 16px; padding: 8px 10px 6px; }
#side .model { color: var(--body-text-color-subdued); font-size: 12.5px; padding: 10px; border-top: 1px solid var(--border-color-primary); }
#new-chat { margin-bottom: 8px; }
#chats { flex: 1; overflow-y: auto; border: 0 !important; background: none; padding: 0; }
#chats .wrap { flex-direction: column; flex-wrap: nowrap; gap: 2px; }
#chats label { display: block; width: 100%; padding: 7px 10px; border: 0; border-radius: var(--radius-lg); background: none; box-shadow: none; cursor: pointer; }
#chats label input { display: none; }
#chats label span { margin: 0; font-size: 14px; color: var(--body-text-color); display: block; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
#chats label:hover { background: var(--background-fill-primary); }
#chats label.selected { background: var(--color-accent-soft); }
#delete-chat { border: 0; background: none; box-shadow: none; color: var(--body-text-color-subdued); font-size: 13px; font-weight: 400; justify-content: flex-start; }
#delete-chat:hover { color: var(--error-text-color); }

/* reading column */
#main { height: 100%; gap: 0; min-width: 0; background: var(--background-fill-primary); }
#chatbot { flex: 1; min-height: 0; height: auto !important; border: 0; background: none; }
#chatbot .bubble-wrap, #chatbot .message-wrap { background: none; }
#chatbot .message-wrap { max-width: 740px; width: 100%; margin: 0 auto; padding: 36px 24px 12px; }
#compose, #main .foot { max-width: 692px; width: calc(100% - 48px); margin: 0 auto; }

/* messages */
#chatbot .message { box-shadow: none; font-size: 16px; line-height: 1.65; }
#chatbot .user { background: var(--color-accent-soft); border: 0; border-radius: 22px; padding: 10px 18px; }
#chatbot .bot, #chatbot .bot .message { background: none; border: 0; padding: 0; }
#chatbot .bot-row .flex-wrap { width: 100%; max-width: 100%; }
#chatbot .message-buttons-right { display: none; }
#chatbot .bot a { color: var(--link-text-color); }

/* lookup steps */
#chatbot .thought-group { margin: 0 0 6px; padding: 0; border: 0; background: none; }
#chatbot .thought-group .content .md, #chatbot .thought-group .content li { color: var(--body-text-color-subdued); font-size: 14px; }
#chatbot .thought-group .title { padding: 0; background: none; color: var(--body-text-color-subdued); font-size: 14px; }
#chatbot .thought-group .title .md, #chatbot .thought-group .title .md p { color: var(--body-text-color-subdued); font-size: 14px; margin: 0; }
#chatbot .thought-group .content { border-left: 2px solid var(--border-color-primary); padding-left: 14px; margin: 6px 0 4px; }
#chatbot .thought-group .content ul { list-style: none; margin: 0; padding: 0; }

/* empty screen */
#chatbot .placeholder-content { justify-content: center; gap: 22px; }
#chatbot .placeholder { flex: none; height: auto; }
#chatbot .placeholder h1 { margin: 0; font-size: 30px; font-weight: 600; letter-spacing: -0.02em; text-align: center; }
#chatbot .examples { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin: 0; }
#chatbot .example { width: auto; flex: none; border: 1px solid var(--border-color-primary); background: var(--background-fill-primary); border-radius: 999px; padding: 7px 14px; font-size: 14px; color: var(--body-text-color); box-shadow: none; }
#chatbot .example:hover { background: var(--background-fill-secondary); }

/* composer */
#compose { position: relative; border: 1px solid var(--border-color-primary); background: var(--input-background-fill); border-radius: 26px; padding: 8px 54px 8px 20px; box-shadow: var(--shadow-drop-lg); flex: none; }
#compose:focus-within { border-color: var(--color-accent); }
#compose .block, #compose label, #compose .input-container { border: 0 !important; box-shadow: none !important; background: none !important; }
#compose textarea { border: 0 !important; box-shadow: none !important; background: none; font-size: 16px; padding: 7px 0; }
#send { position: absolute; right: 8px; bottom: 8px; width: 38px !important; min-width: 38px !important; max-width: 38px !important; height: 38px; padding: 0; border-radius: 50%; font-size: 18px; line-height: 1; }
#main .foot { text-align: center; color: var(--body-text-color-subdued); font-size: 12.5px; padding: 10px 0 14px; flex: none; }

/* one clear affordance: the Chatbot's built-in Clear only resets visible chat */
#chatbot button[aria-label="Clear"] { display: none; }

@media (max-width: 760px) { #side { display: none; } }
"""

EXAMPLES = [
    "Who was Ada Lovelace?",
    "How tall is the Eiffel Tower?",
    "When was the first Turing Award given?",
]

# How each tool call reads in the lookup steps; filled from the call's arguments.
_STEP_TEXT = {
    "search_wikipedia": "Searched for “{query}”",
    "get_page_summary": "Read the summary of {title}",
    "get_page_content": "Read {title}",
    "get_page_section": "Read “{section}” in {title}",
    "get_page_links": "Listed the links in {title}",
    "get_random_pages": "Picked random articles",
}
_READ_TOOLS = ("get_page_summary", "get_page_content", "get_page_section")


def _step_line(name: str, args: dict) -> str:
    return _STEP_TEXT.get(name, name).format_map(defaultdict(str, args))


def _steps_title(calls: list[tuple[str, dict]]) -> str:
    """Summary line for a finished group of tool calls."""
    read = {args.get("title") for name, args in calls if name in _READ_TOOLS}
    parts = []
    if any(name == "search_wikipedia" for name, _ in calls):
        parts.append("searched Wikipedia")
    if read:
        parts.append(f"read {len(read)} article{'' if len(read) == 1 else 's'}")
    text = " and ".join(parts) or "looked on Wikipedia"
    return text[0].upper() + text[1:]


class WikiBotApp:
    """Gradio app: one Agent, a list of chats per browser, centered-column UI.

    The Blocks tree is built by :meth:`build`, which wires submit/click/load
    handlers to the instance methods below. Handlers take the BrowserState
    dict ``{"chats": [{"id", "display", "agent"}, ...], "active": id}``
    (newest chat first, persisted to localStorage), mutate it in place and
    return it with the components that mirror it.
    """

    def __init__(self, cfg: Config):
        self.config = cfg
        self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        self.model = cfg.model
        self.agent = Agent(self.client, cfg.model)

    @staticmethod
    def _box(interactive=True):
        return gr.Textbox(
            value="",
            interactive=interactive,
            autofocus=interactive,
            lines=1,
            max_lines=8,
            placeholder="Ask about anything on Wikipedia",
            show_label=False,
            container=False,
        )

    def _friendly_error(self, exc: Exception) -> tuple[str, str]:
        name, detail = type(exc).__name__, str(exc)
        if isinstance(exc, APIConnectionError):
            return (
                "Can't reach the model gateway",
                "Is it running? Check OPENAI_BASE_URL in .env.",
            )
        if isinstance(exc, RateLimitError):
            return "Rate limited by the gateway", "Wait a moment and try again."
        if isinstance(exc, AuthenticationError):
            return "Authentication failed", "Check OPENAI_API_KEY in .env."
        if isinstance(exc, NotFoundError):
            return (
                "Model or endpoint not found",
                f"Check OPENAI_MODEL (got: {self.model}).",
            )
        if isinstance(exc, APIStatusError):
            return f"Gateway error (HTTP {exc.status_code})", detail
        return name, detail

    @staticmethod
    def _active(browser: dict) -> dict:
        """The chat being shown; starts one if there is none."""
        chats = browser["chats"]
        chat = next((c for c in chats if c["id"] == browser["active"]), None)
        if chat is None:
            if not chats:
                chats.append({"id": uuid.uuid4().hex[:8], "display": [], "agent": []})
            chat = chats[0]
            browser["active"] = chat["id"]
        return chat

    @staticmethod
    def _title(chat: dict) -> str:
        """A chat is listed under its first question."""
        return next(
            (m["content"] for m in chat["display"] if m["role"] == "user"), "New chat"
        )

    def _frame(self, browser: dict, busy: bool = False):
        """Values for build()'s ``outs``. Busy locks everything that could
        change the active chat while an answer is streaming into it."""
        chat = self._active(browser)
        return (
            chat["display"],
            self._box(interactive=not busy),
            browser,
            gr.Radio(
                choices=[(self._title(c), c["id"]) for c in browser["chats"]],
                value=chat["id"],
                interactive=not busy,
            ),
            gr.Button(interactive=not busy),
            gr.Button(interactive=not busy),
        )

    def respond(self, user_msg: str, browser: dict):
        """Chat generator: echo the user message, then stream steps + answer."""
        chat = self._active(browser)
        display, agent_hist = chat["display"], chat["agent"]

        if not user_msg.strip():
            yield self._frame(browser)
            return

        display.append({"role": "user", "content": user_msg})
        yield self._frame(browser, busy=True)

        text_index: int | None = None
        steps: dict | None = None  # the open lookup-steps message
        calls: list[tuple[str, dict]] = []  # tool calls listed in it

        def close_steps():
            nonlocal steps
            if steps is not None:
                steps["metadata"] = {"title": _steps_title(calls), "status": "done"}
                steps = None
                calls.clear()

        try:
            for event in self.agent.run_turn(agent_hist, user_msg):
                kind = event[0]
                if kind == "delta":
                    if text_index is None:
                        close_steps()
                        display.append({"role": "assistant", "content": ""})
                        text_index = len(display) - 1
                    display[text_index]["content"] += event[1]
                elif kind == "tool":
                    text_index = None
                    _, name, args = event
                    if steps is None:
                        steps = {
                            "role": "assistant",
                            "content": "",
                            "metadata": {
                                "title": "Looking on Wikipedia…",
                                "status": "pending",
                            },
                        }
                        display.append(steps)
                    calls.append((name, args))
                    steps["content"] += f"- {_step_line(name, args)}\n"
                elif kind == "tool_result":
                    digest = event[2]
                    # Sizes of fetched text are noise here; result lists and failures are not.
                    if (
                        steps is not None
                        and digest
                        and not digest.startswith(("summary:", "content:"))
                    ):
                        steps["content"] = f"{steps['content'].rstrip()}, {digest}\n"
                elif kind == "done":
                    close_steps()
                    full_text = event[1]
                    if text_index is not None:
                        display[text_index]["content"] = full_text
                    elif full_text:
                        display.append({"role": "assistant", "content": full_text})
                yield self._frame(browser, busy=True)
        except Exception as exc:  # noqa: BLE001
            close_steps()
            title, detail = self._friendly_error(exc)
            display.append(
                {"role": "assistant", "content": f"**{title}**\n\n`{detail}`"}
            )
        yield self._frame(browser)

    def new_chat(self, browser: dict):
        if self._active(browser)["display"]:
            chat = {"id": uuid.uuid4().hex[:8], "display": [], "agent": []}
            browser["chats"].insert(0, chat)
            browser["active"] = chat["id"]
        return self._frame(browser)

    def switch_chat(self, chat_id: str, browser: dict):
        browser["active"] = chat_id
        return self._frame(browser)

    def delete_chat(self, browser: dict):
        browser["chats"] = [c for c in browser["chats"] if c["id"] != browser["active"]]
        return self._frame(browser)

    @staticmethod
    def _example_text(evt: gr.SelectData) -> str:
        return evt.value["text"]

    def build(self):
        with gr.Blocks(title="wikibot") as demo:
            browser = gr.BrowserState(
                default_value={"chats": [], "active": None},
                storage_key="wikibot-chats",
                secret="wikibot-local-only",
            )
            with gr.Row(elem_id="shell"):
                with gr.Column(scale=0, min_width=260, elem_id="side"):
                    gr.HTML('<div class="brand">wikibot</div>')
                    new_btn = gr.Button("New chat", elem_id="new-chat")
                    chats = gr.Radio(show_label=False, container=False, elem_id="chats")
                    del_btn = gr.Button("Delete this chat", elem_id="delete-chat")
                    gr.HTML(
                        f'<div class="model">Model: {html.escape(self.model)}</div>'
                    )
                with gr.Column(elem_id="main"):
                    chatbot = gr.Chatbot(
                        show_label=False,
                        layout="bubble",
                        buttons=["copy"],
                        placeholder="<h1>What do you want to look up?</h1>",
                        examples=[{"text": text} for text in EXAMPLES],
                        elem_id="chatbot",
                    )
                    with gr.Row(elem_id="compose"):
                        msg = self._box()
                        send_btn = gr.Button(
                            "↑", variant="primary", elem_id="send", scale=0
                        )
                    gr.HTML(
                        '<div class="foot">Answers come from English Wikipedia. '
                        "Open the linked article to check anything important.</div>"
                    )

            outs = [chatbot, msg, browser, chats, new_btn, del_btn]
            demo.load(self._frame, [browser], outs)
            msg.submit(self.respond, [msg, browser], outs)
            send_btn.click(self.respond, [msg, browser], outs)
            chatbot.example_select(self._example_text, None, msg).then(
                self.respond, [msg, browser], outs
            )
            new_btn.click(self.new_chat, [browser], outs)
            chats.input(self.switch_chat, [chats, browser], outs)
            del_btn.click(self.delete_chat, [browser], outs)

        demo.queue()
        return demo


if __name__ == "__main__":
    WikiBotApp(config).build().launch(theme=_THEME, css=CSS, footer_links=[])
