"""Gradio web UI: uv run --env-file .env python -m wikibot.webui

Telescope layout: a dark instrument console with a sessions rail and traced
tool-call log lines.
"""

import html
import json

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

BG = "#0c1119"
PANEL = "#101620"
RAIL = "#151d28"
RULE = "#223043"
INK = "#cdd6e0"
DIM = "#77889c"
AMBER = "#e2a552"
AMBER_DIM = "#b98a4a"

_THEME = gr.themes.Base(
    primary_hue="amber",
    neutral_hue="zinc",
    radius_size=gr.themes.sizes.radius_sm,
).set(
    body_background_fill=BG,
    body_text_color=INK,
    block_background_fill=PANEL,
    block_border_color=RULE,
    background_fill_primary=PANEL,
    background_fill_secondary=RAIL,
    panel_background_fill=RAIL,
    panel_border_color=RULE,
    border_color_primary=RULE,
    border_color_accent=AMBER_DIM,
    color_accent=AMBER,
    color_accent_soft="#2b2921",
    input_background_fill=RAIL,
    input_border_color=RULE,
    input_border_color_focus=AMBER_DIM,
    code_background_fill=BG,
    link_text_color=AMBER,
    link_text_color_hover=AMBER,
    button_primary_background_fill=AMBER,
    button_primary_text_color=BG,
    button_secondary_background_fill=RAIL,
    button_secondary_border_color=RULE,
    button_secondary_text_color=INK,
    error_background_fill="#241d1d",
    error_border_color="#6e4a4a",
    error_text_color="#f0b8b8",
)

CSS = """
html, body, .gradio-container { background: var(--tg-bg, #0c1119) !important; }

/* instrument bar */
#topbar { padding: 14px 26px; background: #0c1119; border-bottom: 1px solid #223043; }
#topbar .brand { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 15px; font-weight: 700; color: #e2a552; }
#topbar .brand em { font-style: normal; color: #cdd6e0; }
#topbar .meta { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px; color: #77889c; }

/* sessions rail */
#shell { gap: 0; border-top: 1px solid #223043; }
#rail { background: #151d28; border-right: 1px solid #223043; padding: 20px 14px; }
#rail .rail-label { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 11px; letter-spacing: 0.09em; color: #77889c; text-transform: uppercase; margin: 0 6px 8px; }
#rail .item { padding: 7px 10px; border-radius: 6px; color: #77889c; font-size: 13.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; cursor: default; }
#rail .item + .item { margin-top: 2px; }
#rail .item.active { background: #1d2a3d; color: #cdd6e0; }
#rail .rail-empty { color: #55606e; font-size: 12.5px; margin: 4px 6px; line-height: 1.6; }
#rail-clear { margin-top: auto; }

/* conversation column */
#convo { background: #101620; overflow: hidden; }
#convo .message-row { background: transparent; border: none; box-shadow: none; }

/* role labels */
#convo .user-row .md::before {
  content: "you ▸";
  display: block;
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  font-size: 12px;
  color: #77889c;
  margin-bottom: 3px;
}
#convo .bot-row .md:not(.thought-group .md)::before {
  content: "wikibot ▸";
  display: block;
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  font-size: 12px;
  color: #e2a552;
  margin: 14px 0 3px;
}

/* tool-call log lines */
#convo .thought-group { margin: 2px 0 0; }
#convo .thought-group .title {
  display: flex; align-items: center; gap: 6px;
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  font-size: 12.5px;
  background: #151d28;
  border-radius: 6px;
  padding: 5px 10px;
}
#convo .thought-group .title[aria-busy="true"] .md { color: #e2a552; }
#convo .thought-group .title .md, #convo .thought-group .title .md p { margin: 0; color: #cdd6e0; }
#convo .thought-group .arrow { color: #55606e; }

/* answer prose */
#convo .bot-row .md:not(.thought-group .md) p { font-size: 15px; line-height: 1.7; color: #cdd6e0; }
#convo .bot-row .md:not(.thought-group .md) a { color: #e2a552; text-decoration: underline; text-decoration-color: #6a5c3f; }
#convo .bot-row .md:not(.thought-group .md) a:hover { color: #f0b56a; }
#convo .bot-row .md:not(.thought-group .md) code { background: #151d28; color: #d8c08a; border-radius: 4px; padding: 1px 5px; }

/* composer */
#compose { position: relative; padding: 18px 26px 22px; background: #0c1119; border-top: 1px solid #223043; }
#compose textarea { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 14px; padding-right: 54px; }
#compose textarea::placeholder { color: #55606e; }

/* run button as an icon inside the input box */
#run-button {
  position: absolute;
  top: 50%;
  right: 34px;
  transform: translateY(-50%);
  z-index: 5;
  width: 34px !important;
  min-width: 34px !important;
  max-width: 34px !important;
  height: 34px !important;
  padding: 0 !important;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  line-height: 1;
  font-size: 13px;
}
#run-button .wrap {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0 !important;
}

/* one clear affordance: the Chatbot's built-in Clear only resets visible chat */
#chatbot button[aria-label="Clear"] { display: none; }
"""


class WikiBotApp:
    """Gradio app: one Agent, a per-tab chat history, and the telescope UI.

    The Blocks tree is built by :meth:`build`, which wires submit/click/load
    handlers to the instance methods below.
    """

    def __init__(self, cfg: Config):
        self.config = cfg
        self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        self.model = cfg.model
        self.agent = Agent(self.client, cfg.model)

    @staticmethod
    def _box(value="", interactive=True, autofocus=False):
        return gr.Textbox(
            value=value,
            interactive=interactive,
            autofocus=autofocus,
            lines=1,
            placeholder="ask anything about…",
            show_label=False,
            container=False,
        )

    def _friendly_error(self, exc: Exception) -> tuple[str, str]:
        name, detail = type(exc).__name__, str(exc)
        if isinstance(exc, APIConnectionError):
            return "Can't reach the model gateway", "Is it running? Check OPENAI_BASE_URL in .env."
        if isinstance(exc, RateLimitError):
            return "Rate limited by the gateway", "Wait a moment and try again."
        if isinstance(exc, AuthenticationError):
            return "Authentication failed", "Check OPENAI_API_KEY in .env."
        if isinstance(exc, NotFoundError):
            return "Model or endpoint not found", f"Check OPENAI_MODEL (got: {self.model})."
        if isinstance(exc, APIStatusError):
            return f"Gateway error (HTTP {exc.status_code})", detail
        return name, detail

    @staticmethod
    def _sessions_html(display: list[dict]) -> str:
        """The question rail: one row per user turn, last one active."""
        turns = [m["content"] for m in display if m.get("role") == "user"]
        if not turns:
            return '<div class="rail-label">Sessions</div><div class="rail-empty">No questions yet —<br>ask anything about…</div>'
        items = []
        for i, turn in enumerate(turns):
            active = " active" if i == len(turns) - 1 else ""
            items.append(f'<div class="item{active}">{html.escape(turn)}</div>')
        return '<div class="rail-label">Sessions</div>' + "".join(items)

    def respond(self, user_msg: str, browser: dict):
        """Chat generator: echo the user message, then stream tool log + answer.

        ``browser`` is a BrowserState dict {"display", "agent"} persisted to
        localStorage; it is mutated in place and yielded back each step.
        """
        display, agent_hist = browser["display"], browser["agent"]

        if not user_msg.strip():
            yield display, self._box(autofocus=True), browser, self._sessions_html(display)
            return

        display.append({"role": "user", "content": user_msg})
        yield display, self._box(interactive=False), browser, self._sessions_html(display)

        text_index: int | None = None
        tool_index: int | None = None

        def close_step():
            nonlocal tool_index
            if tool_index is not None:
                display[tool_index]["metadata"]["status"] = "done"
                tool_index = None

        try:
            for event in self.agent.run_turn(agent_hist, user_msg):
                kind = event[0]
                if kind == "delta":
                    if text_index is None:
                        display.append({"role": "assistant", "content": ""})
                        text_index = len(display) - 1
                    display[text_index]["content"] += event[1]
                elif kind == "tool":
                    close_step()
                    text_index = None
                    _, name, args = event
                    display.append(
                        {
                            "role": "assistant",
                            "content": f"`{json.dumps(args, ensure_ascii=False)}`",
                            "metadata": {"title": f"⇢ {name}", "status": "pending"},
                        }
                    )
                    tool_index = len(display) - 1
                elif kind == "tool_result":
                    _, name, digest = event
                    if tool_index is not None:
                        if digest:
                            display[tool_index]["content"] += f"\n\n{digest}"
                            display[tool_index]["metadata"]["title"] = f"✔ {digest}"
                        else:
                            display[tool_index]["metadata"]["title"] = f"✔ {name}"
                        display[tool_index]["metadata"]["status"] = "done"
                        tool_index = None
                elif kind == "done":
                    close_step()
                    full_text = event[1]
                    if text_index is not None:
                        display[text_index]["content"] = full_text
                    elif full_text:
                        display.append({"role": "assistant", "content": full_text})
                yield display, self._box(interactive=False), browser, self._sessions_html(display)
        except Exception as exc:
            close_step()
            title, detail = self._friendly_error(exc)
            display.append({"role": "assistant", "content": f"**{title}**\n\n`{detail}`"})
        yield display, self._box(interactive=True, autofocus=True), browser, self._sessions_html(display)

    def clear(self, browser: dict):
        return [], self._box(autofocus=True), {"display": [], "agent": []}, self._sessions_html([])

    def _hydrate(self, browser: dict):
        return browser["display"], self._sessions_html(browser["display"])

    def build(self):
        with gr.Blocks(title="wikibot — Wikipedia console") as demo:
            browser = gr.BrowserState(
                default_value={"display": [], "agent": []},
                storage_key="wikibot",
                secret="wikibot-local-only",
            )
            gr.Markdown(
                f'<span class="brand">wiki<em>bot</em></span> '
                f'<span class="meta">answers from English Wikipedia, model {self.model}</span>',
                elem_id="topbar",
            )
            with gr.Row(elem_id="shell"):
                with gr.Column(scale=1, min_width=200, elem_id="rail"):
                    rail = gr.HTML(self._sessions_html([]))
                    clear_btn = gr.Button("clear conversation", elem_id="rail-clear", scale=0)
                with gr.Column(scale=5, elem_id="convo"):
                    chatbot = gr.Chatbot(
                        show_label=False,
                        height=540,
                        layout="panel",
                        buttons=["copy"],
                        elem_id="chatbot",
                    )
                    with gr.Row(elem_id="compose"):
                        msg = gr.Textbox(
                            lines=1,
                            placeholder="ask anything about…",
                            show_label=False,
                            container=False,
                            autofocus=True,
                            scale=5,
                        )
                        run_btn = gr.Button("▶", elem_id="run-button", size="sm", scale=0)

            demo.load(self._hydrate, inputs=[browser], outputs=[chatbot, rail])
            msg.submit(self.respond, [msg, browser], [chatbot, msg, browser, rail])
            run_btn.click(self.respond, [msg, browser], [chatbot, msg, browser, rail])
            clear_btn.click(self.clear, [browser], [chatbot, msg, browser, rail])

        demo.queue()
        return demo


if __name__ == "__main__":
    WikiBotApp(config).build().launch(theme=_THEME, css=CSS)