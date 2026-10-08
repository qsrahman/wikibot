"""Gradio web UI: uv run --env-file .env python -m wikibot.webui"""

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

from . import agent
from .config import load_config

api_key, model, base_url = load_config()
_client = OpenAI(api_key=api_key, base_url=base_url)

EXAMPLES = [
    "Who was Alan Turing?",
    "How does photosynthesis work?",
    "What caused the fall of the Roman Empire?",
    "Explain how a Turing machine works",
]

CSS = """
#chat-col { max-width: 860px; margin: 0 auto; }
#wikibot-head h1 { margin: 0.2em 0 0.1em; }
#wikibot-head p { margin: 0 0 0.8em; }
"""


def _box(value="", interactive=True, autofocus=False):
    return gr.Textbox(
        value=value,
        interactive=interactive,
        autofocus=autofocus,
        lines=1,
        placeholder="Ask anything about…",
        show_label=False,
        container=False,
    )


def _friendly_error(exc: Exception) -> tuple[str, str]:
    name, detail = type(exc).__name__, str(exc)
    if isinstance(exc, APIConnectionError):
        return "Can't reach the model gateway", "Is it running? Check OPENAI_BASE_URL in .env."
    if isinstance(exc, RateLimitError):
        return "Rate limited by the gateway", "Wait a moment and try again."
    if isinstance(exc, AuthenticationError):
        return "Authentication failed", "Check OPENAI_API_KEY in .env."
    if isinstance(exc, NotFoundError):
        return "Model or endpoint not found", f"Check OPENAI_MODEL (got: {model})."
    if isinstance(exc, APIStatusError):
        return f"Gateway error (HTTP {exc.status_code})", detail
    return name, detail


def respond(user_msg: str, browser: dict):
    """Chat generator: echo the user message, then stream tool steps + answer.

    ``browser`` is a BrowserState dict {"display", "agent"} persisted to
    localStorage; it is mutated in place and yielded back each step.
    """
    display, agent_hist = browser["display"], browser["agent"]

    if not user_msg.strip():
        yield display, _box(autofocus=True), browser
        return

    display.append({"role": "user", "content": user_msg})
    yield display, _box(interactive=False), browser

    text_index: int | None = None
    tool_index: int | None = None

    def close_step():
        nonlocal tool_index
        if tool_index is not None:
            display[tool_index]["metadata"]["status"] = "done"
            tool_index = None

    try:
        for event in agent.stream_turn(_client, model, agent_hist, user_msg):
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
                        "metadata": {"title": f"🔧 {name}", "status": "pending"},
                    }
                )
                tool_index = len(display) - 1
            elif kind == "tool_result":
                _, name, digest = event
                if tool_index is not None:
                    if digest:
                        # Title is what the collapsed thought-group shows; the
                        # digest must live there or it is hidden behind a click.
                        display[tool_index]["content"] += f"\n\n→ {digest}"
                        display[tool_index]["metadata"]["title"] = (
                            f"🔧 {name} · {digest}"
                        )
                    display[tool_index]["metadata"]["status"] = "done"
                    tool_index = None
            elif kind == "done":
                close_step()
                full_text = event[1]
                if text_index is not None:
                    display[text_index]["content"] = full_text
                elif full_text:
                    display.append({"role": "assistant", "content": full_text})
            yield display, _box(interactive=False), browser
    except Exception as exc:
        close_step()
        title, detail = _friendly_error(exc)
        display.append({"role": "assistant", "content": f"**{title}**\n\n`{detail}`"})
    yield display, _box(interactive=True, autofocus=True), browser


def clear(browser: dict):
    return [], _box(autofocus=True), {"display": [], "agent": []}


def _hydrate(browser: dict):
    return browser["display"]


with gr.Blocks(title="wikibot — Wikipedia chat") as demo:
    browser = gr.BrowserState(
        default_value={"display": [], "agent": []},
        storage_key="wikibot",
        secret="wikibot-local-only",
    )
    with gr.Column(elem_id="chat-col"):
        with gr.Column(elem_id="wikibot-head"):
            gr.Markdown("# wikibot")
            gr.Markdown(f"Answers from English Wikipedia · model `{model}`")
        chatbot = gr.Chatbot(
            label="Conversation",
            height=560,
            buttons=["copy", "copy_all"],
            elem_id="chatbot",
        )
        msg = gr.Textbox(
            lines=1,
            placeholder="Ask anything about…",
            show_label=False,
            container=False,
            autofocus=True,
        )
        gr.Examples(
            examples=[[q] for q in EXAMPLES],
            inputs=[msg],
            label=None,
        )
        with gr.Row():
            clear_btn = gr.Button("Clear", scale=0)

    demo.load(_hydrate, inputs=[browser], outputs=[chatbot])
    msg.submit(respond, [msg, browser], [chatbot, msg, browser])
    clear_btn.click(clear, [browser], [chatbot, msg, browser])

demo.queue()


if __name__ == "__main__":
    demo.launch(css=CSS)
