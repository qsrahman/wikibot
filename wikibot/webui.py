"""Gradio web UI: uv run --env-file .env python -m wikibot.webui"""

import json

import gradio as gr
from openai import OpenAI

from . import agent
from .config import load_config

api_key, model, base_url = load_config()
_client = OpenAI(api_key=api_key, base_url=base_url)


def respond(user_msg: str, display: list, agent_hist: list):
    """Chat generator: echo the user message, then stream tool steps + answer."""
    if not user_msg.strip():
        yield display, "", agent_hist
        return

    display.append({"role": "user", "content": user_msg})
    yield display, "", agent_hist

    text_index: int | None = None
    tool_index: int | None = None

    def close_step():
        nonlocal tool_index
        if tool_index is not None:
            display[tool_index]["metadata"]["status"] = "done"
            tool_index = None

    try:
        for event in agent.stream_turn(_client, model, agent_hist, user_msg):
            if event[0] == "delta":
                if text_index is None:
                    display.append({"role": "assistant", "content": ""})
                    text_index = len(display) - 1
                display[text_index]["content"] += event[1]
            elif event[0] == "tool":
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
            elif event[0] == "done":
                close_step()
                full_text = event[1]
                if text_index is not None:
                    display[text_index]["content"] = full_text
                elif full_text:
                    display.append({"role": "assistant", "content": full_text})
            yield display, "", agent_hist
    except Exception as exc:
        close_step()
        display.append(
            {"role": "assistant", "content": f"**Error:** `{type(exc).__name__}: {exc}`"}
        )
        yield display, "", agent_hist


def clear():
    return [], "", []


with gr.Blocks(title="wikibot") as demo:
    gr.Markdown("# wikibot — answers from English Wikipedia")
    chatbot = gr.Chatbot(label="Conversation", height=560)
    msg = gr.Textbox(
        lines=1,
        placeholder="Ask about anything on Wikipedia…",
        show_label=False,
        container=False,
    )
    with gr.Row():
        clear_btn = gr.Button("Clear", scale=0)
    agent_hist = gr.State([])

    msg.submit(respond, [msg, chatbot, agent_hist], [chatbot, msg, agent_hist])
    clear_btn.click(clear, None, [chatbot, msg, agent_hist])

demo.queue()


if __name__ == "__main__":
    demo.launch()
