"""Preview of Gradio's built-in themes on a wikibot-shaped sample chat.

    uv run python design/theme_gallery.py   ->  http://localhost:7861

Static sample content, no model or API key. Each theme is mounted on its own
path; the bar at the top of every page switches theme and light/dark mode.
"""

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

PORT = 7861
THEMES = [
    "Default", "Soft", "Monochrome", "Glass", "Origin", "Citrus",
    "Ocean", "Base", "Ember", "Neon", "Cyberpunk", "Mario",
]

SAMPLE = [
    {"role": "user", "content": "Who was Ada Lovelace?"},
    {
        "role": "assistant",
        "content": "- Searched for “Ada Lovelace”, 5 results\n- Read the summary of Ada Lovelace",
        "metadata": {"title": "Searched Wikipedia and read 1 article", "status": "done"},
    },
    {
        "role": "assistant",
        "content": (
            "Ada Lovelace was an English mathematician and writer known for her work on "
            "Charles Babbage's Analytical Engine. She is often considered the first "
            "computer programmer. [Ada Lovelace](https://en.wikipedia.org/wiki/Ada_Lovelace)"
        ),
    },
    {"role": "user", "content": "What was the Analytical Engine?"},
    {
        "role": "assistant",
        "content": (
            "A mechanical general-purpose computer designed by Charles Babbage, first "
            "described in 1837 as the successor to his Difference Engine. "
            "[Analytical engine](https://en.wikipedia.org/wiki/Analytical_engine)"
        ),
    },
]


def _nav(current: str) -> str:
    links = " ".join(
        f'<b>{name}</b>' if name == current else f'<a href="/{name.lower()}/">{name}</a>'
        for name in THEMES
    )
    path = f"/{current.lower()}/"
    return (
        f'<div style="display:flex;flex-wrap:wrap;gap:6px 14px;font-size:14px">{links}'
        f'<span style="margin-left:auto"><a href="{path}?__theme=light">Light</a> '
        f'<a href="{path}?__theme=dark">Dark</a></span></div>'
    )


def _sample(name: str) -> gr.Blocks:
    with gr.Blocks(title=f"{name} theme") as demo:
        gr.HTML(_nav(name))
        with gr.Row():
            with gr.Column(scale=1, min_width=220):
                gr.Markdown("### wikibot")
                gr.Button("New chat")
                gr.Radio(
                    ["Who was Ada Lovelace?", "First Turing Award", "World War II casualties"],
                    value="Who was Ada Lovelace?",
                    label="Chats",
                )
                gr.Button("Delete this chat", variant="stop")
            with gr.Column(scale=4):
                gr.Chatbot(SAMPLE, label="Conversation", layout="bubble", height=480, buttons=["copy"])
                with gr.Row():
                    gr.Textbox(placeholder="Ask about anything on Wikipedia", show_label=False, scale=5)
                    gr.Button("Send", variant="primary", scale=0)
    return demo


app = FastAPI()


@app.get("/")
def index():
    return RedirectResponse(f"/{THEMES[0].lower()}/")


for _name in THEMES:
    gr.mount_gradio_app(
        app, _sample(_name), path=f"/{_name.lower()}", theme=getattr(gr.themes, _name)(), footer_links=[]
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
