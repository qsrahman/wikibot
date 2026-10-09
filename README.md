# wikibot

CLI chatbot that answers questions from English Wikipedia, using the OpenAI
Responses API with Wikipedia search/read tools.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

1. Fill in `.env`:

   ```
   OPENAI_API_KEY=sk-...
   OPENAI_MODEL=gpt-4o-mini
   OPENAI_BASE_URL=            # optional, any OpenAI-compatible endpoint
   ```

2. Run:

   ```bash
   uv run --env-file .env -m wikibot
   ```

## REPL

```
you> Who was Alan Turing?
  -> search_wikipedia({"query": "Alan Turing", "limit": 5})
  -> get_page_summary({"title": "Alan Turing"})
bot> Alan Turing was an English mathematician ... [Alan Turing](https://en.wikipedia.org/wiki/Alan_Turing)
```

Commands: `exit` / `quit` to leave, `.reset` to clear conversation history.

## Web UI

```bash
uv run --env-file .env python -m wikibot.webui
```

Opens a Gradio chat app on http://localhost:7860 with:

- **centered-column layout** — a chat list on the left and one reading column;
  your question in a pill on the right, the answer as plain text
- **several chats** — New chat starts one, the list switches between them,
  Delete this chat removes the open one; listed under Today, Yesterday and
  Earlier by when they were last used
- **Try again** — the button under the last answer asks that question afresh
- **streaming answers** — live tokens; the input is locked while the agent works
- **lookup steps** — each answer's tool calls fold into one line such as
  "Searched Wikipedia and read 1 article"; click it for the details
- citations as clickable links that open in a new tab
- **persistent chats** across reloads (localStorage); an answer cut off by a
  closed tab is marked "Lookup interrupted" and the chat stays usable
- on a phone-width screen the chat list becomes a strip across the top
- copy buttons and friendly error bubbles

Same backend as the CLI — `wikibot/agent.py` + `wikibot/wiki.py` are shared.

## Layout

| File | Purpose |
|------|---------|
| `wikibot/wiki.py` | Wikipedia adapter (search, summary, content) over `wikipediaapi` |
| `wikibot/agent.py` | Tool schemas + Responses API agent loop (`run_turn` events) |
| `wikibot/cli.py` | REPL |
| `wikibot/webui.py` | Gradio web UI |
| `wikibot/config.py` | Shared `.env` loading for both entrypoints |

Wikipedia access is free and keyless (MediaWiki Action API); only the OpenAI
key is required. Set a contact address in the User-Agent if you run this at
scale — see `wikibot/wiki.py` (`USER_AGENT`).
