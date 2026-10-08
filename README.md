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

- **telescope layout** — a dark instrument console: instrument bar, a Sessions
  rail, and the composer in a single command-line row
- **streaming answers** — live tokens; the input is locked while the agent works
- **traced tool calls** — each dispatch shows as a log line: `⇢ search_wikipedia {...}`
  resolving to `✔ 5 results: …`
- citations as clickable links that open in a new tab
- **persistent history** across reloads (localStorage); **Clear** wipes it all
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
