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

## Layout

| File | Purpose |
|------|---------|
| `wikibot/wiki.py` | Wikipedia adapter (search, summary, content) over `wikipediaapi` |
| `wikibot/agent.py` | Tool schemas + Responses API agent loop |
| `wikibot/cli.py` | REPL |

Wikipedia access is free and keyless (MediaWiki Action API); only the OpenAI
key is required. Set a contact address in the User-Agent if you run this at
scale — see `wikibot/wiki.py` (`USER_AGENT`).
