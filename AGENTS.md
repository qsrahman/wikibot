# AGENTS.md — Developer Guide

Companion to [README.md](README.md) (user-facing usage). Read this before
touching code.

## Ground rules

- **uv only** — never `pip`, `poetry`, or `venv` by hand. Deps: `uv add`,
  install: `uv sync`, run: `uv run --env-file .env -m wikibot`.
- No dependency outside `openai`, `httpx`, `wikipedia-api`, `gradio` without
  discussing it first.
- `.env` is gitignored and holds only `OPENAI_API_KEY`, `OPENAI_MODEL`,
  `OPENAI_BASE_URL`. Blank string means **absent** — treat `""` as unset
  (`.strip()` before use), never pass an empty value downstream.
- One focused commit per change. Conventional format:
  `feat|fix|docs|refactor|chore(scope): description`.

## Architecture

Three modules, one direction of dependency; two entrypoints on top:

```
cli.py   ─┐
          ├→  agent.py  →  wiki.py  →  wikipediaapi  →  en.wikipedia.org
webui.py ─┘
(REPL)        (run_turn)                 (adapter)
```

Data flow per turn: user input → `run_turn()` generator → Responses API
stream → yields `("tool", name, args)` before dispatch, `("tool_result", name,
digest)` after, then `("delta", text)` / `("done", text)` events; tool calls
dispatch to `wiki.py` and their `function_call_output` is appended to history
before the next round. Both entrypoints consume the events directly; the CLI
prints tool lines and deltas as they arrive and ignores `tool_result`.

## Key files

| File | Purpose |
|------|---------|
| `wikibot/wiki.py` | `WikipediaClient` — the only place that talks to Wikipedia: `wikipediaapi`, plus a direct `httpx` query (`_intro`) for summaries and disambiguation |
| `wikibot/agent.py` | `Agent`, tool schemas, `_TOOL_CALLS` dispatch, `run_turn`, system prompt |
| `wikibot/cli.py` | `Repl` |
| `wikibot/webui.py` | `WikiBotApp` — Gradio app: a list of chats (display + agent history each) in `gr.BrowserState` |
| `wikibot/config.py` | `Config` — the only place `.env` is read |
| `wikibot/__main__.py` | `python -m wikibot` entry |
| `tests/` | Offline `unittest` suite: no network, no API key |

Classes hold a session's moving parts: `WikipediaClient` wraps the wikipedia
connection, `Agent(client, model)` wraps the OpenAI client, `Repl` and
`WikiBotApp` each own one conversation history.

## Conventions

- **Never call `wikipediaapi` outside `wiki.py`.** `agent.py` sees only the
  adapter's dict-returning methods. New Wikipedia capability = one method on
  `WikipediaClient` + one tool schema in `agent.py`.
- **Summaries skip `wikipediaapi`.** `WikipediaClient._intro` asks the
  MediaWiki API directly, because the library downloads the whole article for
  a summary and cannot flag disambiguation pages. Everything else goes
  through the library; keep any other direct query in `wiki.py` too.
- **Shape output before the model sees it.** Strip HTML from search snippets,
  JSON-encode tool results, truncate `page_content` / `page_section` at `MAX_CONTENT_CHARS`
  **with the `truncated` flag set** — never trim silently. A truncated result
  also carries a `sections` outline so the model can reach the rest through
  `get_page_section`.
- **Tool failures return strings, never raise.** `_dispatch` catches all
  exceptions and returns `{"error": ...}` as the tool output so the model can
  recover and the REPL turn never crashes. Same rule at the REPL boundary:
  `cli.py` catches, prints, keeps the session alive.
- **A turn always ends in an answer.** The last of `max_tool_rounds` requests
  is sent without tools and with an "out of lookups" user message that is not
  kept in history, so the model answers from what it gathered. A blank reply
  is retried once, then reported. `done` text has odd citation markers
  (`【Title】`) rewritten as links, using only URLs the tools returned.
- **`function_call` and its `function_call_output` are appended to history
  together**, in order. Splitting a pair makes the API reject the request.
  History lives with the entrypoint: a mutable list in `Repl`, one
  per chat in the `BrowserState` dict of `WikiBotApp`; `run_turn` mutates it in place.
- **The web UI never touches `wikipediaapi` and never builds URLs.** It talks
  only to `agent.py`'s event stream and renders what comes back; citations
  stay tool-output-only, same as the CLI. Errors become an error bubble, not
  a crash (same recovery rule as the CLI).
- **Citations come from tool output only** — the system prompt requires
  `[Title](URL)` with the URL the tools returned; don't add URLs anywhere
  else.
- Tool schemas use `"strict": True`: every property in `required`,
  `additionalProperties: false`. Optional params would need a nullable type
  union, not key omission.
- System prompt carries cross-cutting rules only; "what a tool does and when"
  lives in the tool's own `description`.

## Adding a tool

1. Method on `WikipediaClient` in `wiki.py` returning a plain dict (no HTML;
   a missing page or section is data, network errors may raise).
2. Schema in `agent.py` `TOOLS` (strict mode shape).
3. One row in `agent.py` `_TOOL_CALLS` mapping the tool to its wiki method.
4. Live-test the call path before committing.

## Verification

- `uv run python -m unittest` must pass. It runs offline against stand-ins
  for the OpenAI stream and `wikipediaapi`, so it checks the rules above
  (paired calls, failures as data, the `truncated` flag, the env-var message)
  but not the live services; the checks below still apply to changes that
  touch them. A new tool or event gets a test alongside it.
- `uv run --env-file .env -m wikibot` must start (or fail naming the exact
  missing env var — that message is the contract, don't break it; test with
  `uv run --env-file /dev/null -m wikibot`).
- Web UI: `uv run --env-file .env python -m wikibot.webui` → serves
  http://localhost:7860; one turn must show a lookup-steps line then a streamed
  cited answer; New chat must start with both histories empty, Delete
  this chat must remove both, and Try again must replace the last answer.
- Engine check without a browser:
  `list(Agent(OpenAI(api_key=cfg.api_key, base_url=cfg.base_url), cfg.model).run_turn([], "msg"))`
  → event order `tool … → delta … → done`, every `function_call` paired with
  its `function_call_output`.
- Live testing needs a filled `.env`; unit-style checks of `wiki.py` need no
  key: `uv run python -c "from wikibot.wiki import WikipediaClient; print(WikipediaClient().search_wikipedia('Alan Turing'))"`.
- Wikipedia API is keyless; OpenAI errors are the only auth failures to expect.
