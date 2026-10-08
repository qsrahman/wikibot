"""OpenAI Responses API agent loop with Wikipedia tools."""

import json
from collections.abc import Iterator

from openai import OpenAI

from .wiki import WikipediaClient

MAX_TOOL_ROUNDS = 15

SYSTEM_PROMPT = """\
You are wikibot, an assistant that answers questions from English Wikipedia.
People use you because every fact you give can be traced to an article, so
you look things up instead of recalling them.

Grounding
- Every factual question starts with a tool call. Take facts from text the
  tools returned in this conversation, never from memory, and that includes
  what you believe about dates and recent or future events. Earlier results
  still count for follow-up questions; look up whatever they do not cover.
- Search snippets are fragments for choosing an article. State a fact from
  one only when the snippet says it outright; otherwise read the article.
- When a result is truncated and the answer is not in the part you received,
  fetch the relevant section rather than answering from partial text.
- If the search finds nothing relevant, or a tool returns an error you
  cannot work around, say so plainly instead of filling the gap.
- Report figures the way the article does, including ranges, disputes and
  "as of" dates.

Ambiguous terms
- A page that only lists meanings of a term ("X may refer to ...") is a
  disambiguation page, not an answer. If the conversation makes the intended
  meaning clear, search for that article and answer from it. Otherwise name
  the main options in plain text and ask which one the user means.

Citations
- Cite the article a fact came from as [Title](URL), copying the URL exactly
  as a tool returned it. Never write a URL from memory or build one from a
  title: mention such a page by name without a link, or search for it first.
- One citation per article per paragraph or list is enough; do not repeat the
  same link after every sentence.

Style
- Reply in the user's language. The tools only search English Wikipedia, so
  write search queries in English and leave article titles untranslated.
- Lead with the answer and keep it to a few sentences unless the user asks
  for depth. Use a list only when the content is a list, and bold sparingly.
- Only greetings and questions about yourself are answered without tools.
"""

TOOLS = [
    {
        "type": "function",
        "name": "search_wikipedia",
        "description": (
            "Full-text search of English Wikipedia. Returns matching article "
            "titles, short snippets, and URLs. Call this first to find the "
            "right article title."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms."},
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results, 1-10. Use 5 unless asked otherwise.",
                },
            },
            "required": ["query", "limit"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_page_summary",
        "description": (
            "Fetch the introductory summary of a Wikipedia article by its "
            "exact title, plus the article URL. Cheap; use for short answers."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Exact article title as returned by search_wikipedia.",
                }
            },
            "required": ["title"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_page_content",
        "description": (
            "Fetch the full plain-text content of a Wikipedia article by its "
            "exact title. Long output; use only when the summary is not enough. "
            "Long articles are cut off: the result then has truncated=true and "
            "a sections outline (name, depth, size) to use with get_page_section."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Exact article title as returned by search_wikipedia.",
                }
            },
            "required": ["title"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_page_section",
        "description": (
            "Fetch one section of a Wikipedia article, including its "
            "subsections, by exact section name. Use to read parts of a long "
            "article that get_page_content cut off."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Exact article title as returned by search_wikipedia.",
                },
                "section": {
                    "type": "string",
                    "description": "Exact section name from a sections outline.",
                },
            },
            "required": ["title", "section"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_page_links",
        "description": (
            "List the articles one page links to, as exact titles you can "
            "fetch with get_page_summary. Use to find related articles, for "
            "\"what does this page cover\", or when the user asks what a "
            "topic is connected to."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Exact article title as returned by search_wikipedia.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of links to return, 1-20. Use 10 unless asked otherwise.",
                },
            },
            "required": ["title", "limit"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_random_pages",
        "description": (
            "Fetch random English Wikipedia articles with their introductory "
            "summaries. Use when the user wants a surprise, something new to "
            "read, or \"tell me something interesting\"."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of random articles, 1-5. Use 3 unless asked otherwise.",
                },
            },
            "required": ["limit"],
            "additionalProperties": False,
        },
    },
]


class Agent:
    """Runs the Responses API loop and dispatches Wikipedia tool calls.

    Holds the OpenAI client, model name, and the Wikipedia adapter. One
    instance per conversation session; ``run_turn`` mutates a caller-owned
    ``history`` list in place and yields events.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
        wiki: WikipediaClient | None = None,
        *,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
    ):
        self.client = client
        self.model = model
        self.wiki = wiki if wiki is not None else WikipediaClient()
        self.max_tool_rounds = max_tool_rounds

    def _dispatch(self, name: str, arguments: dict) -> str:
        try:
            if name == "search_wikipedia":
                result = self.wiki.search_wikipedia(arguments["query"], arguments["limit"])
            elif name == "get_page_summary":
                result = self.wiki.page_summary(arguments["title"])
            elif name == "get_page_content":
                result = self.wiki.page_content(arguments["title"])
            elif name == "get_page_section":
                result = self.wiki.page_section(arguments["title"], arguments["section"])
            elif name == "get_page_links":
                result = self.wiki.page_links(arguments["title"], arguments["limit"])
            elif name == "get_random_pages":
                result = self.wiki.random_pages(arguments["limit"])
            else:
                result = {"error": f"unknown tool: {name}"}
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
        return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def _to_input_item(item) -> dict:
        data = item.model_dump(mode="json", by_alias=True, exclude_none=True)
        if data.get("type") == "message":
            for part in data.get("content", []):
                if part.get("type") == "output_text":
                    part["type"] = "input_text"
                    part.pop("annotations", None)
        return data

    @staticmethod
    def _digest(name: str, output: str) -> str:
        """One-line tool result summary for UIs; not sent to the model."""
        try:
            data = json.loads(output)
        except ValueError:
            return ""
        if "error" in data:
            return f"error: {str(data['error'])[:100]}"
        if not data.get("found", True):
            return "section not found" if "section" in data else "page not found"
        if name == "search_wikipedia":
            pages = data.get("pages", [])
            if not pages:
                return "no results"
            titles = ", ".join(p["title"] for p in pages[:3])
            extra = f" (+{len(pages) - 3} more)" if len(pages) > 3 else ""
            return f"{len(pages)} results: {titles}{extra}"
        if name == "get_page_links":
            links = data.get("links", [])
            titles = ", ".join(p["title"] for p in links[:3])
            extra = f" (+{len(links) - 3} more)" if len(links) > 3 else ""
            return f"{len(links)} links: {titles}{extra}"
        if name == "get_random_pages":
            pages = data.get("pages", [])
            titles = ", ".join(p["title"] for p in pages[:3])
            return f"{len(pages)} random pages: {titles}"
        label = "content" if "text" in data else "summary"
        chars = len(data.get("text") or data.get("summary") or "")
        note = ", truncated" if data.get("truncated") else ""
        return f"{label}: {chars:,} chars{note}"

    def run_turn(
        self,
        history: list[dict],
        user_message: str,
    ) -> Iterator[tuple]:
        """Run one user message to completion, yielding events as they happen.

        Events: ("tool", name, args) before each dispatch, ("tool_result",
        name, digest) after it, ("delta", text) for streamed answer tokens,
        ("done", full_text) exactly once at the end. Raises on API errors;
        tool failures are data, never raised.

        History is mutated in place: user message, output items, and
        function_call/function_call_output pairs are appended together so the
        pairs are never split. Raise points occur only at API-call boundaries,
        so a call can never be left without its output.
        """
        history.append({"role": "user", "content": user_message})
        final_text = ""
        for _ in range(self.max_tool_rounds):
            response = None
            with self.client.responses.create(
                model=self.model,
                instructions=SYSTEM_PROMPT,
                input=history,
                tools=TOOLS,
                stream=True,
                store=False,
            ) as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        yield ("delta", event.delta)
                    elif event.type == "response.completed":
                        response = event.response
            if response is None:
                raise RuntimeError("response stream ended without completing")

            output_items = list(response.output)
            history.extend(self._to_input_item(item) for item in output_items)

            calls = [item for item in output_items if item.type == "function_call"]
            if not calls:
                final_text = response.output_text
                break

            for call in calls:
                # Must not raise here: the function_call is already in history and
                # needs its output. Bad JSON becomes {} and _dispatch reports it.
                try:
                    arguments = json.loads(call.arguments)
                except ValueError:
                    arguments = {}
                yield ("tool", call.name, arguments)
                output = self._dispatch(call.name, arguments)
                history.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": output,
                    }
                )
                yield ("tool_result", call.name, self._digest(call.name, output))
        else:
            final_text = (
                "Stopped: tool call limit reached"
                f" ({self.max_tool_rounds} rounds) for this message."
            )
        yield ("done", final_text)