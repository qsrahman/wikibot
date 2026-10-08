"""OpenAI Responses API agent loop with Wikipedia tools."""

import json
from collections.abc import Iterator

from openai import OpenAI

from . import wiki

MAX_TOOL_ROUNDS = 10

SYSTEM_PROMPT = """\
You are wikibot, a chatbot that answers questions using English Wikipedia.

Rules:
- Answer from the Wikipedia tool results, not from memory. Search first when
  you are not sure of the exact article title.
- Cite factual claims with the source article as [Title](URL), using the URL
  the tools returned.
- If Wikipedia has nothing relevant, say so plainly instead of guessing.
- Answer in the language the user writes in. Keep answers concise.
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
            "exact title. Long output; use only when the summary is not enough."
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
]


def _dispatch(name: str, arguments: dict) -> str:
    try:
        if name == "search_wikipedia":
            result = wiki.search_wikipedia(arguments["query"], arguments["limit"])
        elif name == "get_page_summary":
            result = wiki.page_summary(arguments["title"])
        elif name == "get_page_content":
            result = wiki.page_content(arguments["title"])
        else:
            result = {"error": f"unknown tool: {name}"}
    except Exception as exc:
        result = {"error": f"{type(exc).__name__}: {exc}"}
    return json.dumps(result, ensure_ascii=False)


def _to_input_item(item) -> dict:
    data = item.model_dump(mode="json", by_alias=True, exclude_none=True)
    if data.get("type") == "message":
        for part in data.get("content", []):
            if part.get("type") == "output_text":
                part["type"] = "input_text"
                part.pop("annotations", None)
    return data


def stream_turn(
    client: OpenAI,
    model: str,
    history: list,
    user_message: str,
) -> Iterator[tuple]:
    """Run one user message to completion, yielding events as they happen.

    Events: ("tool", name, args) before each dispatch, ("delta", text) for
    streamed answer tokens, ("done", full_text) exactly once at the end.
    Raises on API errors; tool failures are data, never raised.

    History is mutated in place: user message, output items, and
    function_call/function_call_output pairs are appended together so the
    pairs are never split. Raise points occur only at API-call boundaries,
    so a call can never be left without its output.
    """
    history.append({"role": "user", "content": user_message})
    final_text = ""
    for _ in range(MAX_TOOL_ROUNDS):
        with client.responses.stream(
            model=model,
            instructions=SYSTEM_PROMPT,
            input=history,
            tools=TOOLS,
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    yield ("delta", event.delta)
            response = stream.get_final_response()

        output_items = list(response.output)
        history.extend(_to_input_item(item) for item in output_items)

        calls = [item for item in output_items if item.type == "function_call"]
        if not calls:
            final_text = response.output_text
            break

        for call in calls:
            arguments = json.loads(call.arguments)
            yield ("tool", call.name, arguments)
            history.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": _dispatch(call.name, arguments),
                }
            )
    else:
        final_text = "Stopped: tool call limit reached (10 rounds) for this message."
    yield ("done", final_text)


def run_turn(
    client: OpenAI,
    model: str,
    history: list,
    user_message: str,
    on_tool=None,
) -> str:
    """Blocking wrapper over stream_turn for the CLI: ignores deltas."""
    final_text = ""
    for event in stream_turn(client, model, history, user_message):
        if event[0] == "tool" and on_tool:
            on_tool(f"{event[1]}({json.dumps(event[2], ensure_ascii=False)})")
        elif event[0] == "done":
            final_text = event[1]
    return final_text
