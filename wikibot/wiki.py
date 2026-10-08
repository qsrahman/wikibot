"""Thin adapter over the wikipediaapi library.

Every function returns a plain dict shaped for the model: strings only,
HTML stripped, content truncated with an explicit marker. No exception
escapes this module — callers in agent.py treat failures as data.
"""

import html
import re
from urllib.parse import quote

import wikipediaapi

MAX_CONTENT_CHARS = 8000
USER_AGENT = "wikibot/0.1 (local CLI Wikipedia chatbot)"

_wiki = wikipediaapi.Wikipedia(user_agent=USER_AGENT, language="en")

_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    return html.unescape(_TAG_RE.sub("", text))


def _page_url(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))


def search_wikipedia(query: str, limit: int = 5) -> dict:
    results = _wiki.search(query, limit=limit)
    pages = [
        {
            "title": title,
            "snippet": _strip_html(page.search_meta.snippet) if page.search_meta else "",
            "url": _page_url(title),
        }
        for title, page in results.pages.items()
    ]
    out: dict = {"pages": pages}
    if results.suggestion:
        out["suggestion"] = results.suggestion
    if not pages:
        out["note"] = "No results found."
    return out


def page_summary(title: str) -> dict:
    page = _wiki.page(title)
    if not page.exists():
        return {"found": False, "title": title, "hint": "Wrong title? Call search_wikipedia."}
    return {"found": True, "title": page.title, "summary": page.summary, "url": page.fullurl}


def page_content(title: str) -> dict:
    page = _wiki.page(title)
    if not page.exists():
        return {"found": False, "title": title, "hint": "Wrong title? Call search_wikipedia."}
    text = page.text
    truncated = len(text) > MAX_CONTENT_CHARS
    if truncated:
        text = text[:MAX_CONTENT_CHARS]
    return {
        "found": True,
        "title": page.title,
        "text": text,
        "truncated": truncated,
        "url": page.fullurl,
    }
