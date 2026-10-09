"""Thin adapter over the wikipediaapi library.

WikipediaClient returns plain dicts shaped for the model: strings only,
HTML stripped, content truncated with an explicit marker. A page or section
that does not exist is returned as data. Network and API errors are raised;
Agent._dispatch turns those into data.
"""

import html
import re
from urllib.parse import quote

import httpx
import wikipediaapi

MAX_CONTENT_CHARS = 8000
# The library defaults (10s, 3 retries with backoff) can hold one tool call
# for about 45s during an outage; these bound it near 15s.
TIMEOUT = 7.0
MAX_RETRIES = 1
USER_AGENT = "wikibot/0.1 (local CLI Wikipedia chatbot)"

_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    return html.unescape(_TAG_RE.sub("", text))


def _page_url(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))


def _clip(text: str) -> tuple[str, bool]:
    truncated = len(text) > MAX_CONTENT_CHARS
    return (text[:MAX_CONTENT_CHARS] if truncated else text), truncated


def _outline(sections, depth: int = 1) -> list[dict]:
    out = []
    for sec in sections:
        out.append(
            {"section": sec.title, "depth": depth, "chars": len(sec.full_text())}
        )
        out.extend(_outline(sec.sections, depth + 1))
    return out


def _missing_page(title: str) -> dict:
    return {
        "found": False,
        "title": title,
        "hint": "Wrong title? Call search_wikipedia.",
    }


class WikipediaClient:
    def __init__(self, language: str = "en", user_agent: str = USER_AGENT):
        self._wiki = wikipediaapi.Wikipedia(
            user_agent=user_agent,
            language=language,
            timeout=TIMEOUT,
            max_retries=MAX_RETRIES,
        )
        self._api_url = f"https://{language}.wikipedia.org/w/api.php"
        self._headers = {"User-Agent": user_agent}

    def _intro(self, title: str) -> dict:
        """One page's intro, canonical title, URL and disambiguation flag.

        A direct API query, not wikipediaapi: the library downloads the whole
        article to produce a summary and cannot report disambiguation pages.
        """
        response = httpx.get(
            self._api_url,
            headers=self._headers,
            timeout=TIMEOUT,
            params={
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "redirects": 1,
                "titles": title,
                "prop": "extracts|pageprops|info",
                "exintro": 1,
                "explaintext": 1,
                "ppprop": "disambiguation",
                "inprop": "url",
            },
        )
        response.raise_for_status()
        return response.json()["query"]["pages"][0]

    def search_wikipedia(self, query: str, limit: int = 5) -> dict:
        # enable_rewrites: a misspelled query returns the corrected query's
        # results instead of zero hits, saving the model a retry round.
        results = self._wiki.search(query, limit=limit, enable_rewrites=True)
        pages = [
            {
                "title": title,
                "snippet": _strip_html(page.search_meta.snippet)
                if page.search_meta
                else "",
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

    def page_summary(self, title: str) -> dict:
        page = self._intro(title)
        if page.get("missing") or page.get("invalid"):
            return _missing_page(title)
        out = {
            "found": True,
            "title": page["title"],
            "summary": page.get("extract", "").strip(),
            "url": page["fullurl"],
        }
        if "disambiguation" in page.get("pageprops", {}):
            out["disambiguation"] = True
            out["hint"] = (
                "This page lists meanings of the term; it is not an article."
                " Search for the specific one, or ask the user which they mean."
            )
        return out

    def page_content(self, title: str) -> dict:
        page = self._wiki.page(title)
        if not page.exists():
            return _missing_page(title)
        text, truncated = _clip(page.text)
        out = {
            "found": True,
            "title": page.title,
            "text": text,
            "truncated": truncated,
            "url": page.fullurl,
        }
        if truncated:
            out["sections"] = _outline(page.sections)
            out["hint"] = (
                "Text cut off. Call get_page_section for a section listed here."
            )
        return out

    def page_section(self, title: str, section: str) -> dict:
        page = self._wiki.page(title)
        if not page.exists():
            return _missing_page(title)
        # A heading can repeat within one article; return every match.
        matches = page.sections_by_title(section)
        if not matches:
            return {
                "found": False,
                "title": page.title,
                "section": section,
                "hint": "No such section. Use an exact name from sections.",
                "sections": _outline(page.sections),
            }
        text, truncated = _clip("\n\n".join(m.full_text().strip() for m in matches))
        out = {
            "found": True,
            "title": page.title,
            "section": section,
            "text": text,
            "truncated": truncated,
            "url": page.fullurl,
        }
        subsections = _outline(s for m in matches for s in m.sections)
        if truncated and subsections:
            out["sections"] = subsections
            out["hint"] = (
                "Text cut off. Call get_page_section for a subsection listed here."
            )
        return out

    def page_links(self, title: str, limit: int = 10) -> dict:
        page = self._wiki.page(title)
        if not page.exists():
            return _missing_page(title)
        links = list(page.links.items())[:limit]
        return {
            "found": True,
            "title": page.title,
            "links": [{"title": t, "url": _page_url(t)} for t, _ in links],
            "total": len(page.links),
            "url": page.fullurl,
        }

    def random_pages(self, limit: int = 3) -> dict:
        pages = []
        for title in self._wiki.random(limit=limit):
            page = self._intro(title)
            pages.append(
                {
                    "title": page["title"],
                    "summary": page.get("extract", "").strip(),
                    "url": page["fullurl"],
                }
            )
        return {"pages": pages}
