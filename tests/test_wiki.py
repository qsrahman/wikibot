import unittest
from types import SimpleNamespace

from wikibot.wiki import MAX_CONTENT_CHARS, WikipediaClient


class FakeSection:
    def __init__(self, title, text="", sections=()):
        self.title, self.text, self.sections = title, text, list(sections)

    def full_text(self):
        return "\n".join([self.title, self.text, *(s.full_text() for s in self.sections)])


class FakePage:
    def __init__(self, title, summary="", sections=(), links=(), exists=True, disambiguation=False):
        self.title, self.summary, self.sections = title, summary, list(sections)
        self.disambiguation = disambiguation
        self.links = {name: None for name in links}
        self.fullurl = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        self._exists = exists

    def exists(self):
        return self._exists

    @property
    def text(self):
        return "\n\n".join([self.summary, *(s.full_text() for s in self.sections)])

    def sections_by_title(self, title):
        found = []

        def walk(sections):
            for sec in sections:
                if sec.title == title:
                    found.append(sec)
                walk(sec.sections)

        walk(self.sections)
        return found


class FakeWikipedia:
    """Stands in for Wikipedia, both through wikipediaapi and through the
    direct intro query; unknown titles do not exist."""

    def __init__(self, *pages, search=None, redirects=None):
        self.pages = {page.title: page for page in pages}
        self.search_result = search
        self.redirects = redirects or {}

    def intro(self, title):
        page = self.page(self.redirects.get(title, title))
        if not page.exists():
            return {"title": title, "missing": True, "fullurl": page.fullurl}
        out = {"title": page.title, "extract": page.summary + "\n", "fullurl": page.fullurl}
        if page.disambiguation:
            out["pageprops"] = {"disambiguation": ""}
        return out

    def page(self, title):
        return self.pages.get(title, FakePage(title, exists=False))

    def search(self, query, limit, enable_rewrites):
        return self.search_result

    def random(self, limit):
        return dict(list(self.pages.items())[:limit])


def client(*pages, search=None, redirects=None) -> WikipediaClient:
    wiki = WikipediaClient()
    wiki._wiki = FakeWikipedia(*pages, search=search, redirects=redirects)
    wiki._intro = wiki._wiki.intro
    return wiki


LONG = "x" * (MAX_CONTENT_CHARS + 500)
WAR = FakePage(
    "War",
    summary="Intro.",
    sections=[
        FakeSection("Course", sections=[FakeSection("Europe", LONG), FakeSection("Pacific", "p")]),
        FakeSection("Aftermath", "short"),
    ],
)


class Search(unittest.TestCase):
    def test_strips_html_and_builds_urls(self):
        hit = SimpleNamespace(search_meta=SimpleNamespace(snippet='<span class="m">Ada</span> &amp; Co'))
        bare = SimpleNamespace(search_meta=None)
        results = SimpleNamespace(pages={"Ada Lovelace": hit, "A/B?": bare}, suggestion=None)
        out = client(search=results).search_wikipedia("ada", 5)
        self.assertEqual(
            out,
            {
                "pages": [
                    {"title": "Ada Lovelace", "snippet": "Ada & Co", "url": "https://en.wikipedia.org/wiki/Ada_Lovelace"},
                    {"title": "A/B?", "snippet": "", "url": "https://en.wikipedia.org/wiki/A/B%3F"},
                ]
            },
        )

    def test_no_results_carries_note_and_suggestion(self):
        results = SimpleNamespace(pages={}, suggestion="alan turing")
        out = client(search=results).search_wikipedia("alan turnig", 5)
        self.assertEqual(out, {"pages": [], "suggestion": "alan turing", "note": "No results found."})


class Pages(unittest.TestCase):
    def test_missing_page_is_data_for_every_page_tool(self):
        wiki = client()
        for result in (
            wiki.page_summary("Nope"),
            wiki.page_content("Nope"),
            wiki.page_section("Nope", "x"),
            wiki.page_links("Nope", 5),
        ):
            self.assertEqual(result["found"], False)
            self.assertEqual(result["title"], "Nope")
            self.assertIn("search_wikipedia", result["hint"])

    def test_summary(self):
        out = client(FakePage("Ada Lovelace", summary="A mathematician.")).page_summary("Ada Lovelace")
        self.assertEqual(
            out,
            {
                "found": True,
                "title": "Ada Lovelace",
                "summary": "A mathematician.",
                "url": "https://en.wikipedia.org/wiki/Ada_Lovelace",
            },
        )

    def test_summary_of_a_redirect_uses_the_target_title(self):
        uk = FakePage("United Kingdom", summary="A country.")
        out = client(uk, redirects={"UK": "United Kingdom"}).page_summary("UK")
        self.assertEqual(out["title"], "United Kingdom")
        self.assertEqual(out["url"], "https://en.wikipedia.org/wiki/United_Kingdom")

    def test_summary_flags_a_disambiguation_page(self):
        mercury = FakePage("Mercury", summary="Mercury most commonly refers to:", disambiguation=True)
        out = client(mercury).page_summary("Mercury")
        self.assertIs(out["disambiguation"], True)
        self.assertIn("not an article", out["hint"])
        self.assertNotIn("disambiguation", client(FakePage("Ada", summary="x")).page_summary("Ada"))

    def test_short_content_is_whole_and_unflagged(self):
        out = client(FakePage("Tiny", summary="All of it.")).page_content("Tiny")
        self.assertEqual(out["text"], "All of it.")
        self.assertIs(out["truncated"], False)
        self.assertNotIn("sections", out)

    def test_long_content_is_cut_flagged_and_outlined(self):
        out = client(WAR).page_content("War")
        self.assertEqual(len(out["text"]), MAX_CONTENT_CHARS)
        self.assertIs(out["truncated"], True)
        self.assertIn("get_page_section", out["hint"])
        self.assertEqual(
            [(s["section"], s["depth"]) for s in out["sections"]],
            [("Course", 1), ("Europe", 2), ("Pacific", 2), ("Aftermath", 1)],
        )
        self.assertEqual(out["sections"][3]["chars"], len("Aftermath\nshort"))

    def test_links_are_limited_but_total_is_not(self):
        out = client(FakePage("Hub", links=["A b", "C", "D"])).page_links("Hub", 2)
        self.assertEqual(
            out["links"],
            [
                {"title": "A b", "url": "https://en.wikipedia.org/wiki/A_b"},
                {"title": "C", "url": "https://en.wikipedia.org/wiki/C"},
            ],
        )
        self.assertEqual(out["total"], 3)

    def test_random_pages(self):
        out = client(FakePage("One", summary="1"), FakePage("Two", summary="2")).random_pages(1)
        self.assertEqual(
            out, {"pages": [{"title": "One", "summary": "1", "url": "https://en.wikipedia.org/wiki/One"}]}
        )


class Sections(unittest.TestCase):
    def test_section_includes_its_subsections(self):
        out = client(WAR).page_section("War", "Aftermath")
        self.assertEqual(out["text"], "Aftermath\nshort")
        self.assertIs(out["truncated"], False)
        self.assertNotIn("sections", out)

    def test_nested_section_is_found_by_name(self):
        self.assertEqual(client(WAR).page_section("War", "Pacific")["text"], "Pacific\np")

    def test_long_section_is_cut_and_lists_subsections(self):
        out = client(WAR).page_section("War", "Course")
        self.assertEqual(len(out["text"]), MAX_CONTENT_CHARS)
        self.assertIs(out["truncated"], True)
        self.assertEqual([s["section"] for s in out["sections"]], ["Europe", "Pacific"])

    def test_long_leaf_section_has_no_outline_to_offer(self):
        out = client(WAR).page_section("War", "Europe")
        self.assertIs(out["truncated"], True)
        self.assertNotIn("sections", out)
        self.assertNotIn("hint", out)

    def test_repeated_heading_returns_every_match(self):
        page = FakePage("Twice", sections=[FakeSection("Notes", "one"), FakeSection("Notes", "two")])
        self.assertEqual(client(page).page_section("Twice", "Notes")["text"], "Notes\none\n\nNotes\ntwo")

    def test_unknown_section_returns_the_outline(self):
        out = client(WAR).page_section("War", "Nope")
        self.assertIs(out["found"], False)
        self.assertEqual(out["section"], "Nope")
        self.assertEqual(len(out["sections"]), 4)
