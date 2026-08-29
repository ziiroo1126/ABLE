import unittest
from html.parser import HTMLParser
from pathlib import Path


class _ProjectPageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.links = []
        self.images = []
        self.assets = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.add(attributes["id"])
        if tag == "a" and "href" in attributes:
            self.links.append(attributes["href"])
        if tag == "img" and "src" in attributes:
            self.images.append(attributes["src"])
        if tag == "link" and "href" in attributes:
            self.assets.append(attributes["href"])
        if tag == "script" and "src" in attributes:
            self.assets.append(attributes["src"])


class ProjectPageTests(unittest.TestCase):
    def test_project_page_exposes_the_paper_story_and_resources(self):
        page_path = Path("docs/index.html")
        parser = _ProjectPageParser()
        parser.feed(page_path.read_text(encoding="utf-8"))

        self.assertTrue(
            {"overview", "method", "results", "quickstart", "citation"}
            <= parser.ids
        )
        self.assertIn("https://arxiv.org/abs/2606.07524", parser.links)
        self.assertIn("https://github.com/ziiroo1126/ABLE", parser.links)

    def test_local_assets_and_section_links_resolve(self):
        docs_dir = Path("docs")
        parser = _ProjectPageParser()
        parser.feed((docs_dir / "index.html").read_text(encoding="utf-8"))

        for asset in parser.images + parser.assets:
            if "://" not in asset:
                self.assertTrue((docs_dir / asset).is_file(), asset)

        for link in parser.links:
            if link.startswith("#"):
                self.assertIn(link.removeprefix("#"), parser.ids, link)

    def test_repository_documents_branch_based_pages_deployment(self):
        guide = Path("docs/README.md").read_text(encoding="utf-8")
        self.assertIn("Deploy from a branch", guide)
        self.assertIn("main", guide)
        self.assertIn("/docs", guide)
        self.assertTrue(Path("docs/.nojekyll").is_file())


if __name__ == "__main__":
    unittest.main()
