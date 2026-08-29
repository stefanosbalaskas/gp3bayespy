#!/usr/bin/env python3
# Stdlib-only documentation audit for built and deployed gp3bayespy docs.

from __future__ import annotations

import argparse
import sys
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

SITE_PREFIX = "/gp3bayespy/"
EXPECTED_SITE_URL = "https://stefanosbalaskas.github.io/gp3bayespy/"
EXPECTED_SOCIAL = EXPECTED_SITE_URL + "assets/social-card.png"


class AuditFailure(RuntimeError):
    pass


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links = []
        self.images = []
        self.ids = set()
        self.meta_name = {}
        self.meta_property = {}
        self.canonical = None
        self.html_lang = None
        self._in_title = False
        self.title_parts = []

    @property
    def title(self) -> str:
        return "".join(self.title_parts).strip()

    def handle_starttag(self, tag, attrs_list):
        attrs = dict(attrs_list)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "html":
            self.html_lang = attrs.get("lang")
        if tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"])
        if tag == "img" and attrs.get("src"):
            self.images.append((attrs["src"], attrs.get("alt")))
        if tag == "meta":
            content = attrs.get("content", "")
            if attrs.get("name"):
                self.meta_name[attrs["name"].lower()] = content
            if attrs.get("property"):
                self.meta_property[attrs["property"].lower()] = content
        if tag == "link" and "canonical" in attrs.get("rel", "").split():
            self.canonical = attrs.get("href")
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title_parts.append(data)


def parse_html(text: str) -> PageParser:
    parser = PageParser()
    parser.feed(text)
    return parser


def strip_site_prefix(path: str) -> str:
    if path.startswith(SITE_PREFIX):
        return path[len(SITE_PREFIX):]
    return path.lstrip("/")


def resolve_target(site: Path, source: Path, href: str):
    parsed = urllib.parse.urlsplit(href)
    fragment = urllib.parse.unquote(parsed.fragment)

    if parsed.scheme in {"mailto", "tel", "javascript", "data"}:
        return None, fragment

    if parsed.scheme in {"http", "https"}:
        if parsed.netloc != "stefanosbalaskas.github.io":
            return None, fragment
        target = site / urllib.parse.unquote(strip_site_prefix(parsed.path))
    elif href.startswith("/"):
        target = site / urllib.parse.unquote(strip_site_prefix(parsed.path))
    else:
        rel = urllib.parse.unquote(parsed.path)
        target = source.parent / rel if rel else source

    if target.is_dir():
        target = target / "index.html"
    elif not target.suffix:
        if (target / "index.html").exists():
            target = target / "index.html"
        elif target.with_suffix(".html").exists():
            target = target.with_suffix(".html")

    return target.resolve(), fragment


def audit_local(site_dir: Path) -> None:
    site = site_dir.resolve()
    if not site.is_dir():
        raise AuditFailure(f"Site directory not found: {site}")

    pages = sorted(
        page
        for page in site.rglob("*.html")
        if "overrides" not in page.relative_to(site).parts
    )
    if not pages:
        raise AuditFailure("No HTML pages generated.")

    parsed_pages = {}
    failures = []

    for page in pages:
        rel = page.relative_to(site).as_posix()
        parsed = parse_html(page.read_text(encoding="utf-8", errors="replace"))
        parsed_pages[page.resolve()] = parsed

        if parsed.html_lang != "en":
            failures.append(f"{rel}: html lang is not en")
        if not parsed.title:
            failures.append(f"{rel}: missing title")

        if rel != "404.html":
            if not parsed.meta_name.get("description", "").strip():
                failures.append(f"{rel}: missing meta description")
            if not parsed.canonical or not parsed.canonical.startswith(EXPECTED_SITE_URL):
                failures.append(f"{rel}: missing/invalid canonical")
            for key in ("og:title", "og:description", "og:url", "og:image"):
                if not parsed.meta_property.get(key, "").strip():
                    failures.append(f"{rel}: missing {key}")
            for key in ("twitter:card", "twitter:title", "twitter:description", "twitter:image"):
                if not parsed.meta_name.get(key, "").strip():
                    failures.append(f"{rel}: missing {key}")
            if parsed.meta_property.get("og:image") != EXPECTED_SOCIAL:
                failures.append(f"{rel}: wrong og:image")
            if parsed.meta_name.get("twitter:image") != EXPECTED_SOCIAL:
                failures.append(f"{rel}: wrong twitter:image")

        for src, alt in parsed.images:
            target, _ = resolve_target(site, page, src)
            if target is not None and not target.exists():
                failures.append(f"{rel}: missing image {src}")
            if alt is None:
                failures.append(f"{rel}: image missing alt attribute: {src}")

    for page, parsed in list(parsed_pages.items()):
        rel = page.relative_to(site).as_posix()
        for href in parsed.links:
            target, fragment = resolve_target(site, page, href)
            if target is None:
                continue
            try:
                target.relative_to(site)
            except ValueError:
                failures.append(f"{rel}: link escapes site root: {href}")
                continue
            if not target.exists():
                failures.append(f"{rel}: broken internal link: {href}")
                continue
            if fragment and target.suffix.lower() == ".html":
                other = parsed_pages.get(target)
                if other is None:
                    other = parse_html(target.read_text(encoding="utf-8", errors="replace"))
                    parsed_pages[target] = other
                if fragment not in other.ids:
                    failures.append(f"{rel}: missing fragment #{fragment}: {href}")

    required = {
        "index.html",
        "getting-started/index.html",
        "articles/index.html",
        "reference/index.html",
        "plot-gallery/index.html",
        "citation/index.html",
        "release/index.html",
        "robots.txt",
        "sitemap.xml",
        "assets/favicon.svg",
        "assets/social-card.png",
        "assets/gallery/binary-roc.png",
        "assets/gallery/binary-pr.png",
        "assets/gallery/pupil-simulation.png",
    }
    for rel in sorted(required):
        if not (site / rel).exists():
            failures.append(f"missing required output: {rel}")

    robots_path = site / "robots.txt"
    if robots_path.exists():
        robots = robots_path.read_text(encoding="utf-8")
        if "Sitemap: " + EXPECTED_SITE_URL + "sitemap.xml" not in robots:
            failures.append("robots.txt has no canonical Sitemap entry")

    sitemap_path = site / "sitemap.xml"
    if sitemap_path.exists():
        sitemap = sitemap_path.read_text(encoding="utf-8", errors="replace")
        if EXPECTED_SITE_URL not in sitemap:
            failures.append("sitemap.xml has no canonical site URLs")

    css_path = site / "stylesheets" / "extra.css"
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8", errors="replace")
        for breakpoint in ("@media (max-width: 700px)", "@media (max-width: 1000px)"):
            if breakpoint not in css:
                failures.append(f"responsive CSS regression: {breakpoint} missing")
    else:
        failures.append("custom stylesheet missing")

    home_path = site / "index.html"
    if home_path.exists():
        home = home_path.read_text(encoding="utf-8", errors="replace")
        for marker in (
            "Bayesian workflows that keep every decision visible",
            "Choose your path",
            "One workflow, explicit gates",
            "10.5281/zenodo.22150746",
            "SoftwareSourceCode",
        ):
            if marker not in home:
                failures.append(f"homepage marker missing: {marker}")

    if failures:
        raise AuditFailure(
            f"{len(failures)} issue(s):\n- " + "\n- ".join(failures[:100])
        )

    print(
        f"Site audit PASS: {len(pages)} HTML pages; "
        "SEO/social metadata, links, fragments, assets, sitemap, robots, "
        "alt text, and responsive CSS checks passed."
    )


def fetch_text(url: str):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "gp3bayespy-site-audit/1",
            "Cache-Control": "no-cache, no-store, max-age=0",
            "Pragma": "no-cache",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.status, response.read().decode("utf-8", errors="replace")


def audit_live(base_url: str) -> None:
    base = base_url.rstrip("/") + "/"
    required = {
        "": ("Bayesian workflows that keep every decision visible", "10.5281/zenodo.22150746"),
        "getting-started/": ("Getting started",),
        "articles/": ("Article library",),
        "reference/": ("API reference", "458"),
        "plot-gallery/": ("Plot gallery", "pupil-simulation.png"),
        "citation/": ("Citing gp3bayespy", "10.5281/zenodo.22150746"),
        "release/": ("gp3bayespy 0.5.0", "10.5281/zenodo.22150746"),
        "robots.txt": ("Sitemap:",),
        "sitemap.xml": (base,),
    }

    deadline = time.time() + 300
    last = {}
    while time.time() < deadline:
        all_ok = True
        last = {}
        stamp = str(int(time.time()))
        for rel, markers in required.items():
            url = urllib.parse.urljoin(base, rel)
            url += ("&" if "?" in url else "?") + "audit=" + stamp
            try:
                status, body = fetch_text(url)
                missing = [m for m in markers if m not in body]
                if status != 200:
                    missing.append(f"HTTP {status}")
            except Exception as exc:
                missing = [f"HTTP error: {exc}"]
            last[rel or "/"] = missing
            if missing:
                all_ok = False
        if all_ok:
            print("Live site audit PASS:", base)
            return
        print("Live site not fully converged:", last)
        time.sleep(10)
    raise AuditFailure(f"Live audit did not converge: {last}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("site_dir", nargs="?", type=Path)
    parser.add_argument("--live", metavar="URL")
    args = parser.parse_args()
    if bool(args.site_dir) == bool(args.live):
        parser.error("provide exactly one of SITE_DIR or --live URL")
    if args.live:
        audit_live(args.live)
    else:
        audit_local(args.site_dir)


if __name__ == "__main__":
    try:
        main()
    except AuditFailure as exc:
        print("SITE AUDIT FAILED:", exc, file=sys.stderr)
        raise SystemExit(1) from None
