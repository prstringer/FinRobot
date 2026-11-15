#!/usr/bin/env python
# coding: utf-8
"""
PDF renderer: the same HTML as the on-screen report, in print mode, printed by headless Chromium.

One layout codebase: ``render_html.render_html_string(model, print_mode=True)`` carries the print CSS (A4 portrait, one panel
per page, running header and footer, charts as SVG). This module only drives the browser, enforces the page rules and
stamps the PDF metadata. It does no calculation.

Build rules (the build FAILS, it does not warn):
* no panel may be taller than one page's content area (checked in the browser before printing, and again on the PDF);
* the file must have the expected number of pages (cover summary + six panels + end matter);
* every headline, chip value and so-what sentence must be extractable as text when ``pdftotext`` is available.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import tempfile
from typing import Optional

from . import render_html

A4_W_MM, A4_H_MM = 210.0, 297.0
MAX_BYTES = 3 * 1024 * 1024
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
]


class PdfBuildError(RuntimeError):
    pass


def _find_chrome() -> Optional[str]:
    for c in CHROME_CANDIDATES:
        if os.path.exists(c):
            return c
    return shutil.which("google-chrome") or shutil.which("chromium")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace(" ", " ")).strip()


def expected_texts(model: dict) -> list[str]:
    """Every headline, chip value and so-what sentence that must survive text extraction."""
    out = [model["thesis"]]
    for p in model["panels"]:
        out.append(p["headline"])
        for c in p["chips"]:
            out += [c["so_what"], c["value_text"]]
        for t in (p.get("extra") or {}).get("tests", []):
            out.append(t["state"])
    return out


def pdf_text(path: str) -> Optional[list[str]]:
    """Per-page text via pdftotext (poppler), or None when it is not installed."""
    exe = shutil.which("pdftotext") or ("/opt/homebrew/bin/pdftotext" if os.path.exists("/opt/homebrew/bin/pdftotext") else None)
    if not exe:
        return None
    raw = subprocess.run([exe, path, "-"], capture_output=True, text=True, check=True).stdout   # reading order keeps side-by-side chips apart
    return [_norm(p) for p in raw.split("\f") if p.strip()]


def check_pdf(model: dict, path: str, expected_pages: Optional[int] = None) -> dict:
    """Post-build checks. Raises PdfBuildError on any failure; returns a small report otherwise."""
    size = os.path.getsize(path)
    problems = []
    if size > MAX_BYTES:
        problems.append(f"PDF is {size / 1e6:.1f} MB; the limit is {MAX_BYTES / 1e6:.0f} MB")
    pages = pdf_text(path)
    missing: list[str] = []
    split: list[str] = []
    if pages is not None:
        whole = " ".join(pages)
        # ignore punctuation/hyphenation differences introduced by line wrapping
        flat = re.sub(r"[\s\-]+", "", whole)
        for t in expected_texts(model):
            if re.sub(r"[\s\-]+", "", _norm(t)) not in flat:
                missing.append(t)
        # a panel must not split: its headline and its last chip's so-what must sit on the same page
        for p in model["panels"]:
            if not p["chips"]:
                continue
            head, last = re.sub(r"[\s\-]+", "", p["headline"]), re.sub(r"[\s\-]+", "", _norm(p["chips"][-1]["so_what"]))
            pg_head = [i for i, t in enumerate(pages) if head in re.sub(r"[\s\-]+", "", t)]
            pg_last = [i for i, t in enumerate(pages) if last in re.sub(r"[\s\-]+", "", t)]
            if pg_head and pg_last and pg_head[0] != pg_last[0]:
                split.append(p["id"])
        if expected_pages and len(pages) != expected_pages:
            problems.append(f"PDF has {len(pages)} pages; expected {expected_pages}")
    if missing:
        problems.append(f"{len(missing)} headline/chip/so-what texts are not extractable, e.g. {missing[0]!r}")
    if split:
        problems.append(f"Panels split across pages: {', '.join(split)}")
    if problems:
        raise PdfBuildError("; ".join(problems))
    return {"bytes": size, "pages": len(pages) if pages is not None else None, "text_checked": pages is not None}


def _set_metadata(path: str, model: dict) -> None:
    """Title, author, subject and creation date. Uses pypdf when installed; Chromium already sets the title."""
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        return
    r = PdfReader(path)
    w = PdfWriter()
    for pg in r.pages:
        w.add_page(pg)
    m = model["meta"]
    stamp = _dt.datetime.now().strftime("D:%Y%m%d%H%M%S")
    w.add_metadata({"/Title": f"{m['company_name']} ({m['ticker']}) equity research, {m['report_date']}",
                    "/Author": "AI4Finance Foundation FinRobot Equity Research",
                    "/Subject": f"Visual equity research report, snapshot {m['snapshot'].get('price_date') or m['report_date']}",
                    "/CreationDate": stamp, "/Creator": "FinRobot visual report"})
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        w.write(fh)
    os.replace(tmp, path)


def _render_with_chrome_cli(html_path: str, out_path: str) -> None:
    """Fallback without Playwright: Chrome's own print command. The page mirrors its ready flag and any panel overflow
    into <html data-report-ready data-print-overflow>, which a ``--dump-dom`` run reads before the PDF is printed."""
    chrome = _find_chrome()
    if not chrome:
        raise PdfBuildError("No Chrome or Chromium found, and Playwright is not installed.")
    url = "file://" + os.path.abspath(html_path)
    profile = tempfile.mkdtemp(prefix="visual_chrome_")
    base = [chrome, "--headless=new", "--no-sandbox", "--disable-gpu", f"--user-data-dir={profile}", "--virtual-time-budget=15000"]
    try:
        dom = subprocess.run(base + ["--dump-dom", url], capture_output=True, text=True, timeout=120).stdout
        if 'data-report-ready="true"' not in dom:
            raise PdfBuildError("Charts did not finish rendering before printing.")
        m = re.search(r'data-print-overflow="([^"]*)"', dom)
        over = json.loads(m.group(1).replace("&quot;", '"')) if m else []
        if over:
            raise PdfBuildError("Panel overflows its page: " + ", ".join(f"{o['panel']} ({o['height_px']}px > {o['limit_px']}px)" for o in over))
        subprocess.run(base + ["--no-pdf-header-footer", f"--print-to-pdf={out_path}", url], capture_output=True, timeout=180, check=True)
    finally:
        shutil.rmtree(profile, ignore_errors=True)
    if not os.path.exists(out_path):
        raise PdfBuildError("Chrome did not write the PDF.")


def render_pdf(model: dict, out_path: str, keep_html: Optional[str] = None, check: bool = True) -> str:
    """Render ``model`` to an A4 PDF at ``out_path`` (raises PdfBuildError if a page rule fails).

    Uses Playwright when installed, else Chrome's command line (same print CSS, same overflow rule)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sync_playwright = None
    html = render_html.render_html_string(model, print_mode=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="visual_pdf_")
    html_path = keep_html or os.path.join(tmpdir, "report_print.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    chrome = _find_chrome()
    try:
        if sync_playwright is None:
            _render_with_chrome_cli(html_path, out_path)
            _set_metadata(out_path, model)
            if check:
                check_pdf(model, out_path)
            return out_path
        with sync_playwright() as pw:
            launch = {"headless": True}
            browser = (pw.chromium.launch(channel="chrome", **launch) if chrome and "Google Chrome" in chrome
                       else pw.chromium.launch(executable_path=chrome, **launch) if chrome else pw.chromium.launch(**launch))
            page = browser.new_page(viewport={"width": 794, "height": 1123})
            page.emulate_media(media="print")
            page.goto("file://" + os.path.abspath(html_path))
            page.wait_for_function("window.__reportReady === true", timeout=30000)
            overflow = [f"{o['panel']} ({o['height_px']}px > {o['limit_px']}px)" for o in page.evaluate("window.__printOverflow || []")]
            if overflow:
                browser.close()
                raise PdfBuildError("Panel overflows its page: " + ", ".join(overflow))
            page.pdf(path=out_path, format="A4", print_background=True, prefer_css_page_size=True,
                     margin={"top": "0", "right": "0", "bottom": "0", "left": "0"})
            browser.close()
    finally:
        if not keep_html:
            shutil.rmtree(tmpdir, ignore_errors=True)
    _set_metadata(out_path, model)
    if check:
        check_pdf(model, out_path)
    return out_path
