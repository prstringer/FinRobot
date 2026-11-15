"""Tests for the visual report HTML renderer (report_visual.render_html).

Pure tests run everywhere. Browser tests drive headless Chrome directly (no Playwright): the harness appends a probe
script to a copy of the page, runs `chrome --headless=new --dump-dom`, and reads the probe result from the DOM.
They are skipped when no Chrome/Chromium binary is found. The axe-core test also needs axe.min.js (set AXE_JS, or
`npm install axe-core` somewhere listed in AXE_CANDIDATES) and is skipped without it.
"""
import copy
import html as htmllib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC)

from report_visual import render_html as rh  # noqa: E402

STUB_PATH = os.path.join(SRC, "report_visual", "sample", "stub.report.json")
SCRATCH = "/private/tmp/claude-501/-Users-ramjoshi-Coding-Projects-FinRobot/5b26141b-d31e-44a5-aefc-feaa847a9947/scratchpad"
AXE_CANDIDATES = [os.environ.get("AXE_JS", ""), os.path.join(SCRATCH, "npm", "node_modules", "axe-core", "axe.min.js"),
                  os.path.join(SRC, "..", "..", "..", "node_modules", "axe-core", "axe.min.js")]


def _find_chrome():
    for p in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/Applications/Chromium.app/Contents/MacOS/Chromium"):
        if os.path.exists(p):
            return p
    for n in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome"):
        w = shutil.which(n)
        if w:
            return w
    return None


CHROME = _find_chrome()
AXE = next((p for p in AXE_CANDIDATES if p and os.path.exists(p)), None)
needs_chrome = pytest.mark.skipif(CHROME is None, reason="no Chrome/Chromium binary found")
needs_axe = pytest.mark.skipif(CHROME is None or AXE is None, reason="Chrome or axe-core not available")


@pytest.fixture(scope="module")
def stub():
    with open(STUB_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def page(stub):
    return rh.render_html_string(stub)


@pytest.fixture(scope="module")
def print_page(stub):
    return rh.render_html_string(stub, print_mode=True)


def visible_text(page_html: str) -> str:
    """Text content with scripts and styles removed."""
    s = re.sub(r"<script.*?</script>", " ", page_html, flags=re.S)
    s = re.sub(r"<style.*?</style>", " ", s, flags=re.S)
    return htmllib.unescape(re.sub(r"<[^>]+>", " ", s))


def normalise(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------------------------- structure
def test_renders_stub_with_seven_panels_in_order(page, stub):
    ids = re.findall(r'<section class="panel(?: has-small)?" id="p-([a-z_]+)"', page)
    assert ids == [p["id"] for p in stub["panels"]] and len(ids) == 7
    assert page.lower().startswith("<!doctype html>")
    assert len(re.findall(r"<h1[ >]", page)) == 1
    assert 'id="nav"' in page and 'id="report"' in page and "Skip to the report" in page
    assert page.index("<h1") < page.index('class="panel__headline"')       # h1 precedes the first h2


def test_render_html_writes_one_file(stub, tmp_path):
    out = rh.render_html(stub, str(tmp_path / "r.html"))
    assert os.path.getsize(out) > 100_000
    assert "\r" not in open(out, encoding="utf-8", newline="").read()


def test_size_and_python_render_time(stub):
    t = time.time()
    s = rh.render_html_string(stub)
    assert time.time() - t < 1.5
    assert len(s.encode("utf-8")) < 2_000_000


def test_no_external_requests(page):
    markup = re.sub(r"<script[^>]*>.*?</script>", "", page, flags=re.S)
    markup = re.sub(r"<style>.*?</style>", "", markup, flags=re.S)
    assert not re.search(r"(src|href|action)\s*=\s*[\"']\s*(https?:)?//", markup)
    assert "<link" not in markup and "<iframe" not in markup and "<img" not in markup
    assert "<script src" not in page and "<script async" not in page
    # our own code and styles make no network calls (the vendored ECharts only carries inert namespace and licence strings)
    for name in ("report.js", "report.css"):
        src = open(os.path.join(rh.ASSET_DIR, name), encoding="utf-8").read()
        assert not re.search(r"\b(fetch|XMLHttpRequest|sendBeacon|WebSocket|importScripts)\b", src), name
        assert not re.search(r"@import|url\(\s*[\"']?https?:", src), name
        assert "http://" not in src and "https://" not in src, name


def test_every_headline_chip_value_and_so_what_is_in_the_html(page, stub):
    text = normalise(visible_text(page))
    for p in stub["panels"]:
        assert normalise(p["headline"]) in text, p["headline"]
        for c in p["chips"]:
            assert normalise(c["value_text"]) in text
            assert normalise(c["so_what"]) in text
            assert c["label"] in text and c["yardstick_text"] in text
    assert normalise(stub["thesis"]) in text
    for w in stub["warnings"]:
        assert normalise(w["text"]) in text
    for sp in stub["panels"][6]["extra"]["signposts"]:
        assert sp["threshold_text"] in text and sp["current_text"] in text
    for t in stub["panels"][5]["extra"]["tests"]:
        assert t["evidence"] in text


def test_verdict_chips_carry_icon_and_word_not_colour_alone(page):
    chips = re.findall(r'<article class="chip chip--(good|neutral|concern)".*?</header>', page, flags=re.S)
    assert chips
    for m in re.finditer(r'<article class="chip chip--(good|neutral|concern)".*?</header>', page, flags=re.S):
        block = m.group(0)
        assert "<svg" in block and re.search(r"<span>(Good|Neutral|Concern)</span>", block)


def test_general_norm_label_is_shown(page):
    assert "General norm, not company-specific" in page
    assert 'class="tag tag--general"' in page


def test_chart_containers_have_aria_and_values_tables(page, stub):
    n = sum(len(p["charts"]) for p in stub["panels"])
    assert len(re.findall(r'class="chart__canvas" id="c-[a-z_0-9]+-[a-z_0-9]+" role="img" aria-label="[^"]{20,}', page)) == n
    assert len(re.findall(r"<details class=\"values\"", page)) == n
    assert page.count('<th scope="col">') > n


def test_toggles_rendered_for_variants_only(page):
    assert page.count('data-toggle-for="c-valuation-valuation_strip"') == 1
    assert page.count('data-toggle-for="c-price-price_trend"') == 1
    assert 'aria-pressed="true"' in page


def test_glossary_links_first_use_only(page):
    body = page.split('<footer class="endmatter"')[0]
    for term in ("TTM", "SG&amp;A", "EV/EBITDA"):
        gid = {"TTM": "ttm", "SG&amp;A": "sga", "EV/EBITDA": "ev_ebitda"}[term]
        assert body.count(f'href="#gl-{gid}"') == 1, term
    assert 'id="gl-ttm"' in page and 'id="gl-ev_ebitda"' in page


def test_portfolio_block_hidden_when_empty_and_shown_when_set(stub):
    s = rh.render_html_string(stub)
    assert "Your portfolio" not in s
    m = copy.deepcopy(stub)
    m["panels"][6]["extra"]["portfolio"] = {"current_weight": "3%", "target_weight": "5%", "triggers": "Add below Rs 200"}
    s2 = rh.render_html_string(m)
    assert "Your portfolio" in s2 and "Add below Rs 200" in s2


def test_warnings_block_levels_and_ok_state(stub):
    m = copy.deepcopy(stub)
    m["warnings"] = [{"level": "error", "text": "Validation failed: P/E does not reconcile"}]
    s = rh.render_html_string(m)
    assert 'lvl lvl--error">Error' in s and "Validation failed: P/E does not reconcile" in s
    m["warnings"] = []
    assert "No validation flags" in rh.render_html_string(m)


def test_all_model_text_is_escaped_and_json_cannot_break_out(stub):
    m = copy.deepcopy(stub)
    evil = "<script>alert(1)</script><img src=x onerror=alert(2)>"
    m["panels"][0]["headline"] = evil
    m["panels"][1]["chips"][0]["so_what"] = evil
    m["thesis"] = evil
    m["glossary"][0]["definition"] = evil
    m["panels"][1]["charts"][0]["alt"] = '"><script>alert(3)</script>'
    s = rh.render_html_string(m)
    assert "<script>alert" not in s and "<img src=x" not in s
    assert "&lt;script&gt;alert(1)" in s
    # the embedded JSON must not contain a raw closing script tag
    blob = re.search(r'<script type="application/json" id="report-data">(.*?)</script>', s, flags=re.S).group(1)
    assert "<" not in blob and ">" not in blob and json.loads(blob)


def test_em_dash_never_emitted(stub):
    m = copy.deepcopy(stub)
    m["thesis"] = "Costs rose — margins fell."
    m["panels"][2]["charts"][0]["alt"] = "Chart — alt text"
    for pm in (False, True):
        s = rh.render_html_string(m, print_mode=pm)
        assert "—" not in s
    assert "Costs rose, margins fell." in rh.render_html_string(m)


def test_missing_required_fields_raise(stub):
    with pytest.raises(ValueError):
        rh.render_html_string({"panels": stub["panels"]})
    bad = copy.deepcopy(stub)
    del bad["panels"][0]["headline"]
    with pytest.raises(ValueError):
        rh.render_html_string(bad)


def test_inr_and_scale_helpers():
    assert rh.inr(23524) == "23,524"
    assert rh.inr(1234567.891, 2) == "12,34,567.89"
    assert rh.inr(9399.1) == "9,399"
    assert rh.inr(None) == "n/a" and rh.inr(-1500000) == "-15,00,000"
    assert rh.scale_pos({"min": 0, "max": 10}, 5) == 50.0
    assert rh.scale_pos({"min": 0, "max": 10}, 99) == 100.0
    assert rh.scale_pos({"min": 3, "max": 3}, 3) == 50.0
    assert rh.scale_pos({"min": None, "max": 3}, 3) is None


# --------------------------------------------------------------------------------------------- print mode
def test_print_mode_has_no_nav_or_toggles_and_opens_tables(print_page, stub):
    assert 'id="nav"' not in print_page and "theme-btn" not in print_page.split("<script")[0].split("<body")[1]
    assert 'class="seg"' not in print_page and "<body class=\"print\">" in print_page
    assert print_page.count("data-print-panel=") == 7
    assert 'id="page-probe"' in print_page and 'class="pagehead"' in print_page and 'class="pagefoot"' in print_page
    small = [d for d in re.findall(r'<details class="values"( open)? data-rows="(\d+)"', print_page)]
    assert small and all(o == " open" for o, n in small if int(n) <= rh.MAX_PRINT_ROWS)
    # the one-year price table has 12 rows: open; static notes for toggles are present
    assert "Shown: one year." in print_page and "At the report's price of about Rs 129" in print_page
    assert "<details open><summary>Contribution margin (FY26)" in print_page      # calc notes expanded in print


def test_print_mode_long_tables_stay_closed_with_note(stub):
    m = copy.deepcopy(stub)
    big = m["panels"][5]["charts"][0]
    big["datatable"]["rows"] = [[f"D{i}", str(100 + i)] for i in range(120)]
    s = rh.render_html_string(m, print_mode=True)
    assert '<details class="values" data-rows="120">' in s
    assert "120 data rows are not printed" in s


# --------------------------------------------------------------------------------------------- browser
def _run_chrome(html_text: str, probe_js: str, width=1280, height=900, budget=9000, extra=()):
    """Append a probe to a copy of the page, run headless Chrome, return the probe's JSON result."""
    wrapper = ("(function(){var t0=Date.now();function go(){if(!window.__reportReady&&Date.now()-t0<8000){return setTimeout(go,50);}"
               "Promise.resolve().then(function(){return (" + probe_js + ")();}).then(function(r){document.body.setAttribute('data-probe',JSON.stringify(r));})"
               ".catch(function(e){document.body.setAttribute('data-probe',JSON.stringify({error:String(e)}));});}go();})();")
    i = html_text.rfind("</body>")
    doc = html_text[:i] + "<script>" + wrapper + "</script>" + html_text[i:]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "page.html")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(doc)
        cmd = [CHROME, "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", "--allow-file-access-from-files",
               f"--user-data-dir={os.path.join(d, 'profile')}", f"--window-size={width},{height}", f"--virtual-time-budget={budget}",
               "--dump-dom", *extra, f"file://{path}"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=180).stdout
    m = re.search(r'data-probe="([^"]*)"', out)
    assert m, "probe did not run; DOM tail: " + out[-400:]
    return json.loads(htmllib.unescape(m.group(1)))


def _in_iframe(page_html: str, inner_js: str, width: int):
    """Run `inner_js` (a function body taking the iframe window `w` and document `d`) inside a page of exactly `width` px."""
    with tempfile.TemporaryDirectory() as d:
        inner = os.path.join(d, "inner.html")
        with open(inner, "w", encoding="utf-8") as fh:
            fh.write(page_html)
        host = (f'<!doctype html><html><body style="margin:0"><iframe id="f" src="file://{inner}" style="width:{width}px;height:900px;border:0"></iframe>'
                "<script>window.__reportReady=false;var f=document.getElementById('f');"
                "function wait(){var w=f.contentWindow;if(w&&w.__reportReady){window.__reportReady=true;}else{setTimeout(wait,50);}}"
                "f.addEventListener('load',wait);</script></body></html>")
        probe = "async function(){var f=document.getElementById('f');var w=f.contentWindow,d=f.contentDocument;" + inner_js + "}"
        return _run_chrome(host, "(" + probe + ")", width=1280, extra=())


@needs_chrome
def test_all_charts_render_as_svg(page, stub):
    r = _run_chrome(page, "function(){var c=document.querySelectorAll('.chart__canvas svg').length;"
                          "return {svgs:c,count:window.__chartCount,expected:window.__chartsExpected,ready:window.__reportReady,"
                          "registry:Object.keys(window.__charts).length};}")
    n = sum(len(p["charts"]) for p in stub["panels"])
    assert r["ready"] is True and r["count"] == n == r["expected"] == r["registry"]
    assert r["svgs"] == n                                  # vector output: SVG renderer, one svg per chart
    assert "error" not in r


@needs_chrome
def test_colour_tokens_are_resolved_no_raw_at_tokens(page):
    r = _run_chrome(page, "function(){var bad=0;document.querySelectorAll('.chart__canvas svg').forEach(function(s){"
                          "if(/@(good|neutral|concern|accent|grey\\d|text|muted|grid|bg)/.test(s.outerHTML))bad++;});return {bad:bad};}")
    assert r["bad"] == 0


@needs_chrome
def test_375px_viewport_has_no_horizontal_scroll(page):
    r = _in_iframe(page, "var e=d.documentElement;return {sw:e.scrollWidth,cw:e.clientWidth,"
                         "wide:Array.prototype.filter.call(d.querySelectorAll('main *'),function(x){return x.getBoundingClientRect().right>w.innerWidth+1;}).length,"
                         "iw:w.innerWidth};", 375)
    assert r["iw"] == 375
    assert r["sw"] <= r["cw"], r


@needs_chrome
def test_desktop_layout_has_sidebar_nav_and_side_by_side_small_charts(page):
    r = _in_iframe(page, "var nav=d.getElementById('nav').getBoundingClientRect(),main=d.getElementById('report').getBoundingClientRect();"
                         "var sm=d.querySelectorAll('#p-valuation .charts--small .chart');var a=sm[0].getBoundingClientRect(),b=sm[1].getBoundingClientRect();"
                         "return {navRight:nav.right,mainLeft:main.left,sameRow:Math.abs(a.top-b.top)<2,sw:d.documentElement.scrollWidth,cw:d.documentElement.clientWidth};", 1280)
    assert r["navRight"] <= r["mainLeft"] and r["sameRow"] and r["sw"] <= r["cw"]


@needs_chrome
def test_variant_toggle_changes_the_chart(page):
    r = _in_iframe(page, "var id='c-valuation-valuation_strip';var ch=w.__charts[id];var before=JSON.stringify(ch.chart.getOption().series[0].data.map(function(x){return x.value;}));"
                         "var btns=d.querySelectorAll('[data-toggle-for=\"'+id+'\"] button');btns[1].click();"
                         "var after=JSON.stringify(w.__charts[id].chart.getOption().series[0].data.map(function(x){return x.value;}));"
                         "return {before:before,after:after,pressed:[btns[0].getAttribute('aria-pressed'),btns[1].getAttribute('aria-pressed')],variant:w.__charts[id].variant};", 1280)
    assert r["before"] != r["after"] and r["pressed"] == ["false", "true"] and r["variant"] == "report"


@needs_chrome
def test_price_range_toggle_switches_data(page):
    r = _in_iframe(page, "var id='c-price-price_trend';var n=function(){return w.__charts[id].chart.getOption().series[0].data.length;};"
                         "var base=n();var b=d.querySelectorAll('[data-toggle-for=\"'+id+'\"] button');b[0].click();var m6=n();b[2].click();var y2=n();"
                         "return {base:base,m6:m6,y2:y2};", 1280)
    assert r["m6"] < r["base"] < r["y2"]


@needs_chrome
def test_theme_toggle_flips_theme_and_redraws(page):
    r = _in_iframe(page, "var root=d.documentElement;var t0=root.getAttribute('data-theme')||'light';var s0=d.querySelector('.chart__canvas svg').outerHTML.length;"
                         "d.getElementById('theme-btn').click();var t1=root.getAttribute('data-theme');"
                         "var txt=getComputedStyle(d.body).backgroundColor;var svgs=d.querySelectorAll('.chart__canvas svg').length;"
                         "return {t0:t0,t1:t1,bg:txt,svgs:svgs};", 1280)
    assert r["t0"] != r["t1"] and r["svgs"] >= 10


@needs_chrome
def test_nav_highlights_current_panel_on_scroll(page):
    r = _in_iframe(page, "w.scrollTo({top:d.getElementById('p-strength').offsetTop,behavior:'instant'});w.dispatchEvent(new w.Event('scroll'));await new Promise(function(res){setTimeout(res,300);});"
                         "var cur=d.querySelector('.nav__link[aria-current=\"location\"]');return {cur:cur?cur.getAttribute('data-panel'):null};", 1280)
    assert r["cur"] == "strength"


@needs_chrome
def test_chip_popover_shows_calc_on_keyboard_focus(page):
    r = _in_iframe(page, "var chip=d.querySelector('.chip');var pop=chip.querySelector('.popover');var before=getComputedStyle(pop).visibility;"
                         "chip.focus();var after=getComputedStyle(pop).visibility;return {before:before,after:after,text:pop.textContent.slice(0,40)};", 1280)
    assert r["before"] == "hidden" and r["after"] == "visible" and r["text"].strip().startswith("How this was calculated")


@needs_chrome
def test_print_mode_runtime_ready_and_panels_fit_a_page(print_page):
    r = _run_chrome(print_page, "function(){var heights=Array.prototype.map.call(document.querySelectorAll('[data-print-panel]'),function(p){return Math.round(p.getBoundingClientRect().height);});"
                                "return {ready:window.__reportReady,over:window.__printOverflow,heights:heights,theme:document.documentElement.getAttribute('data-theme'),"
                                "nav:!!document.getElementById('nav'),limit:document.getElementById('page-probe').getBoundingClientRect().height,"
                                "openTables:document.querySelectorAll('details.values[open]').length,w:document.querySelector('.layout').getBoundingClientRect().width};}")
    assert r["ready"] is True and r["theme"] == "light" and r["nav"] is False
    assert r["openTables"] >= 10
    assert 680 < r["w"] < 700                                    # 182 mm content width
    assert r["over"] == [], f"panels overflow one A4 page: {r['over']} heights={r['heights']} limit={r['limit']}"


@needs_axe
@pytest.mark.parametrize("width", [1280, 375])
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_axe_no_serious_or_critical_violations(page, width, theme):
    axe_src = open(AXE, encoding="utf-8").read().replace("</script", "<\\/script")
    js = ("var s=d.createElement('script');s.textContent=" + json.dumps(axe_src) + ";d.head.appendChild(s);"
          "if((d.documentElement.getAttribute('data-theme')||'light')!==" + json.dumps(theme) + "){d.getElementById('theme-btn').click();}"
          "await new Promise(function(r){setTimeout(r,300);});"
          "var res=await w.axe.run(d,{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21a','wcag21aa','best-practice']}});"
          "return {theme:d.documentElement.getAttribute('data-theme'),bad:res.violations.filter(function(v){return v.impact==='serious'||v.impact==='critical';})"
          ".map(function(v){return {id:v.id,impact:v.impact,n:v.nodes.length,sample:v.nodes.slice(0,2).map(function(n){return n.html.slice(0,140)+' :: '+(n.failureSummary||'').slice(0,160);})};}),"
          "minor:res.violations.filter(function(v){return v.impact!=='serious'&&v.impact!=='critical';}).map(function(v){return v.id+':'+v.impact;})};")
    r = _in_iframe(page, js, width)
    assert r["theme"] == theme
    assert r["bad"] == [], json.dumps(r["bad"], indent=1)
