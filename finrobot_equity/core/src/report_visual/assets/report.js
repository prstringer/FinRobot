/* Visual report runtime. Reads the embedded report JSON, draws charts with ECharts (SVG renderer so print is vector),
   resolves colour tokens per theme, installs one generic tooltip formatter, and wires the nav, theme and chart toggles.
   No network access. Text and tables are rendered server-side; this script only draws charts and adds interaction. */
(function () {
  "use strict";
  var doc = document, root = doc.documentElement;
  var PRINT = doc.body.classList.contains("print");
  var model = JSON.parse(doc.getElementById("report-data").textContent);
  var FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';

  /* ---------- theme ---------- */
  function currentTheme() { return root.getAttribute("data-theme") || "light"; }
  function setTheme(t, persist) {
    root.setAttribute("data-theme", t);
    var btn = doc.getElementById("theme-btn");
    if (btn) { btn.setAttribute("aria-pressed", t === "dark" ? "true" : "false"); btn.textContent = t === "dark" ? "Light theme" : "Dark theme"; }
    if (persist) { try { localStorage.setItem("report-theme", t); } catch (e) { /* storage may be unavailable */ } }
  }
  if (PRINT) { root.setAttribute("data-theme", "light"); }
  else {
    var saved = null;
    try { saved = localStorage.getItem("report-theme"); } catch (e) { saved = null; }
    var prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    setTheme(saved || root.getAttribute("data-theme") || (prefersDark ? "dark" : "light"), false);
    var tbtn = doc.getElementById("theme-btn");
    if (tbtn) tbtn.addEventListener("click", function () { setTheme(currentTheme() === "dark" ? "light" : "dark", true); renderAll(); });
  }

  /* ---------- tokens ---------- */
  var TOKENS = ["good", "neutral", "concern", "accent", "grey1", "grey2", "grey3", "grey4", "grey5", "text", "muted", "grid", "bg", "surface", "border"];
  function palette() {
    var cs = getComputedStyle(root), p = {};
    TOKENS.forEach(function (t) { p[t] = cs.getPropertyValue("--c-" + t).trim() || "#888888"; });
    return p;
  }
  function resolve(node, p) {
    if (typeof node === "string") {
      return node.replace(/@([a-z0-9]+)/g, function (m, n) { return p[n] || m; });
    }
    if (Array.isArray(node)) return node.map(function (x) { return resolve(x, p); });
    if (node && typeof node === "object") {
      var o = {};
      Object.keys(node).forEach(function (k) { o[k] = resolve(node[k], p); });
      return o;
    }
    return node;
  }

  /* ---------- generic tooltip ---------- */
  function esc(s) { return String(s).replace(/[&<>"']/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]; }); }
  function tipFormatter(params) {
    var arr = Array.isArray(params) ? params : [params], seen = {}, out = [];
    arr.forEach(function (x) {
      var d = x.data, line;
      if (d && typeof d === "object" && !Array.isArray(d) && d.tip) line = d.tip;
      else if (x.seriesName || x.name || x.value !== undefined) {
        var v = Array.isArray(x.value) ? x.value.join(", ") : x.value;
        line = [x.seriesName, x.name, v].filter(function (z) { return z !== undefined && z !== null && z !== ""; }).join(": ");
      } else return;
      if (!seen[line]) { seen[line] = 1; out.push(esc(line)); }
    });
    return out.join("<br/>");
  }

  function labelDefaults(node, p) {
    // ECharts draws labels with a fixed #333 fill and a white outline. Use the theme text colour and no outline,
    // so labels stay legible in both themes (explicit colours in the spec win).
    if (Array.isArray(node)) { node.forEach(function (x) { labelDefaults(x, p); }); return; }
    if (!node || typeof node !== "object") return;
    ["label", "endLabel"].forEach(function (k) {
      var l = node[k];
      if (l && typeof l === "object") {
        if (l.textBorderWidth === undefined) l.textBorderWidth = 0;
        if (l.textBorderColor === undefined) l.textBorderColor = "transparent";
        if (l.color === undefined) l.color = (typeof l.position === "string" && l.position.indexOf("inside") === 0) ? p.bg : p.text;
      }
    });
    Object.keys(node).forEach(function (k) { labelDefaults(node[k], p); });
  }

  function prepare(option, p) {
    var o = resolve(option, p);
    labelDefaults(o, p);
    ["xAxis", "yAxis"].forEach(function (k) {
      [].concat(o[k] || []).forEach(function (ax) {
        if (ax && typeof ax === "object") {
          ax.axisLabel = ax.axisLabel || {};
          if (ax.axisLabel.hideOverlap === undefined) ax.axisLabel.hideOverlap = true;   // never print labels on top of each other
        }
      });
    });
    o.animation = false;
    o.backgroundColor = "transparent";
    o.textStyle = Object.assign({ fontFamily: FONT, color: p.text, fontSize: 12 }, o.textStyle || {});
    o.aria = { enabled: false };
    o.tooltip = Object.assign({
      trigger: "item", confine: true, backgroundColor: p.surface, borderColor: p.border,
      textStyle: { color: p.text, fontSize: 12 }, extraCssText: "max-width:260px;white-space:normal;"
    }, o.tooltip || {}, { formatter: tipFormatter });
    return o;
  }

  /* ---------- charts ---------- */
  var registry = {};     // id -> {chart, def, variant}
  window.__charts = registry;

  function allCharts() {
    var list = [];
    (model.panels || []).forEach(function (pn) { (pn.charts || []).forEach(function (c) { list.push({ panel: pn, def: c }); }); });
    return list;
  }
  function chartOption(def, variantKey) {
    if (def.variants && def.variants.length) {
      var key = variantKey || def.default_variant || def.variants[0].key, v = def.variants[0];
      def.variants.forEach(function (x) { if (x.key === key) v = x; });
      return v.option;
    }
    return def.option;
  }
  function drawChart(entry) {
    var p = palette();
    entry.chart.setOption(prepare(chartOption(entry.def, entry.variant), p), true);
  }
  function renderAll() { Object.keys(registry).forEach(function (k) { drawChart(registry[k]); }); }

  function initCharts() {
    if (typeof echarts === "undefined") return;
    allCharts().forEach(function (it) {
      var id = "c-" + it.panel.id + "-" + it.def.id, el = doc.getElementById(id);
      if (!el) return;
      var chart = echarts.init(el, null, { renderer: "svg" });
      var entry = { chart: chart, def: it.def, variant: it.def.default_variant || null, el: el };
      registry[id] = entry;
      drawChart(entry);
      if ("ResizeObserver" in window) new ResizeObserver(function () { chart.resize(); }).observe(el);
      var group = doc.querySelector('[data-toggle-for="' + id + '"]');
      if (group && !PRINT) {
        group.addEventListener("click", function (ev) {
          var b = ev.target.closest("button[data-variant]");
          if (!b) return;
          entry.variant = b.getAttribute("data-variant");
          Array.prototype.forEach.call(group.querySelectorAll("button"), function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
          drawChart(entry);
          var live = doc.getElementById("live");
          if (live) live.textContent = (it.def.title || "Chart") + ": showing " + b.textContent;
        });
      }
    });
    window.addEventListener("resize", function () { Object.keys(registry).forEach(function (k) { registry[k].chart.resize(); }); });
  }

  /* ---------- navigation ---------- */
  function initNav() {
    var nav = doc.getElementById("nav");
    if (!nav || PRINT) return;
    var toggle = doc.getElementById("nav-toggle"), list = doc.getElementById("nav-list");
    var links = Array.prototype.slice.call(nav.querySelectorAll("a[data-panel]"));
    function narrow() { return !window.matchMedia("(min-width: 1024px)").matches; }
    function sync() { if (narrow()) { list.hidden = toggle.getAttribute("aria-expanded") !== "true"; } else { list.hidden = false; } }
    toggle.addEventListener("click", function () {
      toggle.setAttribute("aria-expanded", toggle.getAttribute("aria-expanded") === "true" ? "false" : "true"); sync();
    });
    links.forEach(function (a) { a.addEventListener("click", function () { if (narrow()) { toggle.setAttribute("aria-expanded", "false"); sync(); } }); });
    window.addEventListener("resize", sync);
    sync();
    function setCurrent(id) {
      links.forEach(function (a) {
        if (a.getAttribute("data-panel") === id) {
          a.setAttribute("aria-current", "location");
          var pos = doc.getElementById("nav-pos");
          if (pos) pos.textContent = a.getAttribute("data-n") + " of " + links.length;
          var txt = doc.getElementById("nav-text");
          if (txt) txt.textContent = a.getAttribute("data-headline");
        } else a.removeAttribute("aria-current");
      });
    }
    var panels = Array.prototype.slice.call(doc.querySelectorAll("[data-panel-id]"));
    if ("IntersectionObserver" in window && panels.length) {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) { if (e.isIntersecting) setCurrent(e.target.getAttribute("data-panel-id")); });
      }, { rootMargin: "-25% 0px -65% 0px", threshold: 0 });
      panels.forEach(function (p) { io.observe(p); });
    }
    function byScroll() {
      var cur = panels.length ? panels[0] : null, line = window.innerHeight * 0.3;
      panels.forEach(function (p) { if (p.getBoundingClientRect().top <= line) cur = p; });
      if (cur) setCurrent(cur.getAttribute("data-panel-id"));
    }
    window.addEventListener("scroll", byScroll, { passive: true });
    byScroll();
  }

  /* ---------- popover: Escape closes ---------- */
  doc.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && doc.activeElement && doc.activeElement.classList.contains("chip")) doc.activeElement.blur();
  });

  /* ---------- print readiness ---------- */
  function checkPrintFit() {
    var probe = doc.getElementById("page-probe"), limit = probe ? probe.getBoundingClientRect().height : 0, over = [];
    Array.prototype.forEach.call(doc.querySelectorAll("[data-print-panel]"), function (p) {
      var h = p.getBoundingClientRect().height;
      if (limit && h > limit + 0.5) over.push({ panel: p.getAttribute("data-panel-id"), height_px: Math.round(h), limit_px: Math.round(limit) });
    });
    window.__printOverflow = over;
  }

  function ready() {
    var expected = allCharts().length, drawn = Object.keys(registry).length;
    window.__chartCount = drawn;
    window.__chartsExpected = expected;
    if (PRINT) checkPrintFit();
    window.__reportReady = (drawn === expected);
    // mirrored into the DOM so a plain `chrome --dump-dom` run (no automation library) can read the print contract
    doc.documentElement.setAttribute("data-report-ready", String(window.__reportReady));
    doc.documentElement.setAttribute("data-print-overflow", JSON.stringify(window.__printOverflow || []));
  }

  window.__reportReady = false;
  initNav();
  initCharts();
  // charts are drawn synchronously (animation off); a frame later the layout has settled
  requestAnimationFrame(function () { requestAnimationFrame(ready); });
  if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(function () { if (PRINT) checkPrintFit(); });
})();
