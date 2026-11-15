# Visual report: contracts

Pipeline: `report_context.json` (+ analysis CSVs) -> `model.py` (verdicts, headlines, charts) -> `<TICKER>_<date>.report.json`
-> `render_html.py` (self-contained HTML) -> `render_pdf.py` (same HTML, print CSS, headless Chromium).
Renderers read ONLY the report JSON. No calculations in renderers. Python package: `report_visual` (under core/src).

## Report JSON (top level)

```
{
  "schema_version": 1,
  "meta": { "ticker", "company_name", "sector", "kind": "nonfinancial"|"bank", "report_date": "2026-11-15",
            "generated_at": ISO, "currency": "INR",
            "snapshot": {"price": 249.0, "price_date": "2026-11-13", "price_time": "15:30 IST" | null,
                         "market_cap_cr": 23524, "week52_low": 107.4, "week52_high": 267.4} },
  "thesis": "one sentence",
  "warnings": [ {"level": "error"|"warning"|"info", "text": "..."} ],        # validation flags + data limitations, shown on Panel 1
  "panels": [ Panel x7, in order ],
  "glossary": [ {"id": "ev_ebitda", "term": "EV/EBITDA", "definition": "..."} ],
  "sources": { "provider": "FinEdge API", "snapshot": "...", "definitions": [ "..." ], "validation": {"errors": [], "warnings": [], "passed": []} },
  "disclaimer": "...",
  "tokens_used": ["good", ...]      # optional
}
```

## Panel

```
{ "id": "summary"|"profit_engine"|"cyclicality"|"strength"|"valuation"|"price"|"watch",
  "n": 1..7, "nav": "short label (<=2 words)", "question": "...",
  "headline": "<= 8 words, states a conclusion",
  "charts": [ Chart ],                 # exactly one role "main"; 0..2 role "small"
  "chips": [ Verdict ],                # 1..3 per panel (summary panel may have 0; its lights are in "extra")
  "extra": { panel-specific, see below },
  "notes": [ "short static text printed under the charts, e.g. capex flag" ] }
```

Panel `extra` blocks:
* summary: `{"lights": [{"key": "business"|"profit"|"price"|"momentum", "label", "light": "good"|"neutral"|"concern", "reason": "one line"}],
  "price_vs_earnings": {"price_change_1y_pct", "eps_fy_change_pct", "eps_ttm_change_pct", "sentence"},
  "header_strip": {...}}`
* profit_engine: `{"bridge": {"from","to","d_ebitda_margin_pts","d_contribution_margin_pts","d_sga_pct_pts","text"}, "operating_leverage": {"text","ebitda_per_100_revenue"}}`
* price: `{"tests": [{"key": "trend"|"momentum"|"participation"|"stretch", "label", "measures", "state", "verdict": good|neutral|concern, "evidence"}],
  "verdict_sentence", "divergence": {...}|null, "rsi_zone", "readings": [[label,value,note],...]}`
* valuation: `{"fair_value_cases": [{"key","label","basis","low_price","high_price","vs_price_pct_low","vs_price_pct_high"}], "price": 249.0,
  "toggle_note": "At the report's price of about Rs 129, EV/EBITDA was 16.6x"}`
* watch: `{"signposts": [{"metric","current","current_text","threshold_text","status": good|neutral|concern,"why"}],
  "portfolio": {"current_weight": null, "target_weight": null, "triggers": null}}`   # hidden when all null

## Verdict (chip) object

```
{ "id": "contribution_margin", "metric": "contribution_margin", "label": "Contribution margin",
  "value": 46.2, "value_text": "46.2%", "unit": "%", "period": "FY26",
  "verdict": "good"|"neutral"|"concern", "icon": "check"|"dash"|"alert",
  "yardstick": "own_history"|"peers"|"general_norm", "yardstick_text": "vs own history" | "vs peers" | "General norm, not company-specific",
  "scale": {"min": 44.2, "max": 47.2, "peer_median": null, "higher_is_better": true, "value": 46.2},
  "so_what": "<= ~25 words, does not restate the number",
  "source": "report_data"|"calculated"|"general_norm",
  "calc": "how it was computed + inputs, shown on hover/tap and printed in data notes" }
```

## Chart object (the declarative spec)

```
{ "id": "margin_scissors", "role": "main"|"small", "title": "...", "subtitle": "..." | null,
  "option": { ECharts option, PURE JSON (no functions) },
  "alt": "ARIA description generated from the data, e.g. 'SG&A rose from 30.7% to 37.8% of revenue ...'",
  "datatable": {"columns": ["Year","Contribution margin %", ...], "rows": [["FY22", "45.3", ...], ...]},   # every plotted value, as strings
  "variants": [ {"key": "1y", "label": "1 year", "option": {...}} ] | null,     # toggles; first/`default_variant` is the static state
  "default_variant": "1y" | null,
  "static_note": "printed in the PDF in place of the toggle" | null,
  "height": 260 }                                                                 # px at desktop; renderer scales on mobile
```

### Chart rules (apply to every option)
* Colours are TOKENS, never hex: strings `"@good" "@neutral" "@concern" "@accent" "@grey1".."@grey5" "@text" "@muted" "@grid" "@bg"`.
  The renderer resolves tokens per theme (light/dark; PDF always light). `@good/@neutral/@concern` are for verdicts ONLY.
  Data uses greys plus ONE accent (`@accent` = this company); peers `@grey3`.
* No functions in options. Labels/tooltips come from data: each data item may be `{"value": ..., "tip": "FY26: SG&A 37.8% of revenue", "label": {"show": true, "formatter": "37.8%"}}`.
  The renderer installs one generic tooltip formatter that shows `params.data.tip` (fallback: series name and value).
* Time left to right, oldest to newest: FY22 ... FY26, TTM, then forecasts `2027E`... Forecast = dashed lines / hatched bars + label "Forecast" (markArea or label).
* Direct labels, no legends where possible; annotate the point that matters (markPoint/markLine with `label.formatter` static strings).
* Bars start at zero. No pie, radar, 3D. No dual axes unless unavoidable (then axis names must say which series).
* Indian formats: "Rs 9,399 cr" (lakh-style grouping `12,34,567` for amounts above 1 lakh when shown in rupees), percentages one decimal.
  Pre-format all label strings in Python.
* `animation: false` in every option (print-safe, test-safe). Font family is set by the renderer; use `textStyle` only for size.
* Use the SVG-safe subset (renders identically with `renderer: 'svg'`).

## Python interfaces
* `report_visual/charts.py`: `build_panel_charts(ctx: dict, analysis_dir: str, kind: str = "nonfinancial") -> dict[str, list[Chart]]`
  keyed by panel id. `ctx` is `report_context.json`; `analysis_dir` holds `historical_price_full.csv`, `income_statement_raw_data.csv`,
  `balance_sheet_raw_data.csv`, `cash_flow_raw_data.csv`, `ratios_raw_data.csv`, `financial_metrics_and_forecasts.csv`, peer CSVs.
* `report_visual/model.py`: `build_report_model(analysis_dir: str, report_date: str|None = None, kind: str|None = None) -> dict` and `save_report_model(model, out_dir) -> path`.
* `report_visual/render_html.py`: `render_html(model: dict, out_path: str) -> str`; also `render_html_string(model, print_mode=False)`.
* `report_visual/render_pdf.py`: `render_pdf(model: dict, out_path: str) -> str` (HTML in print mode -> Chromium -> PDF; fails the build if a panel overflows its page).
* CLI: `--format html|pdf|text|all` on `create_visual_report.py` (new entry) and wired into the web app later.
