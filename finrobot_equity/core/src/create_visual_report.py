#!/usr/bin/env python
# coding: utf-8
"""
Build the visual report formats from a finished analysis folder.

    python create_visual_report.py --analysis-dir output/TICKER/analysis --output-dir output/TICKER/report --format all

--format html   self-contained on-screen report (scrolling, light interaction, works offline)
--format pdf    fixed-layout A4 print report (same content, printed from the same HTML)
--format visual both visual formats (html + pdf)
--format text   the existing text reports (create_equity_report.py and generate_pdf_report.py), unchanged
--format all    text + html + pdf

Both visual formats read only the saved ``<TICKER>_<date>.report.json``; no calculation happens in the renderers.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def run_text_reports(analysis_dir: str, output_dir: str, config_file: str | None) -> int:
    """Run the existing text/professional reports exactly as the web app does, from the analysis summary."""
    with open(os.path.join(analysis_dir, "analysis_summary.json"), encoding="utf-8") as fh:
        s = json.load(fh)
    py, a = sys.executable, analysis_dir
    cfg = config_file or os.path.join(HERE, "..", "config", "config.ini")
    base = ["--company-ticker", s["company_ticker"], "--company-name", s["company_name"]]
    cmd1 = [py, os.path.join(HERE, "create_equity_report.py"), *base,
            "--analysis-csv", os.path.join(a, "financial_metrics_and_forecasts.csv"), "--ratios-csv", os.path.join(a, "ratios_raw_data.csv"),
            "--tagline-file", os.path.join(a, "tagline.txt"), "--company-overview-file", os.path.join(a, "company_overview.txt"),
            "--investment-overview-file", os.path.join(a, "investment_overview.txt"), "--valuation-overview-file", os.path.join(a, "valuation_overview.txt"),
            "--risks-file", os.path.join(a, "risks.txt"), "--competitor-analysis-file", os.path.join(a, "competitor_analysis.txt"),
            "--major-takeaways-file", os.path.join(a, "major_takeaways.txt"), "--news-summary-file", os.path.join(a, "news_summary.txt"),
            "--output-dir", output_dir, "--config-file", cfg, "--data-provider", s.get("data_provider", "finedge")]
    for flag, name in (("--peer-ebitda-csv", "peer_ebitda_comparison.csv"), ("--peer-ev-ebitda-csv", "peer_ev_ebitda_comparison.csv")):
        if os.path.exists(os.path.join(a, name)):
            cmd1 += [flag, os.path.join(a, name)]
    cmd2 = [py, os.path.join(HERE, "generate_pdf_report.py"), *base, "--analysis-dir", a, "--output-dir", output_dir,
            "--config-file", cfg, "--data-provider", s.get("data_provider", "finedge")]
    rc = subprocess.call(cmd1, cwd=HERE)
    return rc or subprocess.call(cmd2, cwd=HERE)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build visual (HTML, PDF) and text reports from an analysis folder.")
    ap.add_argument("--analysis-dir", required=True, help="Folder written by generate_financial_analysis.py (needs report_context.json).")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--format", choices=["html", "pdf", "visual", "text", "all"], default="all")
    ap.add_argument("--report-date", default=None, help="Override the report date (default: the analysis date).")
    ap.add_argument("--config-file", default=None, help="Used only by --format text/all.")
    ap.add_argument("--strict-validation", action="store_true", help="Stop (exit 2) if the report model has validation errors.")
    ap.add_argument("--no-pdf-checks", action="store_true", help="Skip the post-build PDF text and page checks.")
    args = ap.parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    rc = 0
    if args.format in ("text", "all"):
        rc = run_text_reports(args.analysis_dir, args.output_dir, args.config_file)
        if rc:
            print("Text reports failed.")
            if args.format == "text":
                return rc
    if args.format in ("html", "pdf", "visual", "all"):
        from report_visual import model as M
        model = M.build_report_model(args.analysis_dir, report_date=args.report_date)
        json_path = M.save_report_model(model, args.output_dir)
        v = model["sources"]["validation"]
        print(f"Report model: {json_path}")
        print(f"Validation: {len(v['errors'])} error(s), {len(v['warnings'])} warning(s), {len(v['passed'])} check group(s) passed")
        for e in v["errors"]:
            print(f"  ERROR: {e}")
        for w in v["warnings"]:
            print(f"  warning: {w}")
        if v["errors"] and args.strict_validation:
            print("Strict validation is on and errors were found: stopping.")
            return 2
        stem = f"{model['meta']['ticker']}_Visual_Report_{model['meta']['report_date']}"
        if args.format in ("html", "visual", "all"):
            from report_visual.render_html import render_html
            print("HTML:", render_html(model, os.path.join(args.output_dir, stem + ".html")))
        if args.format in ("pdf", "visual", "all"):
            from report_visual.render_pdf import render_pdf
            print("PDF:", render_pdf(model, os.path.join(args.output_dir, stem + ".pdf"), check=not args.no_pdf_checks))
    return rc


if __name__ == "__main__":
    sys.exit(main())
