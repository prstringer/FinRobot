#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys
import pandas as pd
import json 

from modules.common_utils import load_config, get_api_key
from modules.financial_data_processor import calculate_growth_and_forecasts, extract_historical_metrics_from_api_data
from modules.market_data_api import (
    get_comprehensive_financial_data,
    combine_peer_financial_data,
    project_ebitda_for_peers,
    get_company_news,
    get_fmp_peers,
)
from modules.text_generator_agents import generate_text_section, _prepare_user_prompt
from modules import report_context as rctx
from modules import technical_analysis as tech_an
from modules import cashflow_analysis as cf_an
from modules import ratio_bars as rbars
from modules import cost_taxonomy
from modules import fiscal_guard
from modules import sentiment_gate as sgate
from modules.llm_providers import PROVIDERS, resolve_llm_settings

# 新增模块导入
from modules.sensitivity_analyzer import SensitivityAnalyzer
from modules.catalyst_analyzer import CatalystAnalyzer
from modules.news_integrator import NewsIntegrator, get_enhanced_company_news, build_enhanced_news
from modules.google_news import get_google_news
from modules.news_analysis_llm import annotate_news_with_llm
from modules.retail_sentiment_client import RetailSentimentClient
from modules import currency
from modules.finedge_api import (
    FinEdgeClient,
    FinEdgeError,
    resolve_finedge_token,
    normalize_symbol,
    get_finedge_comprehensive_financial_data,
    get_finedge_company_news,
    get_finedge_peers,
    filter_valid_peers,
    get_finedge_price_history,
    get_finedge_company_metrics,
    get_finedge_ttm_and_latest_balance,
    get_finedge_peer_current_multiples,
    get_finedge_peer_frames,
    normalize_cash_flow,
    get_finedge_technical_indicators,
    combine_peer_financial_data_finedge,
    combine_peer_bank_data_finedge,
    build_bank_metrics_table,
)

def main():
    parser = argparse.ArgumentParser(description="Generate financial analysis data from FMP (US) or FinEdge (India) APIs.")
    
    # Company Identifiers
    parser.add_argument("--company-ticker", type=str, required=True, help="Stock ticker (e.g., AAPL).")
    parser.add_argument("--company-name", type=str, required=True, help="Full company name (e.g., Apple Inc.).")
    
    # API Configuration
    parser.add_argument("--config-file", type=str, default=None, help="Path to the configuration file (e.g., config.ini).")
    parser.add_argument("--years-limit", type=int, default=5, help="Number of years of historical data to fetch.")
    parser.add_argument("--data-provider", type=str, default="fmp", choices=["fmp", "finedge"],
                        help="Financial data source: 'fmp' (Financial Modeling Prep, USD) or 'finedge' "
                             "(FinEdge API for NSE/BSE stocks, INR). FinEdge reads FINEDGE_API_TOKEN or "
                             "finedge_api_token in config.ini.")
    parser.add_argument("--statement-type", type=str, default="auto", choices=["auto", "c", "s"],
                        help="FinEdge only: consolidated (c) or standalone (s) statements. 'auto' uses "
                             "consolidated for companies and standalone for banks.")
    parser.add_argument("--no-auto-peers", action="store_true",
                        help="Do not look up sector peers automatically when --peer-tickers is not given.")
    parser.add_argument("--max-peers", type=int, default=5, help="Maximum auto-discovered peers.")
    
    # Output Configuration
    parser.add_argument("--output-dir", type=str, help="Directory to save all outputs. Default: ./output/[TICKER]/analysis/")
    parser.add_argument("--output-csv-name", type=str, default="financial_metrics_and_forecasts.csv", help="Name for the output CSV file.")

    # Peer Analysis
    parser.add_argument("--peer-tickers", type=str, nargs="*", default=[], help="List of peer tickers for comparative analysis (e.g., GOOG MSFT).")

    # Text Generation
    parser.add_argument("--allow-llm-fallback", action="store_true",
                        help="Do not fail the run when an AI text section falls back to placeholder text (default: fail).")
    parser.add_argument("--generate-text-sections", action="store_true", help="Enable generation of text sections with an LLM.")
    parser.add_argument("--llm-provider", type=str, default=None, choices=list(PROVIDERS),
                        help="LLM for the text sections: openai, anthropic (Claude) or gemini. "
                             "Default: llm_provider in config.ini, else openai.")
    parser.add_argument("--llm-model", type=str, default=None,
                        help="Model name for the chosen provider (default: llm_model in config.ini, else the provider default).")
    parser.add_argument("--text-output-dir", type=str, default=None, help="Directory to save generated text files.")

    # NEWS PARAMETERS
    parser.add_argument("--news-days-back", type=int, default=None, help="Number of days to look back for company news (default: 5 for FMP, 30 for FinEdge exchange announcements)")
    parser.add_argument("--news-limit", type=int, default=50, help="Maximum number of news articles to fetch (default: 50)")
    parser.add_argument("--retail-sentiment-days-back", type=int, default=7, help="Number of days to look back for retail sentiment insights (default: 7)")

    # 新增分析选项
    parser.add_argument("--enable-sensitivity-analysis", action="store_true", help="Enable sensitivity analysis for forecasts")
    parser.add_argument("--enable-catalyst-analysis", action="store_true", help="Enable catalyst identification and analysis")
    parser.add_argument("--enable-enhanced-news", action="store_true", help="Enable enhanced news integration with categorization")

    # Forecast Configuration
    # The three growth assumptions apply to the three years after the latest actual year
    # (e.g. 2027E-2029E when the latest actual is FY2026). Flag names are kept for compatibility.
    parser.add_argument("--revenue-growth-2025", type=float, default=0.05, help="Revenue growth for forecast year 1, the year after the latest actual (default: 5%%)")
    parser.add_argument("--revenue-growth-2026", type=float, default=0.06, help="Revenue growth for forecast year 2 (default: 6%%)")
    parser.add_argument("--revenue-growth-2027", type=float, default=0.04, help="Revenue growth for forecast year 3 (default: 4%%)")
    parser.add_argument("--margin-improvement", type=float, default=0.01, help="Annual margin improvement assumption (default: 1%%)")
    parser.add_argument("--sga-margin-improvement", type=float, default=-0.005, help="SG&A margin change assumption (default: -0.5%% efficiency gain)")
    parser.add_argument("--margin-trigger", type=str, default=None,
                        help="A stated fundamental reason (e.g. 'capex completion in FY28') that lifts the forecast EBITDA margin cap "
                             "(historical 5-year peak margin + 1 standard deviation). The trigger text is recorded as a flag in the report.")

    # API Options
    parser.add_argument("--period", type=str, default="annual", choices=["annual", "quarterly"], help="Data period (annual or quarterly)")

    args = parser.parse_args()
    failed_text_sections = []  # AI sections that fell back to placeholder text
    llm_usage = []  # per-call token usage and estimated cost of the AI text sections

    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(".", "output", args.company_ticker, "analysis")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")

    # Text output directory
    text_output_dir = args.text_output_dir if args.text_output_dir else output_dir
    if args.generate_text_sections:
        os.makedirs(text_output_dir, exist_ok=True)
        print(f"Text outputs will be saved to: {text_output_dir}")

    # Load configuration and API keys
    use_finedge = args.data_provider == "finedge"
    if use_finedge:
        args.company_ticker = normalize_symbol(args.company_ticker)
        currency.set_currency("INR")
    if args.news_days_back is None:
        args.news_days_back = 30 if use_finedge else 5
    llm_settings = None
    fmp_api_key = None
    finedge_client = None
    adanos_api_key = os.getenv("ADANOS_API_KEY")
    adanos_base_url = os.getenv("ADANOS_BASE_URL", "https://api.adanos.org")
    try:
        try:
            config = load_config(args.config_file)
        except FileNotFoundError:
            # FinEdge can run from environment variables alone (FINEDGE_API_TOKEN, and the
            # LLM key such as OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY)
            if not use_finedge:
                raise
            config = None
            print("No config.ini found; using API keys from the environment.")
        if use_finedge:
            finedge_client = FinEdgeClient(resolve_finedge_token(config))
            if config is not None:
                fmp_api_key = config.get("API_KEYS", "fmp_api_key", fallback=None)
        else:
            fmp_api_key = get_api_key(config, section="API_KEYS", key="fmp_api_key")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Please ensure config.ini exists with valid API keys:")
        print("[API_KEYS]")
        print("fmp_api_key = YOUR_FMP_API_KEY            # for --data-provider fmp")
        print("finedge_api_token = YOUR_FINEDGE_TOKEN   # for --data-provider finedge (or set FINEDGE_API_TOKEN)")
        print("llm_provider = openai                   # openai | anthropic | gemini")
        print("openai_api_key / anthropic_api_key / gemini_api_key = YOUR_KEY")
        return
    try:
        if config is not None:
            adanos_api_key = config.get("API_KEYS", "adanos_api_key", fallback=adanos_api_key)
            adanos_base_url = config.get("API_KEYS", "adanos_base_url", fallback=adanos_base_url)
        if args.generate_text_sections:
            llm_settings = resolve_llm_settings(config, args.llm_provider, args.llm_model)
            print(f"Text sections will use {llm_settings.label}")
            if llm_settings.base_url:
                print(f"Using OpenAI-compatible base URL: {llm_settings.base_url}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Please ensure config.ini exists with valid API keys:")
        print("[API_KEYS]")
        print("llm_provider = openai   # openai | anthropic | gemini")
        print("openai_api_key / anthropic_api_key / gemini_api_key = YOUR_KEY")
        return

    provider_name = "FinEdge API" if use_finedge else "FMP API"
    print(f"Starting {provider_name}-based financial analysis for {args.company_name} ({args.company_ticker})")

    # 1. Fetch Financial Data
    print(f"Fetching financial data from {provider_name}...")
    data_meta = {"data_source": "Financial Modeling Prep API", "currency": "USD", "is_bank": False, "warnings": []}
    if use_finedge:
        try:
            financial_data = get_finedge_comprehensive_financial_data(
                symbol=args.company_ticker,
                client=finedge_client,
                period=args.period,
                limit=args.years_limit,
                statement_type=args.statement_type,
            )
        except FinEdgeError as e:
            print(f"Error: Could not fetch FinEdge data for {args.company_ticker}: {e}")
            print("On the free plan FinEdge only serves the sample symbols (ITC, RELIANCE, HDFCBANK, ICICIPRULI).")
            return
        data_meta = financial_data.pop("meta")
        for w in data_meta.get("warnings", []):
            print(f"⚠️  Data note: {w}")
    else:
        financial_data = get_comprehensive_financial_data(
            ticker=args.company_ticker, 
            api_key=fmp_api_key, 
            period=args.period, 
            limit=args.years_limit
        )
    is_bank = bool(data_meta.get("is_bank"))

    # Check if we got the required data
    if financial_data.get('income_statement') is None or financial_data['income_statement'].empty:
        print(f"Error: Could not fetch income statement data from {provider_name}. Exiting.")
        print("Please check:")
        print("1. API key/token is valid and has remaining quota")
        print("2. Ticker symbol is correct") 
        print("3. Internet connection is working")
        return

    print(f"Successfully fetched financial data from {provider_name}")
    income_df = financial_data['income_statement']
    print(f"Retrieved {len(income_df)} years of income statement data")
    
    # Display available years for confirmation
    available_years = sorted(income_df['year'].tolist(), reverse=True)
    print(f"Available years: {available_years}")

    # 2. Process Historical Metrics
    print("Processing historical financial metrics...")
    if is_bank:
        print("Bank detected: building bank metrics (NII, NIM, asset quality, capital).")
        historical_metrics_df = build_bank_metrics_table(financial_data)
    else:
        historical_metrics_df = extract_historical_metrics_from_api_data(financial_data)

    if historical_metrics_df is None or historical_metrics_df.empty:
        print("Error: Failed to process historical metrics from API data. Exiting.")
        return
    
    print(f"\nHistorical Metrics Extracted from {provider_name}:")
    print(historical_metrics_df.to_string())

    # 3. Forecasting
    print("Generating forecasts...")
    
    # Determine the latest actual year for base year
    actual_years = [col for col in historical_metrics_df.columns if col.endswith("A") and col != "metrics"]
    latest_year = max(actual_years) if actual_years else "2024A"
    
    print(f"Using {latest_year} as base year for forecasts")
    
    # Forecast the three years after the latest actual year
    base_year_num = int(str(latest_year)[:4])
    forecast_labels = [f"{base_year_num + i}E" for i in (1, 2, 3)]
    growth_inputs = [args.revenue_growth_2025, args.revenue_growth_2026, args.revenue_growth_2027]
    forecast_config = {
        "revenue_base_year": latest_year,
        "revenue_growth_assumptions": dict(zip(forecast_labels, growth_inputs)),
        "ebitda_growth_factor": 1.05,
        "margin_improvement": {} if is_bank else {
            "Contribution Margin": args.margin_improvement, 
            "EBITDA Margin": args.margin_improvement
        },
        "sga_margin_change": 0 if is_bank else args.sga_margin_improvement,
        "margin_trigger": args.margin_trigger or None,
    }
    
    print(f"Forecast assumptions:")
    growth_name = "Total Income Growth" if is_bank else "Revenue Growth"
    for label, g in zip(forecast_labels, growth_inputs):
        print(f"  {growth_name} {label}: {g*100:.1f}%")
    print(f"  Margin Improvement: {args.margin_improvement*100:.1f}% annually")
    print(f"  SG&A Efficiency: {args.sga_margin_improvement*100:.1f}% annually")
    
    # Current market data (price, market cap, 52w range, TTM ratios) is fetched once here: the forecast P/E, the
    # current-valuation block and every text section all use these same values.
    company_metrics = {}
    if use_finedge:
        try:
            company_metrics = get_finedge_company_metrics(args.company_ticker, finedge_client, {**financial_data, "meta": data_meta}) or {}
        except Exception as e:
            print(f"Warning: could not fetch current market data for the forecast P/E and valuation context: {e}")
    current_price = company_metrics.get("share_price")
    bridge_income_df = income_df if (use_finedge and not is_bank) else None
    final_data_df = calculate_growth_and_forecasts(historical_metrics_df, forecast_config,
                                                   income_df=bridge_income_df, current_price=current_price)
    # Cost taxonomy (industrial vs generic labels), split cost rows, and the forecast margin-cap flag as data notes
    _macro = None
    if use_finedge:
        try:
            _macro = (finedge_client.profile(normalize_symbol(args.company_ticker)) or {}).get("macro_sector")
        except Exception:
            _macro = None
    taxonomy = (cost_taxonomy.resolve(kind="generic") if is_bank else
                cost_taxonomy.resolve(company_metrics.get("sector"), company_metrics.get("industry"), _macro))
    if use_finedge and not is_bank:
        final_data_df = cost_taxonomy.add_split_rows(final_data_df, income_df, taxonomy)
    final_data_df.attrs["taxonomy"] = taxonomy
    print(f"Cost taxonomy: {taxonomy['kind']} (sector {taxonomy.get('sector')!r}, matched {taxonomy.get('matched')!r})")
    for _note in cost_taxonomy.notes(taxonomy):
        data_meta.setdefault("warnings", []).append(_note)
    _cap_flag = final_data_df.attrs.get("margin_cap")
    if _cap_flag and _cap_flag.get("message"):
        data_meta.setdefault("warnings", []).append(_cap_flag["message"])
    if final_data_df.attrs.get("forecast_bridge") is not None:
        print("\nForecast P&L bridge (INR crore; EPS in rupees):")
        _bridge = final_data_df.attrs["forecast_bridge"].drop(columns=["shares"]).copy()
        for _c in _bridge.columns:
            if _c != "eps":
                _bridge[_c] = _bridge[_c] / rctx.CRORE
        print(_bridge.round(2).T.to_string())
    print("\nFinal Metrics with Forecasts:")
    print(final_data_df.to_string())

    # Save the main analysis file
    output_csv_path = os.path.join(output_dir, args.output_csv_name)
    final_data_df.to_csv(output_csv_path, index=False)
    print(f"Successfully saved financial analysis to: {output_csv_path}")

    # 4. Peer Comparison Analysis
    projected_peer_ebitda = None
    df_ebitda_peers = None
    df_ev_ebitda_peers = None
    df_peer_current = None
    df_bank_peers = None
    peer_selection_log = None
    if use_finedge:
        from modules import peer_selection as psel
        peer_list = [normalize_symbol(t) for t in args.peer_tickers]
        if peer_list:
            peer_list, ignored = filter_valid_peers(peer_list, finedge_client)
            if ignored:
                print(f"⚠️  Ignoring unrecognised peer symbol(s): {ignored} - using automatic sector peers instead.")
        if peer_list:
            # user-supplied peers are kept as given; any outside the size band or sub-sector are warned about
            peer_selection_log = psel.select_peers(args.company_ticker, finedge_client, explicit=peer_list)
        elif not args.no_auto_peers:
            # automatic peers: same sub-sector and 0.2x to 5.0x the market cap (never silently widened)
            peer_selection_log = psel.select_peers(args.company_ticker, finedge_client, max_peers=args.max_peers)
            peer_list = peer_selection_log["peers"]
        args.peer_tickers = peer_list
        if peer_selection_log is not None:
            margin_check = None
            if not is_bank and peer_list:
                try:
                    _, peer_income = get_finedge_peer_frames(peer_list, finedge_client)
                    peer_margins = {}
                    for _t, _inc in peer_income.items():
                        _r = _inc.sort_values("year").iloc[-1] if _inc is not None and not _inc.empty else None
                        peer_margins[_t] = (float(_r["ebitda"]) / float(_r["revenue"]) * 100) if _r is not None and _r["revenue"] else None
                    _em = final_data_df.loc[final_data_df["metrics"] == "EBITDA Margin", latest_year]
                    subject_margin = float(str(_em.iloc[0]).replace("%", "")) if len(_em) else None
                    margin_check = psel.margin_profile_check(subject_margin, peer_margins)
                    margin_check["peer_margins_pct"] = peer_margins
                except Exception as e:
                    print(f"Warning: margin-profile check skipped: {e}")
            peer_selection_log["margin_profile"] = margin_check
            if margin_check and margin_check.get("warning"):
                peer_selection_log["warnings"].append(margin_check["warning"])
            print("\n" + "\n".join(psel.summary_lines(peer_selection_log, margin_check)))
            with open(os.path.join(output_dir, "peer_selection.json"), "w", encoding="utf-8") as _fh:
                json.dump(peer_selection_log, _fh, indent=2, default=str)
    elif fmp_api_key and not args.peer_tickers and not args.no_auto_peers:
        args.peer_tickers = get_fmp_peers(args.company_ticker, fmp_api_key, max_peers=args.max_peers)
        if args.peer_tickers:
            print(f"\nAuto-discovered FMP peers: {args.peer_tickers}")
    if use_finedge and args.peer_tickers:
        print(f"\nFetching FinEdge data for peer comparison: {args.peer_tickers}")
        all_tickers = [args.company_ticker] + [t for t in args.peer_tickers if t != args.company_ticker]
        if is_bank:
            df_bank_peers, skipped = combine_peer_bank_data_finedge(all_tickers, finedge_client)
            covered = [t for t in all_tickers if t not in skipped and t != args.company_ticker]
            if covered and df_bank_peers is not None and not df_bank_peers.empty:
                path = os.path.join(output_dir, "peer_bank_comparison.csv")
                df_bank_peers.to_csv(path, index=False)
                print(f"Saved peer bank comparison to: {path}")
            else:
                df_bank_peers = None
        else:
            df_ebitda_peers, df_ev_ebitda_peers, skipped = combine_peer_financial_data_finedge(
                all_tickers, finedge_client, years_limit=args.years_limit)
            covered = [t for t in all_tickers if t not in skipped and t != args.company_ticker]
            if covered and not df_ebitda_peers.empty:
                projected_peer_ebitda = project_ebitda_for_peers(df_ebitda_peers, num_projection_years=1)
                peer_ebitda_path = os.path.join(output_dir, "peer_ebitda_comparison.csv")
                projected_peer_ebitda.to_csv(peer_ebitda_path)
                print(f"Saved peer EBITDA comparison to: {peer_ebitda_path}")
            if covered:
                # Same peers on today's prices: current market cap + latest net debt over FY and TTM EBITDA
                df_peer_current = get_finedge_peer_current_multiples(all_tickers, finedge_client)
                if df_peer_current is not None and not df_peer_current.empty:
                    df_peer_current.to_csv(os.path.join(output_dir, "peer_current_multiples.csv"))
                    print(f"Saved peer current-price multiples to: {os.path.join(output_dir, 'peer_current_multiples.csv')}")
                else:
                    df_peer_current = None
            if covered and not df_ev_ebitda_peers.empty:
                peer_ev_ebitda_path = os.path.join(output_dir, "peer_ev_ebitda_comparison.csv")
                df_ev_ebitda_peers.to_csv(peer_ev_ebitda_path)
                print(f"Saved peer EV/EBITDA comparison to: {peer_ev_ebitda_path}")
            else:
                df_ev_ebitda_peers = None
        if skipped:
            data_meta.setdefault("warnings", []).append(
                f"Peer data not available on the current FinEdge plan for: {', '.join(skipped)}.")
            print(f"⚠️  Peers skipped (FinEdge plan): {skipped}")
    elif use_finedge:
        print("Skipping peer comparison (no FinEdge peers found)")
    elif args.peer_tickers and fmp_api_key:
        print(f"\nFetching data for peer comparison: {args.peer_tickers}")
        all_tickers = args.peer_tickers + [args.company_ticker]
        
        try:
            df_ebitda_peers, df_ev_ebitda_peers = combine_peer_financial_data(
                all_tickers, fmp_api_key, years_limit=args.years_limit
            )
            
            if df_ebitda_peers is not None and not df_ebitda_peers.empty:
                projected_peer_ebitda = project_ebitda_for_peers(df_ebitda_peers, num_projection_years=1)
                peer_ebitda_path = os.path.join(output_dir, "peer_ebitda_comparison.csv")
                projected_peer_ebitda.to_csv(peer_ebitda_path)
                print(f"Saved peer EBITDA comparison to: {peer_ebitda_path}")
            else:
                print("Warning: No peer EBITDA data could be retrieved")
            
            if df_ev_ebitda_peers is not None and not df_ev_ebitda_peers.empty:
                peer_ev_ebitda_path = os.path.join(output_dir, "peer_ev_ebitda_comparison.csv")
                df_ev_ebitda_peers.to_csv(peer_ev_ebitda_path)
                print(f"Saved peer EV/EBITDA comparison to: {peer_ev_ebitda_path}")
            else:
                print("Warning: No peer EV/EBITDA data could be retrieved")
                
        except Exception as e:
            print(f"Error fetching peer data: {e}")
            print("Continuing without peer comparison...")
    else:
        print("Skipping peer comparison (no peer tickers provided)")

    # 4.5 Fetch Company News
    company_news = None
    enhanced_news_data = None
    if use_finedge:
        print(f"\nFetching exchange announcements and news headlines for {args.company_ticker} (last {args.news_days_back} days)...")
        filings = get_finedge_company_news(args.company_ticker, finedge_client,
                                           days_back=args.news_days_back, limit=args.news_limit) or []
        headlines = get_google_news(args.company_name, args.company_ticker, days_back=args.news_days_back, limit=15)
        print(f"Found {len(filings)} exchange announcement(s) and {len(headlines)} news headline(s)")
        company_news = sorted(filings + headlines, key=lambda a: a.get('publishedDate', ''), reverse=True)[:args.news_limit]
        if company_news and llm_settings is not None:
            n_ai = annotate_news_with_llm(company_news, llm_settings, args.company_name, usage_sink=llm_usage)
            if n_ai:
                print(f"AI classified sentiment and impact of {n_ai} news item(s)")
        if company_news:
            if args.enable_enhanced_news:
                enhanced_news_data = build_enhanced_news(args.company_ticker, company_news, args.news_days_back,
                                                         company_name=args.company_name)
                with open(os.path.join(output_dir, "enhanced_news.json"), 'w', encoding='utf-8') as f:
                    json.dump(enhanced_news_data, f, indent=2, ensure_ascii=False, default=str)
                with open(os.path.join(output_dir, "news_summary.md"), 'w', encoding='utf-8') as f:
                    f.write(enhanced_news_data.get('summary', ''))
                print("Saved enhanced news (exchange announcements and headlines)")
            news_output_path = os.path.join(output_dir, "company_news.json")
            with open(news_output_path, 'w', encoding='utf-8') as f:
                json.dump(company_news, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(company_news)} news item(s) to: {news_output_path}")
        else:
            print("Warning: No exchange announcements or news headlines found")
    elif fmp_api_key:
        print(f"\nFetching company news for {args.company_ticker}...")
        try:
            if args.enable_enhanced_news:
                # 使用增强版新闻获取
                enhanced_news_data = get_enhanced_company_news(
                    ticker=args.company_ticker,
                    api_key=fmp_api_key,
                    days_back=args.news_days_back,
                    limit=args.news_limit
                )
                company_news = enhanced_news_data.get('articles', [])
                
                # 保存增强版新闻数据
                enhanced_news_path = os.path.join(output_dir, "enhanced_news.json")
                with open(enhanced_news_path, 'w', encoding='utf-8') as f:
                    json.dump(enhanced_news_data, f, indent=2, ensure_ascii=False)
                print(f"Saved enhanced news data to: {enhanced_news_path}")
                
                # 保存新闻摘要
                news_summary_path = os.path.join(output_dir, "news_summary.md")
                with open(news_summary_path, 'w', encoding='utf-8') as f:
                    f.write(enhanced_news_data.get('summary', ''))
                print(f"Saved news summary to: {news_summary_path}")
            else:
                # 使用原始新闻获取
                company_news = get_company_news(
                    ticker=args.company_ticker,
                    api_key=fmp_api_key,
                    days_back=args.news_days_back,
                    limit=args.news_limit
                )
            
            if company_news:
                news_output_path = os.path.join(output_dir, "company_news.json")
                with open(news_output_path, 'w', encoding='utf-8') as f:
                    json.dump(company_news, f, indent=2, ensure_ascii=False)
                print(f"Saved company news to: {news_output_path}")
            else:
                print("Warning: No news data could be retrieved")
        except Exception as e:
            print(f"Error fetching company news: {e}")
            print("Continuing without news data...")
    else:
        print("Skipping news fetch (no FMP API key)")

    # 4.55 Fetch Retail Sentiment Insights
    retail_sentiment_data = None
    if adanos_api_key:
        print(f"\nFetching retail sentiment insights for {args.company_ticker}...")
        try:
            retail_client = RetailSentimentClient(
                api_key=adanos_api_key,
                base_url=adanos_base_url,
            )
            retail_sentiment_data = retail_client.get_snapshot(
                args.company_ticker,
                days_back=args.retail_sentiment_days_back,
            )

            retail_sentiment_path = os.path.join(output_dir, "retail_sentiment.json")
            with open(retail_sentiment_path, "w", encoding="utf-8") as f:
                json.dump(retail_sentiment_data, f, indent=2, ensure_ascii=False)
            print(f"Saved retail sentiment insights to: {retail_sentiment_path}")
        except Exception as e:
            print(f"Error fetching retail sentiment insights: {e}")
            print("Continuing without retail sentiment insights...")
    else:
        print("Skipping retail sentiment insights (no ADANOS_API_KEY / adanos_api_key configured)")

    # 4.6 敏感性分析
    sensitivity_results = None
    if args.enable_sensitivity_analysis:
        print(f"\nPerforming sensitivity analysis...")
        try:
            sensitivity_analyzer = SensitivityAnalyzer(final_data_df)
            
            # 收入敏感性分析
            revenue_sensitivity = sensitivity_analyzer.analyze_revenue_sensitivity()
            
            # 利润率敏感性分析
            margin_sensitivity = sensitivity_analyzer.analyze_margin_sensitivity()
            
            # 综合敏感性表格
            combined_sensitivity = sensitivity_analyzer.generate_sensitivity_table()
            
            # 置信区间
            revenue_ci = sensitivity_analyzer.calculate_confidence_interval('Revenue')
            ebitda_ci = sensitivity_analyzer.calculate_confidence_interval('EBITDA')
            
            sensitivity_results = {
                'revenue_sensitivity': revenue_sensitivity.to_dict() if not revenue_sensitivity.empty else {},
                'margin_sensitivity': margin_sensitivity.to_dict() if not margin_sensitivity.empty else {},
                'combined_sensitivity': combined_sensitivity.to_dict() if not combined_sensitivity.empty else {},
                'confidence_intervals': sensitivity_analyzer.confidence_intervals,
                'summary': sensitivity_analyzer.generate_sensitivity_summary()
            }
            
            # 保存敏感性分析结果
            sensitivity_path = os.path.join(output_dir, "sensitivity_analysis.json")
            with open(sensitivity_path, 'w', encoding='utf-8') as f:
                json.dump(sensitivity_results, f, indent=2, default=str)
            print(f"Saved sensitivity analysis to: {sensitivity_path}")
            
            # 保存敏感性摘要
            sensitivity_summary_path = os.path.join(output_dir, "sensitivity_summary.md")
            with open(sensitivity_summary_path, 'w', encoding='utf-8') as f:
                f.write(sensitivity_results['summary'])
            print(f"Saved sensitivity summary to: {sensitivity_summary_path}")
            
        except Exception as e:
            print(f"Error performing sensitivity analysis: {e}")
            print("Continuing without sensitivity analysis...")

    # 4.7 催化剂分析
    catalyst_results = None
    if args.enable_catalyst_analysis and company_news:
        print(f"\nPerforming catalyst analysis...")
        try:
            catalyst_analyzer = CatalystAnalyzer(args.company_ticker, fmp_api_key, company_name=args.company_name)
            
            # 识别催化剂
            catalysts = catalyst_analyzer.identify_catalysts(company_news)
            
            # 分类催化剂
            categorized_catalysts = catalyst_analyzer.categorize_catalysts()
            
            # 获取顶级催化剂
            top_catalysts = catalyst_analyzer.get_top_catalysts(5)
            
            # 生成摘要
            catalyst_summary = catalyst_analyzer.generate_catalyst_summary()
            
            catalyst_results = {
                'catalysts': [
                    {
                        'event_type': c.event_type,
                        'description': c.description,
                        'expected_date': c.expected_date,
                        'impact_level': c.impact_level,
                        'probability': c.probability,
                        'sentiment': c.sentiment
                    }
                    for c in catalysts
                ],
                'categorized': {
                    k: [{'description': c.description, 'impact': c.impact_level} for c in v]
                    for k, v in categorized_catalysts.items()
                },
                'top_catalysts': top_catalysts,
                'summary': catalyst_summary
            }
            
            # 保存催化剂分析结果
            catalyst_path = os.path.join(output_dir, "catalyst_analysis.json")
            with open(catalyst_path, 'w', encoding='utf-8') as f:
                json.dump(catalyst_results, f, indent=2, default=str)
            print(f"Saved catalyst analysis to: {catalyst_path}")
            
            # 保存催化剂摘要
            catalyst_summary_path = os.path.join(output_dir, "catalyst_summary.md")
            with open(catalyst_summary_path, 'w', encoding='utf-8') as f:
                f.write(catalyst_summary)
            print(f"Saved catalyst summary to: {catalyst_summary_path}")
            
        except Exception as e:
            print(f"Error performing catalyst analysis: {e}")
            print("Continuing without catalyst analysis...")

    # 5. Text Generation (Unified Logic)
    narrative_unit_warnings = []
    report_ctx = None
    # Canonical context: one unit, one set of facts, checks and reconciliation inputs for the whole report
    if use_finedge and not is_bank:
        try:
            ttm_balance = get_finedge_ttm_and_latest_balance(args.company_ticker, finedge_client,
                                                             "s" if data_meta.get("statement_type") == "standalone" else "c")
            technicals = get_finedge_technical_indicators(args.company_ticker, finedge_client)
            extras = {"fact_blocks": []}
            try:
                st_code = "s" if data_meta.get("statement_type") == "standalone" else "c"
                price_df = get_finedge_price_history(args.company_ticker, finedge_client, years=2)
                lf_eps = [rctx._f(final_data_df.loc[final_data_df["metrics"] == "EPS", c].iloc[0]) for c in sorted(
                    c for c in final_data_df.columns if isinstance(c, str) and c.endswith("A"))]
                eps_info = {"eps_fy_now": lf_eps[-1], "eps_fy_prev": lf_eps[-2] if len(lf_eps) > 1 else None,
                            "eps_ttm": (ttm_balance.get("ttm") or {}).get("eps")}
                tf = tech_an.compute_technical_facts(price_df, eps_info)
                tf_json = {k: v for k, v in tf.items() if not k.startswith("_")}
                extras["technical"] = tf_json
                if tf.get("available", True):
                    extras["fact_blocks"].append(rctx.technical_block(tf_json))
            except Exception as e:
                print(f"Warning: technical analysis skipped: {e}")
            try:
                half_cf = normalize_cash_flow(finedge_client.financials(args.company_ticker, "cf", "halfyearly", st_code), "halfyearly")
                ttm_cf = cf_an.derive_ttm_cashflow(financial_data.get("cash_flow"), half_cf)
                cfacts = cf_an.compute_cashflow_facts(financial_data.get("income_statement"), financial_data.get("balance_sheet"),
                                                      financial_data.get("cash_flow"), ttm=ttm_balance.get("ttm"), ttm_cashflow=ttm_cf)
                extras["cashflow"] = cfacts
                extras["fact_blocks"].append("CASH FLOW DIAGNOSTICS (Cr, computed):\n  " + "\n  ".join(cf_an.cashflow_text_lines(cfacts)))
            except Exception as e:
                print(f"Warning: cash-flow analysis skipped: {e}")
            try:
                peer_medians = {}
                if args.peer_tickers:
                    pr, pi = get_finedge_peer_frames(args.peer_tickers, finedge_client)
                    peer_medians = rbars.peer_medians_from_ratios(pr, normalize_symbol(args.company_ticker), pi)
                rb = rbars.compute_ratio_bar_facts(financial_data.get("ratios"), financial_data.get("key_metrics"),
                                                   peer_medians=peer_medians, income_df=financial_data.get("income_statement"))
                extras["ratio_bars"] = rb
                extras["fact_blocks"].append("RATIO YARDSTICKS (computed):\n  " + "\n  ".join(rbars.ratio_bar_text_lines(rb)))
            except Exception as e:
                print(f"Warning: ratio bars skipped: {e}")
            # Fiscal calendar and reporting basis: one basis for every figure, the other basis stated where it differs
            try:
                fiscal_issues = fiscal_guard.check_fiscal_year_ends({
                    "income": financial_data.get("income_statement"), "balance": financial_data.get("balance_sheet"),
                    "cash": financial_data.get("cash_flow")})
                basis = fiscal_guard.quarter_basis_comparison(finedge_client, args.company_ticker, st_code)
                extras["basis"] = {"quarter": basis, "fiscal_issues": fiscal_issues, "fiscal_year_rule": "1 April to 31 March"}
                latest_fy_year = int(str(latest_year)[:4]) if latest_year else None
                if basis:
                    extras["fact_blocks"].append(fiscal_guard.basis_fact_block(basis, latest_fy_year))
                for w in fiscal_guard.basis_warnings(basis, fiscal_issues):
                    print(f"Warning: {w}")
                    data_meta.setdefault("warnings", []).append(w)
            except Exception as e:
                print(f"Warning: basis check skipped: {e}")
            report_ctx = rctx.build_report_context(
                ticker=normalize_symbol(args.company_ticker), company_name=args.company_name,
                as_of=pd.Timestamp.now().strftime("%Y-%m-%d"), final_df=final_data_df,
                financial_data=financial_data, company_metrics=company_metrics, ttm_balance=ttm_balance,
                peer_ev_df=df_ev_ebitda_peers, peer_ebitda_actual=df_ebitda_peers,
                peer_ebitda_projected=projected_peer_ebitda, peer_current_df=df_peer_current,
                technicals=technicals, extras=extras)
            rctx.save_context(report_ctx, os.path.join(output_dir, "report_context.json"))
            # Sentiment gate: valuation outranks headline tone (implied price below 70% of price forces CONTRADICTORY/BEARISH)
            try:
                gate = sgate.apply_gate(enhanced_news_data, report_ctx)
                report_ctx["sentiment_gate"] = gate
                tone = ((enhanced_news_data or {}).get("sentiment_summary") or {}).get("news_tone")
                gb = sgate.gate_fact_block(gate, tone)
                if gb:
                    print("Sentiment gate triggered: " + sgate.gate_note(gate))
                    report_ctx["prompt_block"] = report_ctx["prompt_block"].replace(rctx.FACTS_END, gb + "\n" + rctx.FACTS_END)
                # Peer-set limits: an unreliable peer set must not be presented as a valuation benchmark
                if peer_selection_log:
                    report_ctx["peer_selection"] = peer_selection_log
                    pf = psel.peer_fact_block(peer_selection_log)
                    if pf:
                        report_ctx["prompt_block"] = report_ctx["prompt_block"].replace(rctx.FACTS_END, pf + "\n" + rctx.FACTS_END)
                    for w in psel.peer_warnings(peer_selection_log):
                        data_meta.setdefault("warnings", []).append(w)
                if enhanced_news_data:
                    with open(os.path.join(output_dir, "enhanced_news.json"), 'w', encoding='utf-8') as f:
                        json.dump(enhanced_news_data, f, indent=2, ensure_ascii=False, default=str)
                rctx.save_context(report_ctx, os.path.join(output_dir, "report_context.json"))
            except Exception as e:
                print(f"Warning: sentiment gate skipped: {e}")
            bad = rctx.failed_checks(report_ctx)
            for c in bad:
                print(f"⚠️  Consistency check FAILED: {c['name']} (expected {c['expected']}, got {c['actual']})")
            print(f"Consistency checks: {sum(1 for c in report_ctx['checks'] if c['ok'])} passed, {len(bad)} failed, "
                  f"{sum(1 for c in report_ctx['checks'] if c['ok'] is None)} skipped")
        except Exception as e:
            print(f"Warning: could not build the canonical report context ({e}); sections will use raw tables")
            report_ctx = None

    # Bank context for the visual report formats (a separate, bank-shaped report_context.json; the text pipeline does not use it)
    if use_finedge and is_bank:
        try:
            from report_visual.bank import context as bank_ctx
            st_code = "s" if data_meta.get("statement_type") == "standalone" else "c"
            ttm_b = bank_ctx.fetch_bank_ttm(finedge_client, args.company_ticker, st_code)
            eq_b = bank_ctx.fetch_latest_equity(finedge_client, args.company_ticker, st_code)
            peer_cur = {}
            if df_bank_peers is not None and args.peer_tickers:
                peer_cur = bank_ctx.fetch_peer_current_pb(
                    finedge_client, [normalize_symbol(args.company_ticker)] + [normalize_symbol(t) for t in args.peer_tickers], st_code)
            tech_b = None
            try:
                price_b = get_finedge_price_history(args.company_ticker, finedge_client, years=2)
                eps_cols = sorted(c for c in final_data_df.columns if isinstance(c, str) and c.endswith("A"))
                eps_vals = [rctx._f(final_data_df.loc[final_data_df["metrics"] == "EPS", c].iloc[0]) for c in eps_cols]
                tf_b = tech_an.compute_technical_facts(price_b, {"eps_fy_now": eps_vals[-1], "eps_fy_prev": eps_vals[-2] if len(eps_vals) > 1 else None,
                                                                 "eps_ttm": ttm_b["eps"] if ttm_b else None})
                tech_b = {k: v for k, v in tf_b.items() if not k.startswith("_")}
            except Exception as e:
                print(f"Warning: technical analysis skipped for the bank context: {e}")
            bctx = bank_ctx.build_bank_context(
                ticker=normalize_symbol(args.company_ticker), company_name=args.company_name,
                as_of=pd.Timestamp.now().strftime("%Y-%m-%d"), final_df=final_data_df, financial_data=financial_data,
                company_metrics=company_metrics, ttm=ttm_b, latest_equity=eq_b, peer_bank_df=df_bank_peers, peer_current=peer_cur,
                technical_facts=tech_b, warnings=data_meta.get("warnings", []))
            rctx.save_context(bctx, os.path.join(output_dir, "report_context.json"))
            print("Saved the bank report context for the visual formats")
        except Exception as e:
            print(f"Warning: could not build the bank report context ({e}); the visual formats are unavailable for this run")

    if args.generate_text_sections:
        print("\nGenerating AI-powered text sections...")
        
        if llm_settings is None or not llm_settings.api_key:
            provider = llm_settings.provider if llm_settings else (args.llm_provider or "openai")
            print(f"Error: no API key for the '{provider}' LLM provider "
                  f"(set {provider}_api_key in config.ini or the provider's environment variable). "
                  "Skipping text generation.")
        else:
            data_notes = list(data_meta.get("warnings", []))
            if use_finedge:
                data_notes.insert(0, "Amounts are in Indian rupees (absolute rupees; 1 crore = 10,000,000). "
                                     "Fiscal years end in March: 2026A is FY2025-26.")
            data_for_text_gen = {
                "company_description": (company_metrics or {}).get("description"),
                "data_notes": data_notes,
                "is_bank": is_bank,
                "financial_metrics": final_data_df,
                "peer_ebitda": projected_peer_ebitda,
                "peer_ev_ebitda": df_ev_ebitda_peers,
                "company_news": company_news,
                "enhanced_news": enhanced_news_data,
                "retail_sentiment": retail_sentiment_data,
                "sensitivity_analysis": sensitivity_results,
                "catalyst_analysis": catalyst_results
            }
            
            # A single list for all text types to be generated (including news_summary)
            all_text_types = [
                "tagline", "company_overview", "investment_overview", 
                "valuation_overview", "risks", "competitor_analysis", 
                "major_takeaways", "news_summary"  # NEW
            ]

            if report_ctx is not None:
                # One canonical unit and one set of facts for every section, proven before any model call
                data_for_text_gen = rctx.build_text_data(report_ctx, final_data_df, df_ebitda_peers,
                                                         df_ev_ebitda_peers, base=data_for_text_gen)
                rctx.assert_section_contexts_identical({
                    t: _prepare_user_prompt(data_for_text_gen, t, args.company_name, args.company_ticker)
                    for t in all_text_types})
                print("Section contexts verified identical (same facts block and crore table in every prompt)")
            
            # A single loop calling the unified generation function
            for text_type in all_text_types:
                # Skip news_summary if no news data available
                if text_type == "news_summary" and not company_news:
                    print(f"Skipping 'news_summary' - no news data available")
                    # Create placeholder file
                    fallback_text = f"No recent news available for {args.company_name} ({args.company_ticker})."
                    file_path = os.path.join(text_output_dir, f"{text_type}.txt")
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(fallback_text)
                    print(f"Created placeholder for '{text_type}' at {file_path}")
                    continue
                
                print(f"Generating '{text_type}' for {args.company_name} ({args.company_ticker})...")
                try:
                    # Call the single, unified function for all types
                    generated_text = generate_text_section(
                        data_for_text_gen, 
                        text_type, 
                        llm_settings.api_key, 
                        args.company_name, 
                        args.company_ticker,
                        settings=llm_settings,
                        failures=failed_text_sections,
                        usage_sink=llm_usage,
                    )
                    
                    # Fallback validation can remain here as a safety net
                    if text_type == "competitor_analysis" and (not generated_text or len(generated_text.split('.')) < 3):
                         print(f"⚠️ Warning: Competitor analysis seems too short, using fallback.")
                         failed_text_sections.append(text_type)  # shipping placeholder text: report it
                         generated_text = f"{args.company_name} demonstrates competitive positioning within its industry sector through consistent financial performance and strategic market positioning relative to key competitors."
                    
                    elif text_type == "major_takeaways" and is_bank and "Net Interest" not in generated_text:
                         print(f"⚠️ Warning: Major takeaways missing bank sections, using fallback.")
                         failed_text_sections.append(text_type)  # shipping placeholder text: report it
                         generated_text = (f"Net Interest Income Growth: {args.company_name}'s net interest income trend is shown in the financial summary.\n\n"
                                           f"Net Interest Margin: {args.company_name}'s NIM history is shown in the bank metrics table.\n\n"
                                           f"Asset Quality: {args.company_name}'s gross and net NPA ratios and credit cost are shown in the bank metrics table.\n\n"
                                           f"Profitability: {args.company_name}'s ROA, ROE and cost-to-income ratio are shown in the bank metrics table.")

                    elif text_type == "major_takeaways" and not is_bank and "Revenue Growth:" not in generated_text:
                         print(f"⚠️ Warning: Major takeaways missing required sections, using fallback.")
                         failed_text_sections.append(text_type)  # shipping placeholder text: report it
                         generated_text = f"Revenue Growth: {args.company_name}'s revenue growth shows consistent performance trends.\n\nGross Profit Margin: {args.company_name}'s gross profit margins demonstrate operational effectiveness.\n\nSG&A Expense Margin: {args.company_name}'s SG&A expense management shows disciplined cost control.\n\nEBITDA Margin Stability: {args.company_name}'s EBITDA margin stability reflects strong underlying fundamentals."

                    elif text_type == "news_summary" and (not generated_text or len(generated_text.split()) < 50):
                        print(f"⚠️ Warning: News summary seems too short, using fallback.")
                        failed_text_sections.append(text_type)  # shipping placeholder text: report it
                        generated_text = f"Recent news coverage for {args.company_name} reflects ongoing market interest and developments in the company's operations and strategic initiatives."

                    if report_ctx is not None:
                        for w in rctx.lint_narrative_units(generated_text, report_ctx):
                            narrative_unit_warnings.append(f"{text_type}: {w}")
                            print(f"⚠️  Unit check ({text_type}): {w}")

                    file_path = os.path.join(text_output_dir, f"{text_type}.txt")
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(generated_text)
                    print(f"✅ Successfully generated and saved '{text_type}' to {file_path}")
                    
                except Exception as e:
                    print(f"Error generating text for '{text_type}': {e}")
                    failed_text_sections.append(text_type)
                    # Create a fallback file if generation fails
                    fallback_text = f"{args.company_name} ({args.company_ticker}) {text_type.replace('_', ' ')} analysis not available."
                    file_path = os.path.join(text_output_dir, f"{text_type}.txt")
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(fallback_text)
                    print(f"Created fallback text for '{text_type}' at {file_path}")
        if llm_usage:
            print("\nAI cost per section (estimated from API usage):")
            for u in llm_usage:
                c = u["cost_usd"]
                think = u["thinking_tokens"]
                print(f"  {u['section']:<20} {u['model']:<18} in {u['input_tokens']:>6}  out {u['output_tokens']:>6}"
                      f"{'' if think is None else f' (thinking {think})'}  "
                      f"{'n/a' if c is None else f'${c:.4f}'}")
            print(f"  {'TOTAL':<20} {'':<18} ${sum(u['cost_usd'] or 0 for u in llm_usage):.4f}")
    else:
        print("Skipping text generation (not enabled)")

    # 6. Save additional financial statement data for reference
    print("\nSaving additional financial statement data...")
    
    for statement_name, df in financial_data.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            statement_path = os.path.join(output_dir, f"{statement_name}_raw_data.csv")
            df.to_csv(statement_path, index=False)
            print(f"Saved {statement_name} to: {statement_path} ({len(df)} rows)")

    # 6.5 Save price history (used by the enhanced stock-price and technical charts)
    if use_finedge:
        try:
            price_df = get_finedge_price_history(args.company_ticker, finedge_client, years=2)
            if not price_df.empty:
                price_df.to_csv(os.path.join(output_dir, "historical_price_full.csv"), index=False)
                print(f"Saved price history ({len(price_df)} days) to: {os.path.join(output_dir, 'historical_price_full.csv')}")
        except FinEdgeError as e:
            print(f"Warning: could not save price history: {e}")

    # 6.6 Reconciliation log (price, date, shares, net debt and EPS basis behind every multiple)
    if report_ctx is not None:
        try:
            rctx.write_reconciliation_log(report_ctx, os.path.join(output_dir, "reconciliation_log.md"),
                                          extra_notes=[f"Narrative unit check: {w}" for w in narrative_unit_warnings] or None)
            print(f"Saved reconciliation log to: {os.path.join(output_dir, 'reconciliation_log.md')}")
        except Exception as e:
            print(f"Warning: could not write the reconciliation log: {e}")

    # 7. Create summary report
    all_text_types = ["tagline", "company_overview", "investment_overview", "valuation_overview", "risks", "competitor_analysis", "major_takeaways", "news_summary"]
    
    summary_data = {
        "company_ticker": args.company_ticker,
        "company_name": args.company_name,
        "analysis_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": data_meta.get("data_source", "Financial Modeling Prep API"),
        "data_provider": args.data_provider,
        "currency": data_meta.get("currency", "USD"),
        "statement_type": data_meta.get("statement_type"),
        "template": "bank" if is_bank else "company",
        "llm_provider": llm_settings.provider if llm_settings else None,
        "llm_model": (None if llm_settings.tiered else llm_settings.model_name) if llm_settings else None,  # None = tiered; the report step re-resolves it
        "llm_tiered": bool(llm_settings and llm_settings.tiered),
        "data_warnings": data_meta.get("warnings", []),
        "data_period": args.period,
        "years_analyzed": len(actual_years),
        "available_years": available_years,
        "latest_year": latest_year,
        "forecast_years": list(forecast_config["revenue_growth_assumptions"].keys()),
        "peer_tickers": args.peer_tickers,
        "news_days_back": args.news_days_back,
        "news_articles_fetched": len(company_news) if company_news else 0,
        "enhanced_news_enabled": args.enable_enhanced_news,
        "sensitivity_analysis_enabled": args.enable_sensitivity_analysis,
        "catalyst_analysis_enabled": args.enable_catalyst_analysis,
        "forecast_config": forecast_config,
        "llm_failed_sections": failed_text_sections,
        "consistency_failures": [c["name"] for c in rctx.failed_checks(report_ctx)] if report_ctx is not None else None,
        "narrative_unit_warnings": narrative_unit_warnings,
        "llm_usage": llm_usage,
        "llm_cost_usd": round(sum(u["cost_usd"] or 0 for u in llm_usage), 4),
        "files_generated": {
            "main_analysis": args.output_csv_name,
            "peer_ebitda": "peer_ebitda_comparison.csv" if projected_peer_ebitda is not None else None,
            "peer_ev_ebitda": "peer_ev_ebitda_comparison.csv" if df_ev_ebitda_peers is not None else None,
            "peer_bank": "peer_bank_comparison.csv" if df_bank_peers is not None else None,
            "company_news": "company_news.json" if company_news else None,
            "enhanced_news": "enhanced_news.json" if enhanced_news_data else None,
            "sensitivity_analysis": "sensitivity_analysis.json" if sensitivity_results else None,
            "catalyst_analysis": "catalyst_analysis.json" if catalyst_results else None,
            "text_sections": all_text_types if args.generate_text_sections else []
        }
    }
    
    summary_path = os.path.join(output_dir, "analysis_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2, default=str)
    print(f"Saved analysis summary to: {summary_path}")

    # 8. Print final summary and next steps
    print(f"\n" + "="*60)
    print(f"✅ Financial analysis completed successfully!")
    print(f"📁 All outputs saved to: {output_dir}")
    print(f"📊 Main analysis file: {output_csv_path}")
    if args.generate_text_sections:
        print(f"📝 AI text sections generated and saved in: {text_output_dir}")
    
    print(f"\n🚀 Ready to create equity report using:")
    print(f"python create_equity_report.py \\")
    if use_finedge:
        print(f"  --data-provider finedge \\")
    print(f"  --company-ticker {args.company_ticker} \\")
    print(f"  --company-name \"{args.company_name}\" \\")
    print(f"  --analysis-csv {output_csv_path} \\")
    print(f"  --ratios-csv {os.path.join(output_dir, 'ratios_raw_data.csv')} \\")
    
    if args.peer_tickers and projected_peer_ebitda is not None:
        print(f"  --peer-ebitda-csv {os.path.join(output_dir, 'peer_ebitda_comparison.csv')} \\")
    if args.peer_tickers and df_ev_ebitda_peers is not None:
        print(f"  --peer-ev-ebitda-csv {os.path.join(output_dir, 'peer_ev_ebitda_comparison.csv')} \\")
    
    if args.generate_text_sections:
        for i, text_type in enumerate(all_text_types):
            param_name = text_type.replace('_', '-')
            file_path = os.path.join(text_output_dir, f'{text_type}.txt')
            # Add backslash for line continuation, except for the last item
            continuation = " \\" if i < len(all_text_types) - 1 else ""
            print(f"  --{param_name}-file {file_path}{continuation}")
    
    print(f"\n🎯 All files ready for report generation!")

    if failed_text_sections:
        print("\n" + "!"*60)
        print(f"❌ {len(failed_text_sections)} AI text section(s) FAILED and contain placeholder text: "
              f"{', '.join(failed_text_sections)}")
        print("   Check the LLM API key, credit balance and model availability above, then re-run.")
        print("!"*60)
        if not args.allow_llm_fallback:
            sys.exit(2)


if __name__ == "__main__":
    main()
