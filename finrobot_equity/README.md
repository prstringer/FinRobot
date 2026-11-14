# FinRobot Equity Research Module

AI-powered equity research report generator. Fetches financial data, runs LLM-based analysis, and produces professional multi-page HTML/PDF reports — all with a single command.

## Architecture

```
finrobot_equity/
├── core/                              # Analysis engine
│   ├── config/
│   │   └── config.ini.template        # API key configuration template
│   ├── assets/                        # Static assets (logos for PDF reports)
│   ├── src/
│   │   ├── generate_financial_analysis.py   # Step 1: Data fetching & analysis
│   │   ├── create_equity_report.py          # Step 2: HTML report generation
│   │   ├── generate_pdf_report.py           # Step 3: PDF report generation (optional)
│   │   ├── Run.ipynb                        # Jupyter notebook demo
│   │   └── modules/                         # Core modules
│   │       ├── common_utils.py              #   Config & API key management
│   │       ├── market_data_api.py           #   FMP API client
│   │       ├── finedge_api.py               #   FinEdge API adapter (Indian stocks)
│   │       ├── currency.py                  #   Report currency (USD / INR) formatting
│   │       ├── financial_data_processor.py  #   Metrics extraction & forecasting
│   │       ├── text_generator_agents.py     #   LLM text generation orchestrator
│   │       ├── llm_providers.py             #   OpenAI / Anthropic / Gemini switch
│   │       ├── chart_generator.py           #   Financial chart rendering
│   │       ├── enhanced_chart_generator.py  #   Advanced chart configurations
│   │       ├── html_renderer.py             #   HTML report renderer
│   │       ├── html_template_professional.py#   HTML template definitions
│   │       ├── report_data_loader.py        #   Data loading utilities
│   │       ├── report_structure.py          #   Report layout management
│   │       ├── enhanced_text_generator.py   #   Advanced text processing
│   │       ├── sensitivity_analyzer.py      #   Sensitivity analysis
│   │       ├── catalyst_analyzer.py         #   Market catalyst identification
│   │       ├── news_integrator.py           #   News data integration
│   │       ├── retail_sentiment_client.py   #   Optional retail sentiment insights
│   │       ├── valuation_engine.py          #   Valuation modeling
│   │       ├── pdf_generator.py             #   PDF report generation
│   │       ├── professional_pdf_report.py   #   Professional PDF templates
│   │       └── equity_agents/               #   AI agents for report sections
│   │           ├── agent_manager.py         #     Agent orchestration
│   │           ├── tagline_agent.py         #     Executive tagline
│   │           ├── company_overview_agent.py#     Company overview
│   │           ├── investment_overview_agent.py  # Investment thesis
│   │           ├── valuation_overview_agent.py   # Valuation analysis
│   │           ├── risks_agent.py                # Risk assessment
│   │           ├── competitor_analysis_agent.py  # Competitive landscape
│   │           ├── major_takeaways_agent.py      # Key takeaways
│   │           └── news_summary_agent.py         # News summarization
│   └── tests/                         # Unit tests
│       ├── test_generate_report.py
│       └── test_modules.py
│
└── web_app/                           # Web interface (FastAPI)
    ├── main.py                        # Application entry point & API routes
    ├── auth.py                        # Authentication (local + GitHub OAuth)
    ├── admin_routes.py                # Admin API endpoints
    ├── database/                      # Database layer (SQLAlchemy + SQLite)
    │   ├── connection.py
    │   ├── models.py
    │   └── crud.py
    ├── middleware/
    │   └── request_logger.py          # Request logging
    ├── templates/                     # Jinja2 HTML templates
    ├── static/                        # CSS & images
    └── data/                          # Runtime data (auto-created, gitignored)
```

## Pipeline

The report generation follows a two-step pipeline:

```
┌─────────────────────────────────┐     ┌──────────────────────────────┐
│  generate_financial_analysis.py │     │  create_equity_report.py     │
│                                 │     │                              │
│  1. Fetch data from FMP API     │────>│  1. Load analysis outputs    │
│  2. Process financial metrics   │     │  2. Auto-fetch market data   │
│  3. Generate 3-year forecasts   │     │  3. Generate charts          │
│  4. Run peer comparison         │     │  4. Render HTML report       │
│  5. AI text generation          │     │  5. Validate & regenerate    │
│                                 │     │                              │
│  Output: CSV + JSON + TXT       │     │  Output: 3-page HTML report  │
└─────────────────────────────────┘     └──────────────────────────────┘
```

## Quick Start

### 1. Configure API Keys

```bash
cp finrobot_equity/core/config/config.ini.template finrobot_equity/core/config/config.ini
```

Edit `config.ini` with your keys:

```ini
[API_KEYS]
fmp_api_key = YOUR_FMP_API_KEY          # https://financialmodelingprep.com/developer
openai_api_key = YOUR_OPENAI_API_KEY    # https://platform.openai.com/account/api-keys
adanos_api_key = YOUR_ADANOS_API_KEY    # Optional: enables Retail Sentiment Insights
```

If `adanos_api_key` is configured, the report pipeline adds an optional **Retail Sentiment Insights** layer to the equity research output. It supplements the existing news workflow with structured public-retail activity snapshots across Reddit, X.com, and Polymarket.

#### Choosing the LLM for the AI text sections

The text sections (overview, risks, takeaways, ...) can be written by **OpenAI**, **Anthropic Claude**
or **Google Gemini**. Pick one with `llm_provider` in `config.ini`, `--llm-provider openai|anthropic|gemini`
on either script, or the *AI Model* selector in the web app. Keys come from `openai_api_key`,
`anthropic_api_key` or `gemini_api_key` in `config.ini`, or from the `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY` or `GEMINI_API_KEY` environment variables. `llm_model` / `--llm-model` overrides
the default model (gpt-4o-mini, claude-opus-5-5, gemini-3.8-flash).

### 2. Deploy via Script

```bash
chmod +x deploy.sh
./deploy.sh start
```

Access the web interface at `http://127.0.0.1:8001`.

### 3. Or Run via Command Line

```bash
# Step 1: Financial analysis
python finrobot_equity/core/src/generate_financial_analysis.py \
    --company-ticker NVDA \
    --company-name "NVIDIA Corporation" \
    --config-file finrobot_equity/core/config/config.ini \
    --peer-tickers AMD INTC \
    --generate-text-sections

# Step 2: Generate report
python finrobot_equity/core/src/create_equity_report.py \
    --company-ticker NVDA \
    --company-name "NVIDIA Corporation" \
    --analysis-csv output/NVDA/analysis/financial_metrics_and_forecasts.csv \
    --ratios-csv output/NVDA/analysis/ratios_raw_data.csv \
    --tagline-file output/NVDA/analysis/tagline.txt \
    --company-overview-file output/NVDA/analysis/company_overview.txt \
    --investment-overview-file output/NVDA/analysis/investment_overview.txt \
    --valuation-overview-file output/NVDA/analysis/valuation_overview.txt \
    --risks-file output/NVDA/analysis/risks.txt \
    --competitor-analysis-file output/NVDA/analysis/competitor_analysis.txt \
    --major-takeaways-file output/NVDA/analysis/major_takeaways.txt \
    --peer-ev-ebitda-csv output/NVDA/analysis/peer_ev_ebitda_comparison.csv \
    --enable-text-regeneration \
    --config-file finrobot_equity/core/config/config.ini
```

### Indian Stocks (NSE/BSE) via FinEdge

`--data-provider finedge` pulls statements, ratios, prices, shareholding and exchange
announcements from the [FinEdge API](https://www.finedgeapi.com) and renders the report in
rupees (₹ crore). Set the token as `FINEDGE_API_TOKEN` or `finedge_api_token` in `config.ini`.

```bash
# Step 1: analysis (consolidated for companies, standalone for banks; sector peers found automatically)
python finrobot_equity/core/src/generate_financial_analysis.py \
    --data-provider finedge \
    --company-ticker RELIANCE \
    --company-name "Reliance Industries Ltd" \
    --enable-sensitivity-analysis --enable-enhanced-news

# Step 2: report (provider and currency are read from analysis_summary.json)
python finrobot_equity/core/src/create_equity_report.py \
    --company-ticker RELIANCE \
    --company-name "Reliance Industries Ltd" \
    --analysis-csv output/RELIANCE/analysis/financial_metrics_and_forecasts.csv \
    --ratios-csv output/RELIANCE/analysis/ratios_raw_data.csv \
    ...  # same text-file arguments as above
```

What differs from the FMP path:

- **Banks** (e.g. HDFCBANK) use a bank template: net interest income, pre-provision profit,
  provisions, NIM, cost-to-income, credit cost, GNPA/NNPA, CET1, advances and deposits, and a
  P/B + P/E valuation instead of EV/EBITDA and DCF.
- **Derived figures:** FinEdge has no EBITDA, gross profit or SG&A fields. EBITDA is revenue
  from operations minus operating expenses (excluding other income and exceptional items);
  SG&A is employee plus other expenses.
- **Not available from FinEdge:** analyst target price and rating (shown as N/A), news (exchange
  announcements are used instead) and forward P/E (trailing P/E is shown).
- **Peers:** the free plan only serves the sample symbols (ITC, RELIANCE, HDFCBANK, ICICIPRULI).
  Peer symbols outside the plan are skipped with a note in the report; peer tables fill in
  automatically on a plan that includes them.
- **Data notes** (e.g. ITC's excise-inclusive revenue from Feb 2026, one-off discontinued
  operations, merger years) are printed during step 1, passed to the AI text sections and shown
  in the report footer.

In the web app, choose **India · FinEdge** under *Market* (symbols such as `RELIANCE` or `ITC.NS`
are accepted). The PDF step uses the same data; it prints amounts as "Rs" because the built-in PDF
fonts have no ₹ glyph.

See [docs/finedge_data_quality.md](docs/finedge_data_quality.md) for the data-quality test behind
these choices. Forecast years are the three years after the latest actual year (e.g. 2027E–2029E
when the latest actual is FY2026); `--revenue-growth-2025/2026/2027` set years 1–3.

## Deployment Commands

| Command | Description |
|:---|:---|
| `./deploy.sh start` | Start the web application (auto-installs dependencies) |
| `./deploy.sh stop` | Stop the application |
| `./deploy.sh restart` | Restart the application |
| `./deploy.sh status` | Check running status and recent logs |
| `./deploy.sh install` | Install/update dependencies only |

### Environment Variables

| Variable | Default | Description |
|:---|:---|:---|
| `GITHUB_CLIENT_ID` | — | GitHub OAuth client ID |
| `GITHUB_CLIENT_SECRET` | — | GitHub OAuth client secret |
| `FINROBOT_ADMIN_EMAIL` | `admin@finrobot.com` | Initial admin email |
| `FINROBOT_ADMIN_PASSWORD` | (random) | Initial admin password |
| `WEB_HOST` | `127.0.0.1` | Server bind address |
| `WEB_PORT` | `8001` | Server port |

## API Dependencies

| Service | Required | Purpose |
|:---|:---|:---|
| [Financial Modeling Prep](https://financialmodelingprep.com/developer) | Yes | Financial data, market metrics, peer comparison |
| [FinEdge API](https://www.finedgeapi.com) | For Indian stocks | Financial data for NSE/BSE stocks (`--data-provider finedge`) |
| [OpenAI](https://platform.openai.com/), [Anthropic](https://platform.claude.com/) or [Google Gemini](https://aistudio.google.com/) | One of them, for AI text | AI-powered text generation for report sections (`llm_provider`) |
| Adanos Finance API | No | Optional retail sentiment insights for Reddit, X.com, and Polymarket |

## License

Apache 2.0 — See [LICENSE](../LICENSE) for details.
