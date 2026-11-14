# FinEdge API data quality: ITC, RELIANCE, HDFCBANK

Tested 2026-11-14 on the free plan. 129 calls in total. Raw responses (126 files) were saved outside the repo. The token was masked in every saved file. All money figures below are in ₹ crore (the API value divided by 1e7).

## Headline findings

1. **Units are mixed.** Financial statements, segments and shareholding counts are in **absolute rupees**. `market_cap` in `/quote` and `/company-profile` is in **₹ crore**. Ratios are fractions (0.2853 means 28.53%), and shareholding percentages are 0–100.
2. **The free plan covers only the 4 sample stocks** (ITC, RELIANCE, HDFCBANK, ICICIPRULI). Any other symbol returns `401 ER0012 - user not authorized` (TCS was tested). `/peers` still returns the peer names, but no peer data can be fetched. The competitor-analysis section cannot work on the free plan.
3. **Zero is used where data is missing.** Unused schema lines come back as `0`, not `null`. So do HDFC's consolidated NPA and CET1 values, and the P/B and P/FCF values on interim (Sep/Dec) annual-price-ratio rows. Real nulls appear only in ratios for the oldest year.
4. **The balance sheet and cash flow are half-yearly, not quarterly.** `period=quarterly` returns Sep and Mar periods only, because Indian companies disclose them that way. The P&L is truly quarterly.
5. **Cash-flow history is short: 5 fiscal years** (FY22–FY26). The P&L has 8–9 years and the balance sheet 7–8.
6. **HDFC Bank has a proper bank layout**, with different P&L and balance-sheet fields. Asset quality and capital data (GNPA, NNPA, CET1) is present in **standalone only**. In consolidated it is all 0.
7. **Pledge fields exist** (`pledgedShares`, `pledgedSharesPct` per holder in ownership-history), but they are 0 for every quarter of all 3 stocks. That is plausible (RIL promoters have no pledge, and ITC and HDFC Bank have no promoter), but it means non-zero pledge data could not be checked.

## Coverage table

✅ complete · 🟡 partial or has caveats · ❌ missing or blocked

| Data | ITC | RELIANCE | HDFCBANK | Notes |
|---|---|---|---|---|
| P&L annual (c) | ✅ 9 yrs FY18–FY26 | ✅ 8 yrs FY19–FY26 | ✅ 8 yrs FY19–FY26 | 4 quarters sum exactly to the annual figure for all 3 |
| P&L annual (s) | ✅ 9 yrs | ✅ 8 yrs | ✅ 8 yrs | |
| P&L quarterly (c) | ✅ 30 Q, Jun-19→Sep-26 | ✅ 34 Q, Jun-18→Sep-26 | ✅ 31 Q, Mar-19→Sep-26 | no gaps |
| P&L quarterly (s) | ✅ 35 Q, Mar-18→Sep-26 | ✅ 34 Q | 🟡 33 Q, **Dec-18 missing** | |
| Balance sheet annual | ✅ 8 yrs FY19–FY26 | ✅ 8 yrs | 🟡 c: 7 yrs FY20–FY26; s: 8 yrs | assets = equity + liabilities checks out |
| Balance sheet "quarterly" | 🟡 16–17 half-years | 🟡 17 half-years | 🟡 c 14 / s 17 half-years | Sep and Mar only |
| Cash flow annual | 🟡 5 yrs FY22–FY26 | 🟡 5 yrs | 🟡 5 yrs | too short for a 5-yr CAGR plus a base year |
| Cash flow "quarterly" | 🟡 11 half-years from Sep-21 | 🟡 11 | 🟡 11 | half-yearly |
| TTM P&L (`period=ttm`) | ✅ | (not tested) | (not tested) | matches the sum of the last 4 quarters |
| Ratios (pr/le/li/ef) | ✅ FY18–FY26 + TTM | ✅ FY19–FY26 + TTM | 🟡 bank set only | oldest year has nulls (no prior year to average) |
| Bank ratios | n/a | n/a | 🟡 NIM, ROA, ROE, leverage only | no CASA, cost-to-income, CAR or credit cost |
| Annual price ratios (PE/PB/PS/PFCF) | ✅ 10 rows | 🟡 23 rows, mixed periods | ✅ 11 rows | interim rows have PB/PFCF = 0; filter `header` on "Mar"/"TTM" |
| Daily price ratios | 🟡 2,195 days from 2018 | ✅ 2,131 days | 🟡 2,135 days | ITC has 70 days and HDFC 257 days where PE or PB = 0 |
| Daily quotes (adjusted OHLCV) | 🟡 | 🟡 | 🟡 | 2013-01-01→2026-11-13; adjusted for bonuses (no jump at RIL Oct-24 or HDFC Aug-25); 10–18 rows with close = 0 (holidays/muhurat) must be dropped |
| Current quote | ✅ | ✅ | ✅ | price, 52-week high/low, market cap (₹ cr), shares |
| Company profile | ✅ | ✅ | ✅ | sector, industry, description, website; market_cap in ₹ cr, slightly stale compared with `/quote` |
| Peers (sector) | 🟡 4 names | 🟡 6 names | 🟡 19 names | names only; list includes the stock itself; **peer data is blocked (401)** |
| Shareholding pattern | ✅ 42 Q, Jun-16→Sep-26 | ✅ 42 Q | 🟡 42 Q | see the classification issues below |
| Ownership history (per holder) | ✅ 35 Q | ✅ 34 Q | 🟡 33 Q | HDFC records lack the `group` and `sub_group` keys |
| Promoter pledge | 🟡 field present, all 0 | 🟡 all 0 | 🟡 all 0 | not checked against a stock with real pledges |
| NIFTY 50 history | ✅ | | | 1,240 days for 5 yrs, no zero closes; ends one day before the stock quotes |
| Segment revenue | ✅ 8 yrs, 5 segments | ✅ 8 yrs, 5 segments | ✅ 9 rows, 7 segments | revenue and segment PBIT |
| Dividends / corporate actions | ✅ | ✅ | ✅ | bonuses and splits listed; `adj_amount` is bonus-adjusted; corporate actions need `from_date`/`to_date` despite the spec saying optional |
| Growth metrics (3y/5y CAGR) | ✅ | ✅ | ✅ | `financial-metrics` takes `ratio_type` gr/av/cu, **not** pr (pr returns 400) |
| Concall transcripts | ❌ 0 | 🟡 7 links | 🟡 9 links | exchange-filing PDF links only, no text |
| Announcements / credit ratings | ✅ | ✅ | ✅ | exchange filings, PDF links |
| Analyst target price / rating | ❌ | ❌ | ❌ | no endpoint; the repo uses FMP for these |
| News | ❌ | ❌ | ❌ | announcements are the closest substitute |
| Beta, enterprise value, EBITDA field | ❌ | ❌ | ❌ | must be derived; no `grossProfit`, `ebitda` or SG&A fields |

## Standalone vs consolidated

- Both are available for all 3 stocks across the statements, ratios and price ratios.
- In consolidated, HDFC Bank's P&L has `grossNonPerformingAssets`, `percentageOfGrossNpa`, `percentageOfNpa` and `cET1Ratio` = 0 in every period. They are populated in standalone (latest Sep-26: GNPA 1.11%, NNPA 0.37%, CET1 18.87%). `additionalTier1Ratio` is 0 in both. **For a bank, take the asset-quality and capital fields from standalone.**
- HDFC's consolidated balance sheet starts in FY20 and its P&L in Mar-19. Its standalone data goes back further.

## HDFC Bank: bank-specific lines

- **P&L (27–28 fields):** interestEarned (split into advances / investments / RBI balances / other), interestExpended, otherIncome, employeesCost, otherOperatingExpenses, provisionsForLoanLoss, expenditureExcludingProvisions, NPA and CET1 fields (standalone only), and minority interest. There is no `revenueFromOperations` field, so the generator's revenue mapping must use `interestEarned` or `income`.
- **Balance sheet (14 fields):** deposits, advances, investments, borrowings, cashAndBalancesWithRBI, bankBalancesAndMoneyAtCall, fixedAssets, otherAssets, capital, reserves, otherLiabilitiesAndProvisions. There are no current/non-current splits, so the generator's current ratio, working capital and so on do not apply.
- **Ratios:** the le/li/ef sets shrink to interestCoverage and assetTurnover, and `netInterestMargin` is added.
- **Missing for a bank report:** CASA ratio, capital adequacy (total CAR / Tier 1), cost-to-income and credit cost (derivable), and loan or deposit mix.

## Data issues to handle in an adapter

1. **ITC revenue is not comparable after Feb 2026.** Per ITC's own notes (in `/notes`), excise duty and GST on cigarettes are included in revenue from 1 Feb 2026, and there is no separate excise line. Quarterly revenue jumps from ₹23.1k cr (Jun-25) to ₹29.5k cr (Jun-26). TTM EBIT margin shows 23.3% against 30.0% for FY26. Growth and margin calculations will be distorted unless gross revenue is netted. The notes also say figures from 13 Jun 2025 are not comparable with earlier periods.
2. **Category reclassifications in the shareholding pattern.**
   - ITC: foreign institutions go from 14.5% to 43% and non-institutions from 45% to 14.8% between Mar-22 and Mar-23 (a BAT reclassification).
   - HDFC Bank: the promoter is "Promoter Indian 25.59%" in Mar-23, "Promoter Foreign 25.59%" in Jun-23, then 0 from the merger onwards.
   - Trend charts need annotations for these.
3. **Net profit is not PBT minus tax.** PAT includes associates (ITC +₹377 cr, RIL +₹144 cr). HDFC's PAT includes minority interest (₹3,193 cr). Use `profitOrLossAttributableToOwners` for EPS-consistent figures.
4. **EPS and share counts are restated for bonuses** (RIL shows 1,353 cr shares in every year, and HDFC FY23 EPS of 41.22 is post-bonus). This matches the adjusted prices, which is good.
5. **Field names are inconsistent between company types**: `profitBeforeTax` vs `profitLossBeforeTax`, `profitLossForPeriod` vs `profitLossForThePeriod`, `equityAndLiabilities` vs `capitalAndLiabilities`. There is also a typo, `catagory`, in the shareholding pattern.
6. **The cash-flow schema has 118 fields, of which 34–47 are populated.** Free cash flow = `cashFlowsFromOperatingActivities` − `purchaseOfFixed&IntangibleAssets` (or `purchaseOfPPEClassifiedAsInvesting`).
7. **Share counts differ between endpoints.** ITC: `/quote` shares = 1,224.5 cr, but paid-up shares in the shareholding summary = 1,253.1 cr.
8. **Reliability:** there were no 5xx errors, timeouts or rate limits across the 129 calls (0.4 s spacing). The only failures were 401s for the non-sample symbol (TCS) and 400s for parameters I got wrong (`ratio_type=pr` on financial-metrics, and corporate actions without dates).

## Fit against what finrobot_equity currently pulls from FMP

| Generator input (FMP today) | FinEdge equivalent | Status |
|---|---|---|
| income statement, 5 yrs | `/financials pl annual` | ✅ needs field mapping; derive gross profit, EBITDA and SG&A |
| balance sheet, 5 yrs | `/financials bs annual` | ✅ (bank layout differs) |
| cash flow, 5 yrs | `/financials cf annual` | 🟡 exactly 5 yrs, no base year for growth |
| ratios / key metrics | `/ratios`, `/financial-metrics`, `/annual-price-ratios` | ✅ |
| current price, market cap, 52-week range | `/quote` | ✅ |
| company profile | `/company-profile` | ✅ |
| price history and volume (technicals) | `/daily-quotes` | ✅ after dropping zero rows |
| peer financials | blocked on free plan | ❌ |
| analyst target and rating | none | ❌ |
| news | `/corp-announcements` (filings only) | 🟡 |
| enterprise value history, beta | none; derive from price, debt and cash, and NIFTY 50 | 🟡 |
