#!/usr/bin/env python
# coding: utf-8

import re
import pandas as pd
from typing import Dict, Optional
from modules.llm_providers import LLMSettings, generate_text, normalize_provider

from modules.retail_sentiment_client import format_retail_sentiment_for_prompt, has_coverage
from modules import currency
from modules.report_context import render_table_block


def _get_fallback_text(prompt_type: str, company_name: str) -> str:
    """Returns fallback text when agent generation fails."""
    fallbacks = {
        "tagline": f"{company_name} demonstrates strong financial fundamentals with consistent revenue growth and solid profitability metrics. The company maintains a competitive position in its market segment through operational efficiency and strategic initiatives. Strong balance sheet metrics support continued value creation for shareholders.",
        "company_overview": f"{company_name} operates as a prominent player in its industry sector, demonstrating consistent financial performance through strategic market positioning and operational excellence. The company has shown resilient growth patterns supported by strong demand dynamics and effective cost management strategies.",
        "investment_overview": f"{company_name} has delivered solid financial performance in recent periods, supported by strong operational execution and favorable market conditions. Revenue growth has been driven by robust demand and strategic initiatives, while margin improvements reflect operational efficiency gains.",
        "valuation_overview": f"{company_name} trades at reasonable valuation levels relative to its peer group, supported by strong fundamental metrics and growth prospects. The company's financial profile demonstrates consistent profitability and cash generation capabilities.",
        "risks": "Key risks include: (1) Industry competition and market share pressure, (2) Regulatory changes affecting operations, (3) Economic downturns impacting demand, (4) Technology disruption risks, (5) Supply chain and operational challenges.",
        "competitor_analysis": f"{company_name} demonstrates competitive positioning within its industry through consistent financial performance and strategic market positioning relative to key competitors in the sector.",
        "major_takeaways": f"Revenue Growth: {company_name}'s revenue growth shows consistent performance trends.\n\nGross Profit Margin: {company_name}'s gross profit margins demonstrate operational effectiveness.\n\nSG&A Expense Margin: {company_name}'s SG&A expense management shows disciplined cost control.\n\nEBITDA Margin Stability: {company_name}'s EBITDA margin stability reflects strong underlying fundamentals.",
        "news_summary": f"Recent news coverage for {company_name} reflects ongoing market interest and developments in the company's operations and strategic initiatives."
    }
    return fallbacks.get(prompt_type, f"{company_name} analysis for {prompt_type.replace('_', ' ')} section.")


# System prompts for each text section
SYSTEM_PROMPTS = {
    "tagline": "You are an equity research analyst. Create a 3-sentence professional tagline summarizing the company's financial position. Be concise and professional. Do not use markdown.",
    "company_overview": "You are a financial analyst. Write a company description of 2 to 3 sentences: what the company does, where its revenue comes from, and its position in its industry. Do not discuss recent financial performance or valuation. Use plain text, no markdown.",
    "investment_overview": "You are an investment analyst. Write an investment update (200-300 words) covering recent financial performance, growth drivers, and outlook. Use plain text, no markdown.",
    "valuation_overview": "You are a valuation analyst. Write a valuation analysis (200-300 words) covering current valuation metrics, peer comparison, and fair value assessment. Use plain text, no markdown.",
    "risks": "You are a risk analyst. List 5 key investment risks in bullet point format. Be specific and concise.",
    "competitor_analysis": "You are a competitive analyst. Write a competitor analysis (200-300 words) comparing the company to its peers. Use plain text, no markdown.",
    "major_takeaways": "You are a financial analyst. Provide 4 major takeaways covering: Revenue Growth, Gross Profit Margin, SG&A Expense Margin, and EBITDA Margin. Write each as one paragraph that starts with its label and a colon, then 1-2 sentences, exactly like 'Revenue Growth: ...'. Use the labels 'Revenue Growth:', 'Gross Profit Margin:', 'SG&A Expense Margin:' and 'EBITDA Margin:'. Plain text only: no title, no markdown, no bold, no bullets.",
    "news_summary": "You are a financial news analyst. Summarize the recent news (200-300 words) highlighting key developments and their investment implications. Use plain text, no markdown."
}


BANK_SYSTEM_PROMPTS = {
    "major_takeaways": "You are a bank analyst. Provide 4 major takeaways covering: Net Interest Income Growth, Net Interest Margin, Asset Quality (GNPA/NNPA and credit cost), and Profitability (ROA/ROE and cost-to-income). Write each as one paragraph that starts with its label and a colon, then 1-2 sentences, exactly like 'Net Interest Margin: ...'. Use the labels 'Net Interest Income Growth:', 'Net Interest Margin:', 'Asset Quality:' and 'Profitability:'. Plain text only: no title, no markdown, no bold, no bullets.",
}

BANK_TAKEAWAY_HEADERS = ["Net Interest Income Growth:", "Net Interest Margin:", "Asset Quality:", "Profitability:"]


FINAL_INSTRUCTIONS = {
    "company_overview": ("Write 2 or 3 sentences only: what the company does and where its revenue comes from, using the company profile above. "
                         "Do not mention any financial figure, margin, multiple, peer or forecast."),
    "tagline": "Write 3 short sentences. Use at most one number, taken exactly from the facts.",
}

WRITING_RULES = (
    "\n\nWRITING RULES (apply to every sentence):\n"
    "- Use only the numbers given in the structured facts. Never compute, round differently or invent a figure; copy each figure exactly as written there.\n"
    "- Give every number what it is compared against and what it implies for the business or the investor.\n"
    "- Short sentences and plain language for a non-expert investor. Define a technical term once, on first use.\n"
    "- Do not use em-dashes or en-dashes.\n"
    "- Label any general market norm as 'general norm, not company-specific'.\n"
    "- Where full-year and TTM figures, or the model forecast and the peer-dataset projection, disagree, state the disagreement and its size.\n"
    "- Prefer the computed verdicts, peer statements and explanations in the facts to your own reading. Make peer claims only from the peer statements given.\n"
    "- Every key fact is stated in full in one section only. Follow the SECTION GUIDANCE: do not restate facts owned by other sections.\n"
    "- No filler. Do not write a sentence that would be true of any company; each sentence must cite a specific figure or computed fact."
)


def clean_generated_text(text: str) -> str:
    """Deterministic clean-up of model output: no em-dashes or en-dashes."""
    if not isinstance(text, str):
        return text
    text = re.sub(r"(?<=\d)\s*[\u2013\u2014]\s*(?=\d)", " to ", text)
    text = re.sub(r"\s*[\u2014]\s*", ", ", text)
    return text.replace("\u2013", "-")


def get_system_prompt(prompt_type: str, is_bank: bool = False) -> str:
    """System prompt for a section, with bank-specific variants where they differ, plus the shared writing rules."""
    if is_bank and prompt_type in BANK_SYSTEM_PROMPTS:
        base = BANK_SYSTEM_PROMPTS[prompt_type]
    else:
        base = SYSTEM_PROMPTS.get(prompt_type, f"You are a financial analyst. Provide {prompt_type.replace('_', ' ')} analysis.")
    return base + WRITING_RULES


def _df_to_string(df: Optional[pd.DataFrame], name: str) -> str:
    """Converts a DataFrame to a markdown string for use in a prompt."""
    if df is None or df.empty:
        return f"{name}:\n[Data not available]\n"
    
    try:
        return f"{name}:\n{df.to_markdown()}\n"
    except Exception as e:
        return f"{name}:\n[Error formatting data: {e}]\n"


def _prepare_user_prompt(data: Dict, prompt_type: str, company_name: str, company_ticker: str) -> str:
    """Prepare user prompt with financial data."""
    financial_metrics = data.get('financial_metrics')
    peer_ebitda = data.get('peer_ebitda')
    peer_ev_ebitda = data.get('peer_ev_ebitda')
    company_news = data.get('company_news')
    retail_sentiment = data.get('retail_sentiment')
    
    prompt = f"Company: {company_name} ({company_ticker})\n\n"
    
    facts_block = data.get('facts_block')   # canonical, code-computed facts (modules.report_context)
    if financial_metrics is not None and not financial_metrics.empty:
        table = _df_to_string(financial_metrics, "Financial Metrics")
        prompt += render_table_block(table) + "\n" if facts_block else table

    if facts_block:
        prompt += "\n" + facts_block + "\n\n"
        ins = (data.get('report_context') or {}).get('insights')
        if ins:
            from modules.insights import section_guidance
            prompt += section_guidance(ins, prompt_type) + "\n\n"

    if peer_ebitda is not None and not peer_ebitda.empty:
        prompt += _df_to_string(peer_ebitda, "Peer EBITDA Comparison (INR crore, actual years only)" if facts_block else "Peer EBITDA Comparison")

    if peer_ev_ebitda is not None and not peer_ev_ebitda.empty:
        prompt += _df_to_string(peer_ev_ebitda, "Peer EV/EBITDA Comparison (fiscal-year average price basis, includes the subject company)" if facts_block else "Peer EV/EBITDA Comparison")

    if prompt_type == "news_summary" and company_news:
        prompt += f"\n## Recent News:\n"
        # Headlines first, then routine exchange filings (a stable sort keeps newest-first within each group)
        ordered = sorted(company_news, key=lambda a: a.get('site') == 'NSE/BSE filing')
        for i, article in enumerate(ordered[:10], 1):  # Limit to 10 articles
            prompt += f"{i}. {article.get('title', 'N/A')} ({article.get('publishedDate', 'N/A')[:10]})\n"
            prompt += f"   {article.get('text', 'N/A')[:200]}...\n\n"

    if prompt_type == "news_summary" and has_coverage(retail_sentiment):
        prompt += "\n" + format_retail_sentiment_for_prompt(retail_sentiment) + "\n"

    data_notes = data.get('data_notes')
    if data_notes:
        prompt += "\nData notes (read before interpreting the numbers):\n" + "\n".join(f"- {n}" for n in data_notes) + "\n"

    if currency.get_currency() == "INR":
        prompt += ("\nUnits: write rupee amounts in crore or lakh crore (1 crore = 10 million), "
                   "never in millions, billions or trillions.\n")

    desc = data.get('company_description')
    if desc and prompt_type in ('company_overview', 'tagline'):
        prompt += "\nCompany profile from the data provider (use this, not general knowledge):\n" + str(desc)[:900] + "\n"
    final = FINAL_INSTRUCTIONS.get(prompt_type)
    prompt += "\n" + (final or f"Please provide the {prompt_type.replace('_', ' ')} based on the above data.")
    return prompt


def generate_text_section(data: Dict, prompt_type: str, api_key: str, company_name: str, company_ticker: str,
                          base_url: str = None, model: str = None, provider: str = "openai",
                          settings: Optional[LLMSettings] = None,
                          failures: Optional[list] = None,
                          usage_sink: Optional[list] = None) -> str:
    """
    Generates a specific text section for the equity report with the chosen LLM provider.
    
    Args:
        data: Financial data dictionary
        prompt_type: Type of text section to generate
        api_key: API key for the provider (ignored when ``settings`` is given)
        company_name: Company name
        company_ticker: Stock ticker
        base_url: Optional OpenAI-compatible base URL (OpenAI provider only)
        model: Optional model name (default depends on the provider)
        provider: 'openai', 'anthropic' or 'gemini'
        settings: Ready-made LLMSettings (overrides api_key/base_url/model/provider)
        failures: Optional list; ``prompt_type`` is appended whenever placeholder fallback
            text is returned instead of real LLM output, so callers can detect it.
        usage_sink: Optional list; token usage and estimated cost of the call are appended to it.
    """
    
    print(f"🤖 Generating '{prompt_type}' text section...")
    if settings is None:
        settings = LLMSettings(provider=normalize_provider(provider), api_key=api_key,
                               model=model, base_url=base_url)
    
    # Validate API key
    if not settings.api_key:
        print(f"⚠️ Warning: No API key provided. Using fallback text for '{prompt_type}'.")
        if failures is not None:
            failures.append(prompt_type)
        return _get_fallback_text(prompt_type, company_name)
    
    if settings.base_url:
        print(f"📡 Using API base URL: {settings.base_url}")
    print(f"🤖 Using {settings.label}")
    
    # Get system prompt
    system_prompt = get_system_prompt(prompt_type, bool(data.get('is_bank')))
    
    # Prepare user prompt with data
    user_prompt = _prepare_user_prompt(data, prompt_type, company_name, company_ticker)
    
    try:
        generated_text = generate_text(settings, system_prompt, user_prompt, max_tokens=1000, temperature=0.7,
                                      usage_sink=usage_sink, section=prompt_type)
        generated_text = clean_generated_text(generated_text)
        print(f"✅ Successfully generated '{prompt_type}' ({len(generated_text)} chars)")
        return generated_text
    except Exception as e:
        print(f"❌ Error generating '{prompt_type}': {e}")
        if failures is not None:
            failures.append(prompt_type)
        return _get_fallback_text(prompt_type, company_name)

# Backward compatibility - keep old function signature
def _query_openai(prompt: str, api_key: str) -> str:
    """Legacy function for backward compatibility."""
    return "Text generation now handled by agents."

if __name__ == '__main__':
    print("Testing agent-based text_generator...")
