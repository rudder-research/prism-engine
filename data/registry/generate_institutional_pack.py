#!/usr/bin/env python3
"""
PRISM Registry Generator - Full Institutional Pack (250+ Indicators)
====================================================================

This script programmatically generates all registry YAML files for the
Full Institutional Pack. It creates 250+ indicators across categories:

- Market Structure (Equities/Indexes)
- Rates & Yield Curve
- Credit Markets
- Liquidity
- FX & Commodities
- Volatility
- Economic Indicators
- Synthetic Indicators
- Technical Indicators

Usage:
    python data/registry/generate_institutional_pack.py

Output files are written to data/registry/yaml/
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml


# =============================================================================
# Configuration
# =============================================================================

REGISTRY_DIR = Path(__file__).parent
YAML_DIR = REGISTRY_DIR / "yaml"

# Ensure YAML directory exists
YAML_DIR.mkdir(exist_ok=True)


# =============================================================================
# Category Definitions - Market Structure
# =============================================================================

MARKET_GLOBAL_INDICATORS = [
    # U.S. Equity Indices (ETF proxies)
    {"id": "spy", "name": "S&P 500 ETF", "source": "yahoo", "ticker": "SPY", "group": "us_equity_index", "asset_class": "equity"},
    {"id": "qqq", "name": "NASDAQ 100 ETF", "source": "yahoo", "ticker": "QQQ", "group": "us_equity_index", "asset_class": "equity"},
    {"id": "dia", "name": "Dow Jones ETF", "source": "yahoo", "ticker": "DIA", "group": "us_equity_index", "asset_class": "equity"},
    {"id": "iwm", "name": "Russell 2000 ETF", "source": "yahoo", "ticker": "IWM", "group": "us_equity_index", "asset_class": "equity"},
    {"id": "mdy", "name": "S&P MidCap 400 ETF", "source": "yahoo", "ticker": "MDY", "group": "us_equity_index", "asset_class": "equity"},
    {"id": "ijh", "name": "S&P MidCap 400 iShares", "source": "yahoo", "ticker": "IJH", "group": "us_equity_index", "asset_class": "equity"},
    {"id": "sly", "name": "S&P SmallCap 600 ETF", "source": "yahoo", "ticker": "SLY", "group": "us_equity_index", "asset_class": "equity"},

    # U.S. Index Futures/Direct
    {"id": "spx_index", "name": "S&P 500 Index", "source": "yahoo", "ticker": "^GSPC", "group": "us_equity_index", "asset_class": "equity"},
    {"id": "ndx_index", "name": "NASDAQ 100 Index", "source": "yahoo", "ticker": "^NDX", "group": "us_equity_index", "asset_class": "equity"},
    {"id": "dji_index", "name": "Dow Jones Industrial", "source": "yahoo", "ticker": "^DJI", "group": "us_equity_index", "asset_class": "equity"},
    {"id": "rut_index", "name": "Russell 2000 Index", "source": "yahoo", "ticker": "^RUT", "group": "us_equity_index", "asset_class": "equity"},

    # Global Indices
    {"id": "ftse_100", "name": "FTSE 100 Index", "source": "yahoo", "ticker": "^FTSE", "group": "global_equity_index", "asset_class": "equity"},
    {"id": "dax", "name": "DAX Index", "source": "yahoo", "ticker": "^GDAXI", "group": "global_equity_index", "asset_class": "equity"},
    {"id": "cac_40", "name": "CAC 40 Index", "source": "yahoo", "ticker": "^FCHI", "group": "global_equity_index", "asset_class": "equity"},
    {"id": "nikkei_225", "name": "Nikkei 225 Index", "source": "yahoo", "ticker": "^N225", "group": "global_equity_index", "asset_class": "equity"},
    {"id": "hang_seng", "name": "Hang Seng Index", "source": "yahoo", "ticker": "^HSI", "group": "global_equity_index", "asset_class": "equity"},
    {"id": "shanghai_comp", "name": "Shanghai Composite", "source": "yahoo", "ticker": "000001.SS", "group": "global_equity_index", "asset_class": "equity"},
    {"id": "stoxx_600", "name": "STOXX Europe 600", "source": "yahoo", "ticker": "^STOXX", "group": "global_equity_index", "asset_class": "equity"},
    {"id": "asx_200", "name": "ASX 200 Index", "source": "yahoo", "ticker": "^AXJO", "group": "global_equity_index", "asset_class": "equity"},
    {"id": "tsx_comp", "name": "TSX Composite", "source": "yahoo", "ticker": "^GSPTSE", "group": "global_equity_index", "asset_class": "equity"},
    {"id": "bovespa", "name": "Bovespa Index", "source": "yahoo", "ticker": "^BVSP", "group": "global_equity_index", "asset_class": "equity"},
    {"id": "kospi", "name": "KOSPI Index", "source": "yahoo", "ticker": "^KS11", "group": "global_equity_index", "asset_class": "equity"},
    {"id": "sensex", "name": "BSE Sensex", "source": "yahoo", "ticker": "^BSESN", "group": "global_equity_index", "asset_class": "equity"},
    {"id": "nifty_50", "name": "NIFTY 50 Index", "source": "yahoo", "ticker": "^NSEI", "group": "global_equity_index", "asset_class": "equity"},

    # MSCI ETF Proxies
    {"id": "acwi", "name": "MSCI ACWI ETF", "source": "yahoo", "ticker": "ACWI", "group": "msci_index", "asset_class": "equity"},
    {"id": "efa", "name": "MSCI EAFE ETF", "source": "yahoo", "ticker": "EFA", "group": "msci_index", "asset_class": "equity"},
    {"id": "eem", "name": "MSCI Emerging Markets ETF", "source": "yahoo", "ticker": "EEM", "group": "msci_index", "asset_class": "equity"},
    {"id": "vea", "name": "Developed Markets ETF", "source": "yahoo", "ticker": "VEA", "group": "msci_index", "asset_class": "equity"},
    {"id": "vwo", "name": "Emerging Markets ETF", "source": "yahoo", "ticker": "VWO", "group": "msci_index", "asset_class": "equity"},
    {"id": "iemg", "name": "Core EM ETF", "source": "yahoo", "ticker": "IEMG", "group": "msci_index", "asset_class": "equity"},

    # Country ETFs
    {"id": "ewj", "name": "Japan ETF", "source": "yahoo", "ticker": "EWJ", "group": "country_equity", "asset_class": "equity"},
    {"id": "ewg", "name": "Germany ETF", "source": "yahoo", "ticker": "EWG", "group": "country_equity", "asset_class": "equity"},
    {"id": "ewu", "name": "UK ETF", "source": "yahoo", "ticker": "EWU", "group": "country_equity", "asset_class": "equity"},
    {"id": "fxi", "name": "China Large-Cap ETF", "source": "yahoo", "ticker": "FXI", "group": "country_equity", "asset_class": "equity"},
    {"id": "ewz", "name": "Brazil ETF", "source": "yahoo", "ticker": "EWZ", "group": "country_equity", "asset_class": "equity"},
    {"id": "inda", "name": "India ETF", "source": "yahoo", "ticker": "INDA", "group": "country_equity", "asset_class": "equity"},
    {"id": "ewy", "name": "South Korea ETF", "source": "yahoo", "ticker": "EWY", "group": "country_equity", "asset_class": "equity"},
    {"id": "ewc", "name": "Canada ETF", "source": "yahoo", "ticker": "EWC", "group": "country_equity", "asset_class": "equity"},
    {"id": "ewa", "name": "Australia ETF", "source": "yahoo", "ticker": "EWA", "group": "country_equity", "asset_class": "equity"},
    {"id": "fez", "name": "Euro Stoxx 50 ETF", "source": "yahoo", "ticker": "FEZ", "group": "country_equity", "asset_class": "equity"},

    # U.S. Sector ETFs (Select Sector SPDRs)
    {"id": "xlk", "name": "Technology Sector", "source": "yahoo", "ticker": "XLK", "group": "us_sector", "asset_class": "equity"},
    {"id": "xlf", "name": "Financial Sector", "source": "yahoo", "ticker": "XLF", "group": "us_sector", "asset_class": "equity"},
    {"id": "xle", "name": "Energy Sector", "source": "yahoo", "ticker": "XLE", "group": "us_sector", "asset_class": "equity"},
    {"id": "xlv", "name": "Health Care Sector", "source": "yahoo", "ticker": "XLV", "group": "us_sector", "asset_class": "equity"},
    {"id": "xli", "name": "Industrial Sector", "source": "yahoo", "ticker": "XLI", "group": "us_sector", "asset_class": "equity"},
    {"id": "xly", "name": "Consumer Discretionary", "source": "yahoo", "ticker": "XLY", "group": "us_sector", "asset_class": "equity"},
    {"id": "xlp", "name": "Consumer Staples", "source": "yahoo", "ticker": "XLP", "group": "us_sector", "asset_class": "equity"},
    {"id": "xlb", "name": "Materials Sector", "source": "yahoo", "ticker": "XLB", "group": "us_sector", "asset_class": "equity"},
    {"id": "xlu", "name": "Utilities Sector", "source": "yahoo", "ticker": "XLU", "group": "us_sector", "asset_class": "equity"},
    {"id": "xlre", "name": "Real Estate Sector", "source": "yahoo", "ticker": "XLRE", "group": "us_sector", "asset_class": "equity"},
    {"id": "xlc", "name": "Communication Services", "source": "yahoo", "ticker": "XLC", "group": "us_sector", "asset_class": "equity"},

    # Factor ETFs
    {"id": "mtum", "name": "Momentum Factor ETF", "source": "yahoo", "ticker": "MTUM", "group": "factor", "asset_class": "equity"},
    {"id": "vlue", "name": "Value Factor ETF", "source": "yahoo", "ticker": "VLUE", "group": "factor", "asset_class": "equity"},
    {"id": "qual", "name": "Quality Factor ETF", "source": "yahoo", "ticker": "QUAL", "group": "factor", "asset_class": "equity"},
    {"id": "usmv", "name": "Min Volatility ETF", "source": "yahoo", "ticker": "USMV", "group": "factor", "asset_class": "equity"},
    {"id": "size", "name": "Size Factor ETF", "source": "yahoo", "ticker": "SIZE", "group": "factor", "asset_class": "equity"},
]


# =============================================================================
# Category Definitions - U.S. Rates & Yield Curve
# =============================================================================

RATES_US_INDICATORS = [
    # Treasury Yields (full curve)
    {"id": "dgs1mo", "name": "1-Month Treasury Rate", "source": "fred", "series_id": "DGS1MO", "group": "us_treasury", "category": "interest_rates"},
    {"id": "dgs3mo", "name": "3-Month Treasury Rate", "source": "fred", "series_id": "DGS3MO", "group": "us_treasury", "category": "interest_rates"},
    {"id": "dgs6mo", "name": "6-Month Treasury Rate", "source": "fred", "series_id": "DGS6MO", "group": "us_treasury", "category": "interest_rates"},
    {"id": "dgs1", "name": "1-Year Treasury Rate", "source": "fred", "series_id": "DGS1", "group": "us_treasury", "category": "interest_rates"},
    {"id": "dgs2", "name": "2-Year Treasury Rate", "source": "fred", "series_id": "DGS2", "group": "us_treasury", "category": "interest_rates"},
    {"id": "dgs3", "name": "3-Year Treasury Rate", "source": "fred", "series_id": "DGS3", "group": "us_treasury", "category": "interest_rates"},
    {"id": "dgs5", "name": "5-Year Treasury Rate", "source": "fred", "series_id": "DGS5", "group": "us_treasury", "category": "interest_rates"},
    {"id": "dgs7", "name": "7-Year Treasury Rate", "source": "fred", "series_id": "DGS7", "group": "us_treasury", "category": "interest_rates"},
    {"id": "dgs10", "name": "10-Year Treasury Rate", "source": "fred", "series_id": "DGS10", "group": "us_treasury", "category": "interest_rates"},
    {"id": "dgs20", "name": "20-Year Treasury Rate", "source": "fred", "series_id": "DGS20", "group": "us_treasury", "category": "interest_rates"},
    {"id": "dgs30", "name": "30-Year Treasury Rate", "source": "fred", "series_id": "DGS30", "group": "us_treasury", "category": "interest_rates"},

    # TIPS (Inflation-Protected)
    {"id": "dfii5", "name": "5-Year TIPS Rate", "source": "fred", "series_id": "DFII5", "group": "us_tips", "category": "interest_rates"},
    {"id": "dfii10", "name": "10-Year TIPS Rate", "source": "fred", "series_id": "DFII10", "group": "us_tips", "category": "interest_rates"},
    {"id": "dfii20", "name": "20-Year TIPS Rate", "source": "fred", "series_id": "DFII20", "group": "us_tips", "category": "interest_rates"},
    {"id": "dfii30", "name": "30-Year TIPS Rate", "source": "fred", "series_id": "DFII30", "group": "us_tips", "category": "interest_rates"},

    # Breakeven Inflation Rates
    {"id": "t5yie", "name": "5-Year Breakeven Inflation", "source": "fred", "series_id": "T5YIE", "group": "breakeven", "category": "inflation_expectations"},
    {"id": "t10yie", "name": "10-Year Breakeven Inflation", "source": "fred", "series_id": "T10YIE", "group": "breakeven", "category": "inflation_expectations"},

    # Fed Funds & Policy Rates
    {"id": "fedfunds", "name": "Effective Federal Funds Rate", "source": "fred", "series_id": "FEDFUNDS", "group": "policy_rates", "category": "interest_rates"},
    {"id": "dfedtaru", "name": "Fed Funds Target Upper", "source": "fred", "series_id": "DFEDTARU", "group": "policy_rates", "category": "interest_rates"},
    {"id": "dfedtarl", "name": "Fed Funds Target Lower", "source": "fred", "series_id": "DFEDTARL", "group": "policy_rates", "category": "interest_rates"},
    {"id": "iorb", "name": "Interest on Reserve Balances", "source": "fred", "series_id": "IORB", "group": "policy_rates", "category": "interest_rates"},

    # SOFR
    {"id": "sofr", "name": "Secured Overnight Financing Rate", "source": "fred", "series_id": "SOFR", "group": "money_market", "category": "interest_rates"},
    {"id": "sofr30", "name": "30-Day Average SOFR", "source": "fred", "series_id": "SOFR30DAYAVG", "group": "money_market", "category": "interest_rates"},
    {"id": "sofr90", "name": "90-Day Average SOFR", "source": "fred", "series_id": "SOFR90DAYAVG", "group": "money_market", "category": "interest_rates"},

    # Treasury ETF proxies
    {"id": "tlt", "name": "20+ Year Treasury ETF", "source": "yahoo", "ticker": "TLT", "group": "treasury_etf", "asset_class": "fixed_income"},
    {"id": "ief", "name": "7-10 Year Treasury ETF", "source": "yahoo", "ticker": "IEF", "group": "treasury_etf", "asset_class": "fixed_income"},
    {"id": "shy", "name": "1-3 Year Treasury ETF", "source": "yahoo", "ticker": "SHY", "group": "treasury_etf", "asset_class": "fixed_income"},
    {"id": "shv", "name": "Short Treasury ETF", "source": "yahoo", "ticker": "SHV", "group": "treasury_etf", "asset_class": "fixed_income"},
    {"id": "bil", "name": "T-Bill ETF", "source": "yahoo", "ticker": "BIL", "group": "treasury_etf", "asset_class": "fixed_income"},
    {"id": "govt", "name": "Treasury Bond ETF", "source": "yahoo", "ticker": "GOVT", "group": "treasury_etf", "asset_class": "fixed_income"},
    {"id": "tip", "name": "TIPS ETF", "source": "yahoo", "ticker": "TIP", "group": "treasury_etf", "asset_class": "fixed_income"},
]


# =============================================================================
# Category Definitions - Global Rates
# =============================================================================

RATES_GLOBAL_INDICATORS = [
    # Germany Sovereign Yields
    {"id": "de_10y", "name": "Germany 10-Year Yield", "source": "fred", "series_id": "IRLTLT01DEM156N", "group": "germany_sovereign", "category": "interest_rates"},
    {"id": "de_2y", "name": "Germany 2-Year Yield", "source": "fred", "series_id": "IRLTST01DEM156N", "group": "germany_sovereign", "category": "interest_rates"},

    # UK Sovereign Yields
    {"id": "uk_10y", "name": "UK 10-Year Gilt Yield", "source": "fred", "series_id": "IRLTLT01GBM156N", "group": "uk_sovereign", "category": "interest_rates"},

    # Japan Sovereign Yields
    {"id": "jp_10y", "name": "Japan 10-Year JGB Yield", "source": "fred", "series_id": "IRLTLT01JPM156N", "group": "japan_sovereign", "category": "interest_rates"},

    # Italy Sovereign Yields
    {"id": "it_10y", "name": "Italy 10-Year BTP Yield", "source": "fred", "series_id": "IRLTLT01ITM156N", "group": "italy_sovereign", "category": "interest_rates"},

    # Canada Sovereign Yields
    {"id": "ca_10y", "name": "Canada 10-Year Yield", "source": "fred", "series_id": "IRLTLT01CAM156N", "group": "canada_sovereign", "category": "interest_rates"},

    # Australia
    {"id": "au_10y", "name": "Australia 10-Year Yield", "source": "fred", "series_id": "IRLTLT01AUM156N", "group": "australia_sovereign", "category": "interest_rates"},

    # Global Bond ETFs
    {"id": "bndx", "name": "International Bond ETF", "source": "yahoo", "ticker": "BNDX", "group": "global_bond_etf", "asset_class": "fixed_income"},
    {"id": "iagg", "name": "Intl Aggregate Bond ETF", "source": "yahoo", "ticker": "IAGG", "group": "global_bond_etf", "asset_class": "fixed_income"},
    {"id": "emb", "name": "EM USD Bond ETF", "source": "yahoo", "ticker": "EMB", "group": "em_bond_etf", "asset_class": "fixed_income"},
    {"id": "vwob", "name": "EM Government Bond ETF", "source": "yahoo", "ticker": "VWOB", "group": "em_bond_etf", "asset_class": "fixed_income"},
]


# =============================================================================
# Category Definitions - Credit Markets
# =============================================================================

CREDIT_INDICATORS = [
    # Corporate Yields
    {"id": "aaa_yield", "name": "AAA Corporate Bond Yield", "source": "fred", "series_id": "AAA", "group": "corporate_yield", "category": "credit"},
    {"id": "baa_yield", "name": "BAA Corporate Bond Yield", "source": "fred", "series_id": "BAA", "group": "corporate_yield", "category": "credit"},
    {"id": "baaffm", "name": "BAA Corporate Yield (Monthly)", "source": "fred", "series_id": "BAAFFM", "group": "corporate_yield", "category": "credit"},

    # Credit Spreads (direct from FRED)
    {"id": "aaa10y", "name": "AAA-10Y Treasury Spread", "source": "fred", "series_id": "AAA10Y", "group": "credit_spread", "category": "credit"},
    {"id": "baa10ym", "name": "BAA-10Y Treasury Spread", "source": "fred", "series_id": "BAA10YM", "group": "credit_spread", "category": "credit"},

    # High Yield OAS
    {"id": "bamlh0a0hym2", "name": "US HY OAS", "source": "fred", "series_id": "BAMLH0A0HYM2", "group": "hy_spread", "category": "credit"},
    {"id": "bamlc0a0cm", "name": "US IG OAS", "source": "fred", "series_id": "BAMLC0A0CM", "group": "ig_spread", "category": "credit"},
    {"id": "bamlc0a4cbbb", "name": "US BBB OAS", "source": "fred", "series_id": "BAMLC0A4CBBB", "group": "ig_spread", "category": "credit"},
    {"id": "bamlemcbpioas", "name": "EM Corporate OAS", "source": "fred", "series_id": "BAMLEMCBPIOAS", "group": "em_spread", "category": "credit"},

    # Money Market Spreads
    {"id": "tedrate", "name": "TED Spread", "source": "fred", "series_id": "TEDRATE", "group": "money_market_spread", "category": "credit"},

    # Credit ETFs
    {"id": "hyg", "name": "High Yield Corporate ETF", "source": "yahoo", "ticker": "HYG", "group": "credit_etf", "asset_class": "fixed_income"},
    {"id": "jnk", "name": "High Yield Bond ETF", "source": "yahoo", "ticker": "JNK", "group": "credit_etf", "asset_class": "fixed_income"},
    {"id": "lqd", "name": "IG Corporate Bond ETF", "source": "yahoo", "ticker": "LQD", "group": "credit_etf", "asset_class": "fixed_income"},
    {"id": "vcit", "name": "Intermediate IG Corp ETF", "source": "yahoo", "ticker": "VCIT", "group": "credit_etf", "asset_class": "fixed_income"},
    {"id": "vcsh", "name": "Short-Term Corp Bond ETF", "source": "yahoo", "ticker": "VCSH", "group": "credit_etf", "asset_class": "fixed_income"},
    {"id": "bnd", "name": "Total Bond Market ETF", "source": "yahoo", "ticker": "BND", "group": "credit_etf", "asset_class": "fixed_income"},
    {"id": "agg", "name": "Aggregate Bond ETF", "source": "yahoo", "ticker": "AGG", "group": "credit_etf", "asset_class": "fixed_income"},
    {"id": "mub", "name": "Municipal Bond ETF", "source": "yahoo", "ticker": "MUB", "group": "muni_etf", "asset_class": "fixed_income"},
]


# =============================================================================
# Category Definitions - Liquidity
# =============================================================================

LIQUIDITY_INDICATORS = [
    # Money Supply
    {"id": "m1", "name": "M1 Money Stock", "source": "fred", "series_id": "M1SL", "group": "money_supply", "category": "liquidity"},
    {"id": "m2", "name": "M2 Money Stock", "source": "fred", "series_id": "M2SL", "group": "money_supply", "category": "liquidity"},
    {"id": "mbase", "name": "Monetary Base", "source": "fred", "series_id": "BOGMBASE", "group": "money_supply", "category": "liquidity"},

    # Fed Balance Sheet
    {"id": "walcl", "name": "Fed Total Assets", "source": "fred", "series_id": "WALCL", "group": "fed_balance_sheet", "category": "liquidity"},
    {"id": "treast", "name": "Fed Treasury Holdings", "source": "fred", "series_id": "TREAST", "group": "fed_balance_sheet", "category": "liquidity"},
    {"id": "wshomcb", "name": "Fed MBS Holdings", "source": "fred", "series_id": "WSHOMCB", "group": "fed_balance_sheet", "category": "liquidity"},

    # Reverse Repo
    {"id": "rrpontsyd", "name": "Overnight Reverse Repo", "source": "fred", "series_id": "RRPONTSYD", "group": "repo_operations", "category": "liquidity"},
    {"id": "wlrral", "name": "Fed Reverse Repo Agreements", "source": "fred", "series_id": "WLRRAL", "group": "repo_operations", "category": "liquidity"},

    # Reserve Balances
    {"id": "totresns", "name": "Total Reserves", "source": "fred", "series_id": "TOTRESNS", "group": "reserves", "category": "liquidity"},
    {"id": "excsresns", "name": "Excess Reserves", "source": "fred", "series_id": "EXCSRESNS", "group": "reserves", "category": "liquidity"},
    {"id": "wresbal", "name": "Reserve Balances", "source": "fred", "series_id": "WRESBAL", "group": "reserves", "category": "liquidity"},

    # Treasury General Account
    {"id": "wtregen", "name": "Treasury General Account", "source": "fred", "series_id": "WTREGEN", "group": "tga", "category": "liquidity"},

    # Financial Conditions
    {"id": "nfci", "name": "Chicago Fed NFCI", "source": "fred", "series_id": "NFCI", "group": "financial_conditions", "category": "liquidity"},
    {"id": "anfci", "name": "Chicago Fed Adjusted NFCI", "source": "fred", "series_id": "ANFCI", "group": "financial_conditions", "category": "liquidity"},
    {"id": "stlfsi4", "name": "St Louis Fed FSI", "source": "fred", "series_id": "STLFSI4", "group": "financial_conditions", "category": "liquidity"},
]


# =============================================================================
# Category Definitions - FX
# =============================================================================

FX_INDICATORS = [
    # Dollar Index
    {"id": "dxy", "name": "US Dollar Index", "source": "yahoo", "ticker": "DX-Y.NYB", "group": "fx_index", "asset_class": "fx"},

    # Major Pairs (vs USD)
    {"id": "eurusd", "name": "EUR/USD", "source": "yahoo", "ticker": "EURUSD=X", "group": "fx_major", "asset_class": "fx"},
    {"id": "gbpusd", "name": "GBP/USD", "source": "yahoo", "ticker": "GBPUSD=X", "group": "fx_major", "asset_class": "fx"},
    {"id": "usdjpy", "name": "USD/JPY", "source": "yahoo", "ticker": "USDJPY=X", "group": "fx_major", "asset_class": "fx"},
    {"id": "usdchf", "name": "USD/CHF", "source": "yahoo", "ticker": "USDCHF=X", "group": "fx_major", "asset_class": "fx"},
    {"id": "audusd", "name": "AUD/USD", "source": "yahoo", "ticker": "AUDUSD=X", "group": "fx_major", "asset_class": "fx"},
    {"id": "usdcad", "name": "USD/CAD", "source": "yahoo", "ticker": "USDCAD=X", "group": "fx_major", "asset_class": "fx"},
    {"id": "nzdusd", "name": "NZD/USD", "source": "yahoo", "ticker": "NZDUSD=X", "group": "fx_major", "asset_class": "fx"},

    # EM Currencies
    {"id": "usdcny", "name": "USD/CNY", "source": "yahoo", "ticker": "USDCNY=X", "group": "fx_em", "asset_class": "fx"},
    {"id": "usdmxn", "name": "USD/MXN", "source": "yahoo", "ticker": "USDMXN=X", "group": "fx_em", "asset_class": "fx"},
    {"id": "usdbrl", "name": "USD/BRL", "source": "yahoo", "ticker": "USDBRL=X", "group": "fx_em", "asset_class": "fx"},
    {"id": "usdinr", "name": "USD/INR", "source": "yahoo", "ticker": "USDINR=X", "group": "fx_em", "asset_class": "fx"},
    {"id": "usdkrw", "name": "USD/KRW", "source": "yahoo", "ticker": "USDKRW=X", "group": "fx_em", "asset_class": "fx"},
    {"id": "usdzar", "name": "USD/ZAR", "source": "yahoo", "ticker": "USDZAR=X", "group": "fx_em", "asset_class": "fx"},
    {"id": "usdtry", "name": "USD/TRY", "source": "yahoo", "ticker": "USDTRY=X", "group": "fx_em", "asset_class": "fx"},
    {"id": "usdrub", "name": "USD/RUB", "source": "yahoo", "ticker": "USDRUB=X", "group": "fx_em", "asset_class": "fx"},

    # Cross Rates
    {"id": "eurgbp", "name": "EUR/GBP", "source": "yahoo", "ticker": "EURGBP=X", "group": "fx_cross", "asset_class": "fx"},
    {"id": "eurjpy", "name": "EUR/JPY", "source": "yahoo", "ticker": "EURJPY=X", "group": "fx_cross", "asset_class": "fx"},
    {"id": "gbpjpy", "name": "GBP/JPY", "source": "yahoo", "ticker": "GBPJPY=X", "group": "fx_cross", "asset_class": "fx"},

    # Currency ETFs
    {"id": "uup", "name": "US Dollar Bullish ETF", "source": "yahoo", "ticker": "UUP", "group": "fx_etf", "asset_class": "fx"},
    {"id": "fxe", "name": "Euro ETF", "source": "yahoo", "ticker": "FXE", "group": "fx_etf", "asset_class": "fx"},
    {"id": "fxy", "name": "Japanese Yen ETF", "source": "yahoo", "ticker": "FXY", "group": "fx_etf", "asset_class": "fx"},
]


# =============================================================================
# Category Definitions - Commodities
# =============================================================================

COMMODITY_INDICATORS = [
    # Crude Oil
    {"id": "cl_f", "name": "WTI Crude Oil Futures", "source": "yahoo", "ticker": "CL=F", "group": "energy", "asset_class": "commodity"},
    {"id": "bz_f", "name": "Brent Crude Futures", "source": "yahoo", "ticker": "BZ=F", "group": "energy", "asset_class": "commodity"},
    {"id": "uso", "name": "US Oil ETF", "source": "yahoo", "ticker": "USO", "group": "energy_etf", "asset_class": "commodity"},
    {"id": "bno", "name": "Brent Oil ETF", "source": "yahoo", "ticker": "BNO", "group": "energy_etf", "asset_class": "commodity"},

    # Natural Gas
    {"id": "ng_f", "name": "Natural Gas Futures", "source": "yahoo", "ticker": "NG=F", "group": "energy", "asset_class": "commodity"},
    {"id": "ung", "name": "Natural Gas ETF", "source": "yahoo", "ticker": "UNG", "group": "energy_etf", "asset_class": "commodity"},

    # Refined Products
    {"id": "rb_f", "name": "RBOB Gasoline Futures", "source": "yahoo", "ticker": "RB=F", "group": "energy", "asset_class": "commodity"},
    {"id": "ho_f", "name": "Heating Oil Futures", "source": "yahoo", "ticker": "HO=F", "group": "energy", "asset_class": "commodity"},

    # Precious Metals
    {"id": "gc_f", "name": "Gold Futures", "source": "yahoo", "ticker": "GC=F", "group": "precious_metals", "asset_class": "commodity"},
    {"id": "si_f", "name": "Silver Futures", "source": "yahoo", "ticker": "SI=F", "group": "precious_metals", "asset_class": "commodity"},
    {"id": "pl_f", "name": "Platinum Futures", "source": "yahoo", "ticker": "PL=F", "group": "precious_metals", "asset_class": "commodity"},
    {"id": "pa_f", "name": "Palladium Futures", "source": "yahoo", "ticker": "PA=F", "group": "precious_metals", "asset_class": "commodity"},
    {"id": "gld", "name": "Gold ETF", "source": "yahoo", "ticker": "GLD", "group": "precious_metals_etf", "asset_class": "commodity"},
    {"id": "slv", "name": "Silver ETF", "source": "yahoo", "ticker": "SLV", "group": "precious_metals_etf", "asset_class": "commodity"},
    {"id": "iau", "name": "Gold iShares", "source": "yahoo", "ticker": "IAU", "group": "precious_metals_etf", "asset_class": "commodity"},
    {"id": "gldm", "name": "Gold MiniShares", "source": "yahoo", "ticker": "GLDM", "group": "precious_metals_etf", "asset_class": "commodity"},

    # Industrial Metals
    {"id": "hg_f", "name": "Copper Futures", "source": "yahoo", "ticker": "HG=F", "group": "industrial_metals", "asset_class": "commodity"},
    {"id": "copx", "name": "Copper Miners ETF", "source": "yahoo", "ticker": "COPX", "group": "industrial_metals_etf", "asset_class": "commodity"},

    # Agriculture - Grains
    {"id": "zc_f", "name": "Corn Futures", "source": "yahoo", "ticker": "ZC=F", "group": "agriculture", "asset_class": "commodity"},
    {"id": "zw_f", "name": "Wheat Futures", "source": "yahoo", "ticker": "ZW=F", "group": "agriculture", "asset_class": "commodity"},
    {"id": "zs_f", "name": "Soybean Futures", "source": "yahoo", "ticker": "ZS=F", "group": "agriculture", "asset_class": "commodity"},
    {"id": "zl_f", "name": "Soybean Oil Futures", "source": "yahoo", "ticker": "ZL=F", "group": "agriculture", "asset_class": "commodity"},
    {"id": "zm_f", "name": "Soybean Meal Futures", "source": "yahoo", "ticker": "ZM=F", "group": "agriculture", "asset_class": "commodity"},

    # Agriculture - Softs
    {"id": "sb_f", "name": "Sugar Futures", "source": "yahoo", "ticker": "SB=F", "group": "softs", "asset_class": "commodity"},
    {"id": "kc_f", "name": "Coffee Futures", "source": "yahoo", "ticker": "KC=F", "group": "softs", "asset_class": "commodity"},
    {"id": "cc_f", "name": "Cocoa Futures", "source": "yahoo", "ticker": "CC=F", "group": "softs", "asset_class": "commodity"},
    {"id": "ct_f", "name": "Cotton Futures", "source": "yahoo", "ticker": "CT=F", "group": "softs", "asset_class": "commodity"},

    # Livestock
    {"id": "le_f", "name": "Live Cattle Futures", "source": "yahoo", "ticker": "LE=F", "group": "livestock", "asset_class": "commodity"},
    {"id": "he_f", "name": "Lean Hogs Futures", "source": "yahoo", "ticker": "HE=F", "group": "livestock", "asset_class": "commodity"},

    # Broad Commodity ETFs
    {"id": "dbc", "name": "Commodity Index ETF", "source": "yahoo", "ticker": "DBC", "group": "commodity_etf", "asset_class": "commodity"},
    {"id": "dba", "name": "Agriculture ETF", "source": "yahoo", "ticker": "DBA", "group": "commodity_etf", "asset_class": "commodity"},
    {"id": "dbb", "name": "Base Metals ETF", "source": "yahoo", "ticker": "DBB", "group": "commodity_etf", "asset_class": "commodity"},
    {"id": "dbe", "name": "Energy ETF", "source": "yahoo", "ticker": "DBE", "group": "commodity_etf", "asset_class": "commodity"},
    {"id": "gsg", "name": "GSCI Commodity ETF", "source": "yahoo", "ticker": "GSG", "group": "commodity_etf", "asset_class": "commodity"},
    {"id": "pdbc", "name": "Optimum Yield Commodity ETF", "source": "yahoo", "ticker": "PDBC", "group": "commodity_etf", "asset_class": "commodity"},
]


# =============================================================================
# Category Definitions - Volatility
# =============================================================================

VOLATILITY_INDICATORS = [
    # VIX Family
    {"id": "vix", "name": "CBOE VIX Index", "source": "yahoo", "ticker": "^VIX", "group": "equity_vol", "asset_class": "volatility"},
    {"id": "vix9d", "name": "CBOE 9-Day VIX", "source": "yahoo", "ticker": "^VIX9D", "group": "equity_vol", "asset_class": "volatility"},
    {"id": "vix3m", "name": "CBOE 3-Month VIX", "source": "yahoo", "ticker": "^VIX3M", "group": "equity_vol", "asset_class": "volatility"},
    {"id": "vix6m", "name": "CBOE 6-Month VIX", "source": "yahoo", "ticker": "^VIX6M", "group": "equity_vol", "asset_class": "volatility"},
    {"id": "vxn", "name": "CBOE NASDAQ Volatility", "source": "yahoo", "ticker": "^VXN", "group": "equity_vol", "asset_class": "volatility"},
    {"id": "rvx", "name": "CBOE Russell 2000 Vol", "source": "yahoo", "ticker": "^RVX", "group": "equity_vol", "asset_class": "volatility"},
    {"id": "vxd", "name": "CBOE DJIA Volatility", "source": "yahoo", "ticker": "^VXD", "group": "equity_vol", "asset_class": "volatility"},

    # MOVE Index (Treasury vol) - via FRED
    {"id": "move", "name": "ICE BofA MOVE Index", "source": "fred", "series_id": "MOVEIDX", "group": "rate_vol", "category": "volatility"},

    # Volatility ETFs/ETNs
    {"id": "vixy", "name": "VIX Short-Term Futures ETF", "source": "yahoo", "ticker": "VIXY", "group": "vol_etf", "asset_class": "volatility"},
    {"id": "vixm", "name": "VIX Mid-Term Futures ETF", "source": "yahoo", "ticker": "VIXM", "group": "vol_etf", "asset_class": "volatility"},
    {"id": "svxy", "name": "Short VIX Futures ETF", "source": "yahoo", "ticker": "SVXY", "group": "vol_etf", "asset_class": "volatility"},
    {"id": "uvxy", "name": "Ultra VIX Short-Term ETF", "source": "yahoo", "ticker": "UVXY", "group": "vol_etf", "asset_class": "volatility"},

    # Currency Vol
    {"id": "evz", "name": "Euro Currency Vol", "source": "yahoo", "ticker": "^EVZ", "group": "fx_vol", "asset_class": "volatility"},

    # Oil Vol
    {"id": "ovx", "name": "CBOE Crude Oil Vol", "source": "yahoo", "ticker": "^OVX", "group": "commodity_vol", "asset_class": "volatility"},
    {"id": "gvz", "name": "CBOE Gold Vol", "source": "yahoo", "ticker": "^GVZ", "group": "commodity_vol", "asset_class": "volatility"},
]


# =============================================================================
# Category Definitions - U.S. Economics
# =============================================================================

ECONOMICS_US_INDICATORS = [
    # Inflation
    {"id": "cpi", "name": "CPI All Urban Consumers", "source": "fred", "series_id": "CPIAUCSL", "group": "inflation", "category": "prices", "frequency": "monthly"},
    {"id": "cpi_core", "name": "Core CPI (Less Food & Energy)", "source": "fred", "series_id": "CPILFESL", "group": "inflation", "category": "prices", "frequency": "monthly"},
    {"id": "pce", "name": "PCE Price Index", "source": "fred", "series_id": "PCEPI", "group": "inflation", "category": "prices", "frequency": "monthly"},
    {"id": "pce_core", "name": "Core PCE Price Index", "source": "fred", "series_id": "PCEPILFE", "group": "inflation", "category": "prices", "frequency": "monthly"},
    {"id": "ppi", "name": "Producer Price Index", "source": "fred", "series_id": "PPIACO", "group": "inflation", "category": "prices", "frequency": "monthly"},
    {"id": "ppi_core", "name": "Core PPI", "source": "fred", "series_id": "PPIFES", "group": "inflation", "category": "prices", "frequency": "monthly"},
    {"id": "cpi_yoy", "name": "CPI YoY Change", "source": "fred", "series_id": "CPALTT01USM657N", "group": "inflation", "category": "prices", "frequency": "monthly"},

    # Employment
    {"id": "payems", "name": "Nonfarm Payrolls", "source": "fred", "series_id": "PAYEMS", "group": "employment", "category": "labor", "frequency": "monthly"},
    {"id": "unrate", "name": "Unemployment Rate", "source": "fred", "series_id": "UNRATE", "group": "employment", "category": "labor", "frequency": "monthly"},
    {"id": "u6rate", "name": "U-6 Unemployment Rate", "source": "fred", "series_id": "U6RATE", "group": "employment", "category": "labor", "frequency": "monthly"},
    {"id": "icsa", "name": "Initial Jobless Claims", "source": "fred", "series_id": "ICSA", "group": "employment", "category": "labor", "frequency": "weekly"},
    {"id": "ccsa", "name": "Continuing Claims", "source": "fred", "series_id": "CCSA", "group": "employment", "category": "labor", "frequency": "weekly"},
    {"id": "jolts", "name": "JOLTS Job Openings", "source": "fred", "series_id": "JTSJOL", "group": "employment", "category": "labor", "frequency": "monthly"},
    {"id": "jolts_quits", "name": "JOLTS Quits Rate", "source": "fred", "series_id": "JTSQUR", "group": "employment", "category": "labor", "frequency": "monthly"},
    {"id": "civpart", "name": "Labor Force Participation", "source": "fred", "series_id": "CIVPART", "group": "employment", "category": "labor", "frequency": "monthly"},
    {"id": "awhaetp", "name": "Avg Weekly Hours", "source": "fred", "series_id": "AWHAETP", "group": "employment", "category": "labor", "frequency": "monthly"},
    {"id": "ahetpi", "name": "Avg Hourly Earnings", "source": "fred", "series_id": "AHETPI", "group": "employment", "category": "labor", "frequency": "monthly"},

    # GDP & Output
    {"id": "gdp", "name": "Nominal GDP", "source": "fred", "series_id": "GDP", "group": "output", "category": "gdp", "frequency": "quarterly"},
    {"id": "gdpc1", "name": "Real GDP", "source": "fred", "series_id": "GDPC1", "group": "output", "category": "gdp", "frequency": "quarterly"},
    {"id": "gdp_growth", "name": "Real GDP Growth Rate", "source": "fred", "series_id": "A191RL1Q225SBEA", "group": "output", "category": "gdp", "frequency": "quarterly"},
    {"id": "indpro", "name": "Industrial Production", "source": "fred", "series_id": "INDPRO", "group": "output", "category": "production", "frequency": "monthly"},
    {"id": "capacity", "name": "Capacity Utilization", "source": "fred", "series_id": "TCU", "group": "output", "category": "production", "frequency": "monthly"},

    # Business Surveys (ISM/PMI)
    {"id": "ism_mfg", "name": "ISM Manufacturing PMI", "source": "fred", "series_id": "MANEMP", "group": "surveys", "category": "business", "frequency": "monthly"},
    {"id": "ism_nonmfg", "name": "ISM Non-Manufacturing PMI", "source": "fred", "series_id": "NMFBACTIVITYEI", "group": "surveys", "category": "business", "frequency": "monthly"},
    {"id": "mfg_orders", "name": "Manufacturing New Orders", "source": "fred", "series_id": "AMTMNO", "group": "surveys", "category": "business", "frequency": "monthly"},
    {"id": "durable_goods", "name": "Durable Goods Orders", "source": "fred", "series_id": "DGORDER", "group": "surveys", "category": "business", "frequency": "monthly"},
    {"id": "cem_business_inv", "name": "Business Inventories", "source": "fred", "series_id": "BUSINV", "group": "surveys", "category": "business", "frequency": "monthly"},

    # Consumer
    {"id": "umcsent", "name": "U Mich Consumer Sentiment", "source": "fred", "series_id": "UMCSENT", "group": "consumer", "category": "sentiment", "frequency": "monthly"},
    {"id": "concord", "name": "Conference Board Consumer Confidence", "source": "fred", "series_id": "CSCICP03USM665S", "group": "consumer", "category": "sentiment", "frequency": "monthly"},
    {"id": "rsafs", "name": "Retail Sales", "source": "fred", "series_id": "RSAFS", "group": "consumer", "category": "spending", "frequency": "monthly"},
    {"id": "pce_real", "name": "Real PCE", "source": "fred", "series_id": "PCEC96", "group": "consumer", "category": "spending", "frequency": "monthly"},
    {"id": "pce_goods", "name": "PCE Goods", "source": "fred", "series_id": "DGDSRX1M027SBEA", "group": "consumer", "category": "spending", "frequency": "monthly"},
    {"id": "pce_services", "name": "PCE Services", "source": "fred", "series_id": "DSERSRX1M027SBEA", "group": "consumer", "category": "spending", "frequency": "monthly"},

    # Housing
    {"id": "houst", "name": "Housing Starts", "source": "fred", "series_id": "HOUST", "group": "housing", "category": "construction", "frequency": "monthly"},
    {"id": "permit", "name": "Building Permits", "source": "fred", "series_id": "PERMIT", "group": "housing", "category": "construction", "frequency": "monthly"},
    {"id": "hpinsa", "name": "Case-Shiller Home Price Index", "source": "fred", "series_id": "CSUSHPINSA", "group": "housing", "category": "prices", "frequency": "monthly"},
    {"id": "nhsltot", "name": "New Home Sales", "source": "fred", "series_id": "HSN1F", "group": "housing", "category": "sales", "frequency": "monthly"},
    {"id": "exhoslusm495s", "name": "Existing Home Sales", "source": "fred", "series_id": "EXHOSLUSM495S", "group": "housing", "category": "sales", "frequency": "monthly"},
    {"id": "mortgage30us", "name": "30-Year Mortgage Rate", "source": "fred", "series_id": "MORTGAGE30US", "group": "housing", "category": "rates", "frequency": "weekly"},
    {"id": "mortgage15us", "name": "15-Year Mortgage Rate", "source": "fred", "series_id": "MORTGAGE15US", "group": "housing", "category": "rates", "frequency": "weekly"},

    # Trade
    {"id": "bopgstb", "name": "Trade Balance", "source": "fred", "series_id": "BOPGSTB", "group": "trade", "category": "external", "frequency": "monthly"},
    {"id": "expgs", "name": "Exports of Goods & Services", "source": "fred", "series_id": "EXPGS", "group": "trade", "category": "external", "frequency": "quarterly"},
    {"id": "impgs", "name": "Imports of Goods & Services", "source": "fred", "series_id": "IMPGS", "group": "trade", "category": "external", "frequency": "quarterly"},
]


# =============================================================================
# Category Definitions - Global Economics
# =============================================================================

ECONOMICS_GLOBAL_INDICATORS = [
    # Eurozone
    {"id": "eu_cpi", "name": "Eurozone HICP", "source": "fred", "series_id": "CP0000EZ19M086NEST", "group": "eu_economic", "category": "inflation", "frequency": "monthly"},
    {"id": "eu_unemployment", "name": "Eurozone Unemployment", "source": "fred", "series_id": "LRHUTTTTEZM156S", "group": "eu_economic", "category": "labor", "frequency": "monthly"},
    {"id": "eu_gdp", "name": "Eurozone Real GDP", "source": "fred", "series_id": "CLVMEURSCAB1GQEA19", "group": "eu_economic", "category": "output", "frequency": "quarterly"},

    # UK
    {"id": "uk_cpi", "name": "UK CPI", "source": "fred", "series_id": "CPALTT01GBM659N", "group": "uk_economic", "category": "inflation", "frequency": "monthly"},
    {"id": "uk_unemployment", "name": "UK Unemployment", "source": "fred", "series_id": "LRHUTTTTGBM156S", "group": "uk_economic", "category": "labor", "frequency": "monthly"},

    # Japan
    {"id": "jp_cpi", "name": "Japan CPI", "source": "fred", "series_id": "JPNCPIALLMINMEI", "group": "jp_economic", "category": "inflation", "frequency": "monthly"},
    {"id": "jp_unemployment", "name": "Japan Unemployment", "source": "fred", "series_id": "LRHUTTTTJPM156S", "group": "jp_economic", "category": "labor", "frequency": "monthly"},

    # China
    {"id": "cn_cpi", "name": "China CPI", "source": "fred", "series_id": "CHNCPIALLMINMEI", "group": "cn_economic", "category": "inflation", "frequency": "monthly"},
    {"id": "cn_ppi", "name": "China PPI", "source": "fred", "series_id": "CHNPIEAMP01GYM", "group": "cn_economic", "category": "inflation", "frequency": "monthly"},

    # Canada
    {"id": "ca_cpi", "name": "Canada CPI", "source": "fred", "series_id": "CPALCY01CAM661N", "group": "ca_economic", "category": "inflation", "frequency": "monthly"},
    {"id": "ca_unemployment", "name": "Canada Unemployment", "source": "fred", "series_id": "LRUNTTTTCAM156S", "group": "ca_economic", "category": "labor", "frequency": "monthly"},

    # Global Aggregates
    {"id": "wld_trade", "name": "World Trade Volume", "source": "fred", "series_id": "WDXTEXVA01IXOBM", "group": "global_trade", "category": "external", "frequency": "monthly"},
]


# =============================================================================
# Synthetic Indicator Definitions
# =============================================================================

SYNTHETIC_INDICATORS = [
    # Yield Curve Spreads
    {"id": "spread_10y2y", "name": "10Y-2Y Yield Spread", "source": "synthetic", "formula": "spread", "inputs": ["dgs10", "dgs2"], "group": "yield_curve"},
    {"id": "spread_10y3m", "name": "10Y-3M Yield Spread", "source": "synthetic", "formula": "spread", "inputs": ["dgs10", "dgs3mo"], "group": "yield_curve"},
    {"id": "spread_5y2y", "name": "5Y-2Y Yield Spread", "source": "synthetic", "formula": "spread", "inputs": ["dgs5", "dgs2"], "group": "yield_curve"},
    {"id": "spread_30y10y", "name": "30Y-10Y Yield Spread", "source": "synthetic", "formula": "spread", "inputs": ["dgs30", "dgs10"], "group": "yield_curve"},
    {"id": "spread_2y3m", "name": "2Y-3M Yield Spread", "source": "synthetic", "formula": "spread", "inputs": ["dgs2", "dgs3mo"], "group": "yield_curve"},
    {"id": "spread_10y5y", "name": "10Y-5Y Yield Spread", "source": "synthetic", "formula": "spread", "inputs": ["dgs10", "dgs5"], "group": "yield_curve"},

    # Real Yields (Nominal - Inflation Expectations)
    {"id": "real_yield_5y", "name": "5Y Real Yield", "source": "synthetic", "formula": "spread", "inputs": ["dgs5", "t5yie"], "group": "real_yield"},
    {"id": "real_yield_10y", "name": "10Y Real Yield", "source": "synthetic", "formula": "spread", "inputs": ["dgs10", "t10yie"], "group": "real_yield"},

    # Credit Spreads
    {"id": "credit_aaa", "name": "AAA Credit Spread", "source": "synthetic", "formula": "spread", "inputs": ["aaa_yield", "dgs10"], "group": "credit_spread"},
    {"id": "credit_baa", "name": "BAA Credit Spread", "source": "synthetic", "formula": "spread", "inputs": ["baa_yield", "dgs10"], "group": "credit_spread"},
    {"id": "credit_hy_ig", "name": "HY-IG Spread", "source": "synthetic", "formula": "spread", "inputs": ["bamlh0a0hym2", "bamlc0a0cm"], "group": "credit_spread"},

    # Liquidity Ratios
    {"id": "rrp_m2_ratio", "name": "RRP/M2 Ratio", "source": "synthetic", "formula": "ratio", "inputs": ["rrpontsyd", "m2"], "group": "liquidity_ratio"},
    {"id": "tga_m2_ratio", "name": "TGA/M2 Ratio", "source": "synthetic", "formula": "ratio", "inputs": ["wtregen", "m2"], "group": "liquidity_ratio"},
    {"id": "fed_assets_m2", "name": "Fed Assets/M2", "source": "synthetic", "formula": "ratio", "inputs": ["walcl", "m2"], "group": "liquidity_ratio"},

    # Employment Spreads
    {"id": "employment_spread", "name": "Employment Breadth", "source": "synthetic", "formula": "spread", "inputs": ["unrate", "u6rate"], "group": "employment_spread"},
    {"id": "jolts_unemployed", "name": "Job Openings per Unemployed", "source": "synthetic", "formula": "ratio", "inputs": ["jolts", "unrate"], "group": "employment_spread"},

    # Volatility Term Structure
    {"id": "vix_term_spread", "name": "VIX Term Spread", "source": "synthetic", "formula": "spread", "inputs": ["vix3m", "vix"], "group": "vol_term_structure"},
    {"id": "vix_9d_1m", "name": "VIX 9D-1M Spread", "source": "synthetic", "formula": "spread", "inputs": ["vix", "vix9d"], "group": "vol_term_structure"},

    # Cross-Asset Ratios
    {"id": "spx_gold_ratio", "name": "SPX/Gold Ratio", "source": "synthetic", "formula": "ratio", "inputs": ["spy", "gld"], "group": "cross_asset"},
    {"id": "spx_tlt_ratio", "name": "SPX/TLT Ratio", "source": "synthetic", "formula": "ratio", "inputs": ["spy", "tlt"], "group": "cross_asset"},
    {"id": "copper_gold_ratio", "name": "Copper/Gold Ratio", "source": "synthetic", "formula": "ratio", "inputs": ["hg_f", "gc_f"], "group": "cross_asset"},
    {"id": "oil_gold_ratio", "name": "Oil/Gold Ratio", "source": "synthetic", "formula": "ratio", "inputs": ["cl_f", "gc_f"], "group": "cross_asset"},
    {"id": "russell_spy_ratio", "name": "Russell/SPY Ratio", "source": "synthetic", "formula": "ratio", "inputs": ["iwm", "spy"], "group": "cross_asset"},
    {"id": "qqq_spy_ratio", "name": "QQQ/SPY Ratio", "source": "synthetic", "formula": "ratio", "inputs": ["qqq", "spy"], "group": "cross_asset"},

    # YoY Changes
    {"id": "spy_yoy", "name": "SPY YoY Return", "source": "synthetic", "formula": "yoy", "inputs": ["spy"], "group": "momentum"},
    {"id": "m2_yoy", "name": "M2 YoY Growth", "source": "synthetic", "formula": "yoy", "inputs": ["m2"], "group": "monetary_growth"},
    {"id": "fed_assets_yoy", "name": "Fed Assets YoY", "source": "synthetic", "formula": "yoy", "inputs": ["walcl"], "group": "monetary_growth"},

    # Global Spreads
    {"id": "us_de_10y_spread", "name": "US-Germany 10Y Spread", "source": "synthetic", "formula": "spread", "inputs": ["dgs10", "de_10y"], "group": "global_spread"},
    {"id": "us_jp_10y_spread", "name": "US-Japan 10Y Spread", "source": "synthetic", "formula": "spread", "inputs": ["dgs10", "jp_10y"], "group": "global_spread"},
    {"id": "italy_germany_spread", "name": "Italy-Germany 10Y Spread", "source": "synthetic", "formula": "spread", "inputs": ["it_10y", "de_10y"], "group": "global_spread"},
]


# =============================================================================
# Technical Indicator Auto-Apply Rules
# =============================================================================

TECHNICAL_RULES = {
    "groups_to_apply": [
        "us_equity_index",
        "global_equity_index",
        "msci_index",
        "us_sector",
        "factor",
        "fx_major",
        "fx_index",
        "precious_metals",
        "energy",
        "commodity_etf",
    ],
    "indicators": [
        {"type": "macd", "params": {"fast": 12, "slow": 26, "signal": 9}},
        {"type": "rsi", "params": {"window": 14}},
        {"type": "sma", "params": {"window": 50}, "suffix": "sma50"},
        {"type": "sma", "params": {"window": 100}, "suffix": "sma100"},
        {"type": "sma", "params": {"window": 200}, "suffix": "sma200"},
        {"type": "ema", "params": {"window": 21}, "suffix": "ema21"},
        {"type": "bollinger_upper", "params": {"window": 20, "std_dev": 2}, "suffix": "bb_upper"},
        {"type": "bollinger_lower", "params": {"window": 20, "std_dev": 2}, "suffix": "bb_lower"},
        {"type": "volatility", "params": {"window": 20}, "suffix": "vol20"},
        {"type": "volatility", "params": {"window": 60}, "suffix": "vol60"},
        {"type": "momentum", "params": {"window": 21}, "suffix": "mom1m"},
        {"type": "momentum", "params": {"window": 63}, "suffix": "mom3m"},
        {"type": "momentum", "params": {"window": 126}, "suffix": "mom6m"},
        {"type": "momentum", "params": {"window": 252}, "suffix": "mom12m"},
    ],
    "zscore": {
        "windows": [20, 60, 126],  # Rolling z-score lookback periods
    },
}


# =============================================================================
# YAML Generation Functions
# =============================================================================

def generate_yaml_header(title: str, description: str) -> Dict[str, Any]:
    """Generate standard YAML header."""
    return {
        "version": "2.0.0",
        "generated": datetime.now().strftime("%Y-%m-%d"),
        "title": title,
        "description": description,
    }


def generate_market_global_yaml() -> None:
    """Generate market_global.yaml with equity indices and sectors."""
    data = generate_yaml_header(
        "Market Global Registry",
        "U.S. and global equity indices, sectors, country ETFs, and factor exposures"
    )

    indicators = []
    for ind in MARKET_GLOBAL_INDICATORS:
        entry = {
            "id": ind["id"],
            "name": ind["name"],
            "source": ind["source"],
            "group": ind["group"],
            "frequency": "daily",
        }
        if "ticker" in ind:
            entry["params"] = {"ticker": ind["ticker"]}
        if "asset_class" in ind:
            entry["asset_class"] = ind["asset_class"]
        indicators.append(entry)

    data["indicators"] = indicators

    with open(YAML_DIR / "market_global.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def generate_rates_us_yaml() -> None:
    """Generate rates_us.yaml with U.S. Treasury yields and rates."""
    data = generate_yaml_header(
        "U.S. Rates Registry",
        "U.S. Treasury yields across the full curve, TIPS, breakevens, and policy rates"
    )

    indicators = []
    for ind in RATES_US_INDICATORS:
        entry = {
            "id": ind["id"],
            "name": ind["name"],
            "source": ind["source"],
            "group": ind["group"],
            "frequency": "daily",
        }
        if "series_id" in ind:
            entry["params"] = {"series_id": ind["series_id"]}
        if "ticker" in ind:
            entry["params"] = {"ticker": ind["ticker"]}
        if "category" in ind:
            entry["category"] = ind["category"]
        if "asset_class" in ind:
            entry["asset_class"] = ind["asset_class"]
        indicators.append(entry)

    data["indicators"] = indicators

    with open(YAML_DIR / "rates_us.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def generate_rates_global_yaml() -> None:
    """Generate rates_global.yaml with global sovereign yields."""
    data = generate_yaml_header(
        "Global Rates Registry",
        "Sovereign yields from major economies: Germany, UK, Japan, Italy, Canada, Australia"
    )

    indicators = []
    for ind in RATES_GLOBAL_INDICATORS:
        entry = {
            "id": ind["id"],
            "name": ind["name"],
            "source": ind["source"],
            "group": ind["group"],
            "frequency": "daily" if ind["source"] == "yahoo" else "monthly",
        }
        if "series_id" in ind:
            entry["params"] = {"series_id": ind["series_id"]}
        if "ticker" in ind:
            entry["params"] = {"ticker": ind["ticker"]}
        if "category" in ind:
            entry["category"] = ind["category"]
        if "asset_class" in ind:
            entry["asset_class"] = ind["asset_class"]
        indicators.append(entry)

    data["indicators"] = indicators

    with open(YAML_DIR / "rates_global.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def generate_credit_yaml() -> None:
    """Generate credit.yaml with corporate yields and spreads."""
    data = generate_yaml_header(
        "Credit Markets Registry",
        "Corporate bond yields, credit spreads (AAA, BBB, HY), OAS, and credit ETFs"
    )

    indicators = []
    for ind in CREDIT_INDICATORS:
        entry = {
            "id": ind["id"],
            "name": ind["name"],
            "source": ind["source"],
            "group": ind["group"],
            "frequency": "daily",
        }
        if "series_id" in ind:
            entry["params"] = {"series_id": ind["series_id"]}
        if "ticker" in ind:
            entry["params"] = {"ticker": ind["ticker"]}
        if "category" in ind:
            entry["category"] = ind["category"]
        if "asset_class" in ind:
            entry["asset_class"] = ind["asset_class"]
        indicators.append(entry)

    data["indicators"] = indicators

    with open(YAML_DIR / "credit.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def generate_liquidity_yaml() -> None:
    """Generate liquidity.yaml with Fed balance sheet and liquidity metrics."""
    data = generate_yaml_header(
        "Liquidity Registry",
        "Money supply (M1, M2), Fed balance sheet, RRP, reserves, and financial conditions"
    )

    indicators = []
    for ind in LIQUIDITY_INDICATORS:
        entry = {
            "id": ind["id"],
            "name": ind["name"],
            "source": ind["source"],
            "group": ind["group"],
            "frequency": "weekly" if "weekly" in ind.get("group", "") else "monthly",
        }
        if "series_id" in ind:
            entry["params"] = {"series_id": ind["series_id"]}
        if "category" in ind:
            entry["category"] = ind["category"]
        indicators.append(entry)

    data["indicators"] = indicators

    with open(YAML_DIR / "liquidity.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def generate_fx_yaml() -> None:
    """Generate fx.yaml with currency pairs and FX indicators."""
    data = generate_yaml_header(
        "FX Registry",
        "Major currency pairs, EM currencies, cross rates, and currency ETFs"
    )

    indicators = []
    for ind in FX_INDICATORS:
        entry = {
            "id": ind["id"],
            "name": ind["name"],
            "source": ind["source"],
            "group": ind["group"],
            "frequency": "daily",
        }
        if "ticker" in ind:
            entry["params"] = {"ticker": ind["ticker"]}
        if "asset_class" in ind:
            entry["asset_class"] = ind["asset_class"]
        indicators.append(entry)

    data["indicators"] = indicators

    with open(YAML_DIR / "fx.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def generate_commodities_yaml() -> None:
    """Generate commodities.yaml with oil, metals, and agriculture."""
    data = generate_yaml_header(
        "Commodities Registry",
        "Energy (oil, gas), precious metals, industrial metals, agriculture, and commodity ETFs"
    )

    indicators = []
    for ind in COMMODITY_INDICATORS:
        entry = {
            "id": ind["id"],
            "name": ind["name"],
            "source": ind["source"],
            "group": ind["group"],
            "frequency": "daily",
        }
        if "ticker" in ind:
            entry["params"] = {"ticker": ind["ticker"]}
        if "asset_class" in ind:
            entry["asset_class"] = ind["asset_class"]
        indicators.append(entry)

    data["indicators"] = indicators

    with open(YAML_DIR / "commodities.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def generate_volatility_yaml() -> None:
    """Generate volatility.yaml with VIX variants and vol products."""
    data = generate_yaml_header(
        "Volatility Registry",
        "VIX family (9D, 3M, 6M), sector vol, MOVE index, and volatility ETFs"
    )

    indicators = []
    for ind in VOLATILITY_INDICATORS:
        entry = {
            "id": ind["id"],
            "name": ind["name"],
            "source": ind["source"],
            "group": ind["group"],
            "frequency": "daily",
        }
        if "ticker" in ind:
            entry["params"] = {"ticker": ind["ticker"]}
        if "series_id" in ind:
            entry["params"] = {"series_id": ind["series_id"]}
        if "asset_class" in ind:
            entry["asset_class"] = ind["asset_class"]
        if "category" in ind:
            entry["category"] = ind["category"]
        indicators.append(entry)

    data["indicators"] = indicators

    with open(YAML_DIR / "volatility.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def generate_economics_us_yaml() -> None:
    """Generate economics_us.yaml with U.S. economic indicators."""
    data = generate_yaml_header(
        "U.S. Economics Registry",
        "CPI, PCE, employment, GDP, housing, consumer sentiment, ISM, and trade data"
    )

    indicators = []
    for ind in ECONOMICS_US_INDICATORS:
        entry = {
            "id": ind["id"],
            "name": ind["name"],
            "source": ind["source"],
            "group": ind["group"],
            "frequency": ind.get("frequency", "monthly"),
        }
        if "series_id" in ind:
            entry["params"] = {"series_id": ind["series_id"]}
        if "category" in ind:
            entry["category"] = ind["category"]
        indicators.append(entry)

    data["indicators"] = indicators

    with open(YAML_DIR / "economics_us.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def generate_economics_global_yaml() -> None:
    """Generate economics_global.yaml with global economic indicators."""
    data = generate_yaml_header(
        "Global Economics Registry",
        "CPI, unemployment, and GDP for Eurozone, UK, Japan, China, Canada"
    )

    indicators = []
    for ind in ECONOMICS_GLOBAL_INDICATORS:
        entry = {
            "id": ind["id"],
            "name": ind["name"],
            "source": ind["source"],
            "group": ind["group"],
            "frequency": ind.get("frequency", "monthly"),
        }
        if "series_id" in ind:
            entry["params"] = {"series_id": ind["series_id"]}
        if "category" in ind:
            entry["category"] = ind["category"]
        indicators.append(entry)

    data["indicators"] = indicators

    with open(YAML_DIR / "economics_global.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def generate_synthetic_yaml() -> None:
    """Generate synthetic.yaml with derived indicators."""
    data = generate_yaml_header(
        "Synthetic Indicators Registry",
        "Yield curve spreads, real yields, credit spreads, liquidity ratios, cross-asset ratios"
    )

    indicators = []
    for ind in SYNTHETIC_INDICATORS:
        entry = {
            "id": ind["id"],
            "name": ind["name"],
            "source": "synthetic",
            "group": ind["group"],
            "frequency": "daily",
            "params": {
                "formula": ind["formula"],
                "inputs": ind["inputs"],
            },
        }
        indicators.append(entry)

    data["indicators"] = indicators

    with open(YAML_DIR / "synthetic.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def generate_technical_yaml() -> None:
    """Generate technical.yaml with auto-apply rules for technicals."""
    data = generate_yaml_header(
        "Technical Indicators Registry",
        "MACD, RSI, SMA, EMA, Bollinger Bands, volatility, momentum - auto-applied by group"
    )

    data["auto_apply_rules"] = TECHNICAL_RULES

    # Generate explicit technical indicators for key assets
    explicit_technicals = []
    key_assets = ["spy", "qqq", "iwm", "dxy", "gc_f", "cl_f", "eurusd"]

    for asset in key_assets:
        for tech in TECHNICAL_RULES["indicators"]:
            suffix = tech.get("suffix", tech["type"])
            entry = {
                "id": f"{asset}_{suffix}",
                "name": f"{asset.upper()} {tech['type'].upper()}",
                "source": "technical",
                "group": "technical",
                "frequency": "daily",
                "params": {
                    "base": asset,
                    "type": tech["type"],
                    **tech["params"],
                },
            }
            explicit_technicals.append(entry)

    data["indicators"] = explicit_technicals

    with open(YAML_DIR / "technical.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def count_indicators() -> Dict[str, int]:
    """Count indicators by category."""
    counts = {
        "market_global": len(MARKET_GLOBAL_INDICATORS),
        "rates_us": len(RATES_US_INDICATORS),
        "rates_global": len(RATES_GLOBAL_INDICATORS),
        "credit": len(CREDIT_INDICATORS),
        "liquidity": len(LIQUIDITY_INDICATORS),
        "fx": len(FX_INDICATORS),
        "commodities": len(COMMODITY_INDICATORS),
        "volatility": len(VOLATILITY_INDICATORS),
        "economics_us": len(ECONOMICS_US_INDICATORS),
        "economics_global": len(ECONOMICS_GLOBAL_INDICATORS),
        "synthetic": len(SYNTHETIC_INDICATORS),
    }

    # Technical indicators: key_assets * tech_types
    key_assets = ["spy", "qqq", "iwm", "dxy", "gc_f", "cl_f", "eurusd"]
    counts["technical_explicit"] = len(key_assets) * len(TECHNICAL_RULES["indicators"])

    counts["total"] = sum(counts.values())

    return counts


def print_summary() -> None:
    """Print generation summary."""
    counts = count_indicators()

    print("\n" + "=" * 60)
    print("PRISM REGISTRY GENERATOR - FULL INSTITUTIONAL PACK")
    print("=" * 60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {YAML_DIR}")
    print()
    print("Indicator Counts by Category:")
    print("-" * 40)

    for category, count in counts.items():
        if category != "total":
            print(f"  {category:25s}: {count:4d}")

    print("-" * 40)
    print(f"  {'TOTAL':25s}: {counts['total']:4d}")
    print()

    # Source breakdown
    all_indicators = (
        MARKET_GLOBAL_INDICATORS + RATES_US_INDICATORS + RATES_GLOBAL_INDICATORS +
        CREDIT_INDICATORS + LIQUIDITY_INDICATORS + FX_INDICATORS +
        COMMODITY_INDICATORS + VOLATILITY_INDICATORS + ECONOMICS_US_INDICATORS +
        ECONOMICS_GLOBAL_INDICATORS
    )

    source_counts = {}
    for ind in all_indicators:
        src = ind.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    source_counts["synthetic"] = len(SYNTHETIC_INDICATORS)
    source_counts["technical"] = counts["technical_explicit"]

    print("Indicator Counts by Source:")
    print("-" * 40)
    for source, count in sorted(source_counts.items()):
        print(f"  {source:25s}: {count:4d}")
    print()


def main() -> None:
    """Generate all registry YAML files."""
    print("Generating PRISM Full Institutional Pack registry files...")

    # Generate all YAML files
    generate_market_global_yaml()
    print("  ✓ market_global.yaml")

    generate_rates_us_yaml()
    print("  ✓ rates_us.yaml")

    generate_rates_global_yaml()
    print("  ✓ rates_global.yaml")

    generate_credit_yaml()
    print("  ✓ credit.yaml")

    generate_liquidity_yaml()
    print("  ✓ liquidity.yaml")

    generate_fx_yaml()
    print("  ✓ fx.yaml")

    generate_commodities_yaml()
    print("  ✓ commodities.yaml")

    generate_volatility_yaml()
    print("  ✓ volatility.yaml")

    generate_economics_us_yaml()
    print("  ✓ economics_us.yaml")

    generate_economics_global_yaml()
    print("  ✓ economics_global.yaml")

    generate_synthetic_yaml()
    print("  ✓ synthetic.yaml")

    generate_technical_yaml()
    print("  ✓ technical.yaml")

    print_summary()


if __name__ == "__main__":
    main()
