# PRISM Data Pipeline

This document describes the end-to-end data flow in the PRISM system, from data fetching through panel building and analysis.

## Overview

The PRISM pipeline processes two primary data streams:

1. **Market Pipeline**: High-frequency trading data from Yahoo Finance
2. **Economic Pipeline**: Mixed-frequency macroeconomic data from FRED

Both pipelines converge into a unified panel format for analysis.

## Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA FETCH LAYER (01_fetch)                  │
├─────────────────────────────────────────────────────────────────┤
│  Yahoo Fetcher → OHLCV Data          FRED Fetcher → Macro Data  │
│  (Daily trading prices)              (Mixed frequencies)        │
│         ↓                                   ↓                    │
│  {ticker: DataFrame}                {series: DataFrame}         │
└──────────────┬──────────────────────────────┬────────────────────┘
               │                              │
               ↓                              ↓
┌──────────────────────────────────────────────────────────────────┐
│                  CLEANING LAYER (03_cleaning)                    │
├──────────────────────────────────────────────────────────────────┤
│  Sanitize:                  Align Frequencies:                   │
│  • Lowercase columns        • Resample to daily                  │
│  • Parse dates              • Forward-fill gaps                  │
│  • Standardize format       • Merge on date index                │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ↓
┌──────────────────────────────────────────────────────────────────┐
│              PANEL CONSOLIDATION (data/panels)                   │
├──────────────────────────────────────────────────────────────────┤
│  master_panel.csv: 32+ indicators × N days                      │
│  • Date index (first column)                                     │
│  • All indicators aligned on common dates                        │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ↓
┌──────────────────────────────────────────────────────────────────┐
│                  ANALYSIS ENGINE (05_engine)                     │
├──────────────────────────────────────────────────────────────────┤
│  14 Lenses → Rankings → Consensus → Reports                     │
└──────────────────────────────────────────────────────────────────┘
```

## Market vs Economic Pipeline

### Market Data Pipeline

**Source**: Yahoo Finance API via `yfinance` library

**Characteristics**:
- Daily frequency (trading days only)
- OHLCV data (Open, High, Low, Close, Volume)
- Uses Adjusted Close for splits/dividends
- No weekend/holiday data points

**Indicators**:
- Equity indices: ^GSPC (S&P 500), ^VIX
- Sector ETFs: XLF, XLK, XLE, XLV, XLI, XLU
- Commodities: GC=F (Gold), CL=F (Oil)
- Bonds: TLT, IEF, HYG, LQD

**Processing**:
```python
# Yahoo fetcher output format
{
    "date": datetime,
    "spy_close": float,      # Adjusted close
    "spy_open": float,
    "spy_high": float,
    "spy_low": float,
    "spy_volume": int
}
```

### Economic Data Pipeline

**Source**: Federal Reserve Economic Data (FRED) API

**Characteristics**:
- Mixed frequencies: Daily, Monthly, Quarterly
- Single value per date (no OHLCV)
- Gaps based on release schedule
- Forward-fill required for alignment

**Indicators**:
- Interest Rates (Daily): DGS10, DGS2, DFF
- Inflation (Monthly): CPIAUCSL, CPILFESL
- Employment (Monthly): UNRATE, PAYEMS
- Activity (Monthly): INDPRO, HOUST
- Money Supply (Weekly): M2SL, WALCL

**Processing**:
```python
# FRED fetcher output format
{
    "date": datetime,
    "value": float
}
```

## Registry Role

The registry (`data/registry/metric_registry.json`) defines available indicators:

```json
{
    "key": "dgs10",           // Internal identifier
    "source": "fred",         // Data source
    "ticker": "DGS10"         // Source-specific ID
}
```

See [Registry System](registry_system.md) for details.

## Database Storage

Fetched data can be stored in SQLite for persistence:

1. **Indicators table**: Metadata (name, system, frequency, source)
2. **Indicator values table**: Time series data (indicator_id, date, value)

See [Database Schema](db_schema.md) for details.

## Panel Building

The `DataAligner` class handles frequency alignment and panel construction:

1. **Fetch** data from each source
2. **Align** all series to target frequency (default: daily)
3. **Merge** on date index with outer join
4. **Fill** missing values (forward-fill, then backward-fill)
5. **Export** as CSV to `data/panels/`

See [Panel Architecture](panel_architecture.md) for details.

## Data Quality

The cleaning layer (`03_cleaning/`) provides:

- **NaN Strategies**: Forward-fill, backward-fill, linear interpolation, spline, Kalman
- **Outlier Detection**: Z-score, IQR, rolling window methods
- **NaN Analysis**: Pattern detection and imputation recommendations

## Configuration

Data sources are configured in YAML files:

- `01_fetch/configs/financial_sources.yaml`: Market and economic sources
- `01_fetch/configs/climate_sources.yaml`: Climate data sources

```yaml
# Example from financial_sources.yaml
sources:
  fred:
    - ticker: DGS10
      name: 10-Year Treasury Yield
      category: rates
  yahoo:
    - ticker: ^GSPC
      name: S&P 500
      category: equities

settings:
  start_date: "2000-01-01"
  frequency: daily
```

## Error Handling

Each fetcher implements:
- **Retry logic**: Configurable attempts with exponential backoff
- **Validation**: Response validation before processing
- **Checkpointing**: Save intermediate results to `checkpoints/` directory
- **Logging**: Comprehensive logging of fetch operations

## Usage Example

```python
from 01_fetch.fetcher_yahoo import YahooFetcher
from 01_fetch.fetcher_fred import FREDFetcher
from 03_cleaning.alignment import DataAligner

# Fetch data
yahoo = YahooFetcher()
fred = FREDFetcher(api_key="your-key")

market_data = yahoo.fetch_batch(['SPY', 'QQQ', 'TLT'])
econ_data = fred.fetch_batch(['DGS10', 'CPIAUCSL', 'UNRATE'])

# Align and build panel
aligner = DataAligner()
panel = aligner.create_master_panel(
    {**market_data, **econ_data},
    target_freq='D',
    start_date='2010-01-01'
)

# Save panel
panel.to_csv('data/panels/master_panel.csv', index=False)
```
