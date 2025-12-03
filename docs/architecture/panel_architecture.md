# Panel Architecture

This document describes how analysis-ready panels are built from heterogeneous time series data in PRISM.

## Overview

A **panel** is a consolidated DataFrame containing multiple indicators aligned on a common date index. Panels are the primary input format for the PRISM analysis engine.

## Panel Format

### Structure

```
| date       | indicator_1 | indicator_2 | indicator_3 | ... |
|------------|-------------|-------------|-------------|-----|
| 2020-01-01 | 100.5       | 2.5         | 3.2         | ... |
| 2020-01-02 | 101.2       | 2.5         | 3.1         | ... |
| 2020-01-03 | 99.8        | 2.6         | 3.3         | ... |
```

### Requirements

- **Date column**: First column, datetime-parseable
- **Indicator columns**: Numeric values (float)
- **Minimum rows**: 100+ data points for meaningful analysis
- **Missing values**: Handled via forward-fill or imputation

### File Location

Panels are stored in `data/panels/`:

```
data/panels/
├── master_panel.csv          # Primary US market + economic panel
├── master_panel_climate.csv  # Climate indicators (if available)
└── master_panel_test.csv     # Synthetic test data
```

## Panel Building Process

### Step 1: Data Fetching

Data is fetched from multiple sources with different frequencies:

```python
from 01_fetch.fetcher_yahoo import YahooFetcher
from 01_fetch.fetcher_fred import FREDFetcher

yahoo = YahooFetcher()
fred = FREDFetcher(api_key="...")

# Market data (daily)
market = yahoo.fetch_batch(['SPY', 'QQQ', 'TLT'])

# Economic data (mixed frequency)
econ = fred.fetch_batch(['DGS10', 'CPIAUCSL', 'UNRATE'])
```

### Step 2: Frequency Alignment

The `DataAligner` class handles frequency mismatches:

```python
from 03_cleaning.alignment import DataAligner

aligner = DataAligner()

# Align all to daily frequency
aligned_market = aligner.align_to_frequency(market, 'D')
aligned_econ = aligner.align_to_frequency(econ, 'D', fill_method='ffill')
```

**Alignment Methods**:

| Method | Description |
|--------|-------------|
| `last` | Use last value in period (default) |
| `first` | Use first value in period |
| `mean` | Average of values in period |
| `sum` | Sum of values in period |

**Fill Methods**:

| Method | Description |
|--------|-------------|
| `ffill` | Forward-fill from last known value |
| `bfill` | Backward-fill from next known value |
| `None` | Leave as NaN |

### Step 3: Series Selection

Not all fetched series may be needed. Selection criteria:

1. **Relevance**: Indicators related to analysis domain
2. **Coverage**: Sufficient data points in target period
3. **Quality**: Minimal missing values
4. **Uniqueness**: Avoid redundant indicators

### Step 4: Date Range Alignment

Find common date range across all series:

```python
# Get common date range
start, end = aligner.get_common_date_range([df1, df2, df3])

# Or specify explicit range
panel = aligner.align_date_range(combined, start_date='2010-01-01')
```

### Step 5: Consolidation

Merge all series on date index:

```python
panel = aligner.create_master_panel(
    dfs={**market, **econ},
    target_freq='D',
    start_date='2010-01-01',
    end_date='2024-01-01',
    agg_method='last'
)
```

### Step 6: NaN Handling

After merging, fill remaining gaps:

```python
# Forward-fill, then backward-fill
panel = panel.ffill().bfill()

# Or use advanced strategies
from 03_cleaning.nan_strategies import LinearInterpolateStrategy

strategy = LinearInterpolateStrategy()
panel = strategy.apply(panel)
```

### Step 7: Export

Save as CSV:

```python
panel.to_csv('data/panels/master_panel.csv', index=False)
```

## DataAligner API

### Initialization

```python
aligner = DataAligner(date_col='date')
```

### Methods

#### align_to_frequency

Align DataFrame to target frequency:

```python
aligned = aligner.align_to_frequency(
    df,
    target_freq='D',      # D, W, M, Q, Y
    agg_method='last',    # last, first, mean, sum
    fill_method='ffill'   # ffill, bfill, None
)
```

#### align_multiple

Align and merge multiple DataFrames:

```python
combined = aligner.align_multiple(
    dfs={'spy': df1, 'gdp': df2, 'cpi': df3},
    target_freq='D',
    agg_method='last'
)
```

#### align_date_range

Filter to specific date range:

```python
filtered = aligner.align_date_range(
    df,
    start_date='2020-01-01',
    end_date='2024-01-01',
    method='inner'  # inner or outer
)
```

#### get_common_date_range

Find overlapping date range:

```python
start, end = aligner.get_common_date_range([df1, df2, df3])
```

#### create_master_panel

Full panel building pipeline:

```python
panel = aligner.create_master_panel(
    dfs=all_data,
    target_freq='D',
    start_date='2010-01-01',
    end_date='2024-01-01',
    agg_method='last'
)
```

## Frequency Mapping

Internal frequency codes:

| Code | Description | Pandas Freq |
|------|-------------|-------------|
| D | Daily | D |
| W | Weekly (Friday) | W-FRI |
| M | Monthly (End) | ME |
| Q | Quarterly (End) | QE |
| Y | Yearly (End) | YE |

## Master Panel Structure

The default master panel (`master_panel.csv`) contains:

### Column Structure

```
date,               # Index column
dgs10,              # 10-Year Treasury
dgs2,               # 2-Year Treasury
dgs3mo,             # 3-Month Treasury
t10y2y,             # 10Y-2Y Spread
t10y3m,             # 10Y-3M Spread
cpiaucsl,           # CPI
cpilfesl,           # Core CPI
ppiaco,             # PPI
unrate,             # Unemployment
payems,             # Payrolls
indpro,             # Industrial Production
houst,              # Housing Starts
permit,             # Building Permits
m2sl,               # M2 Money Supply
walcl,              # Fed Balance Sheet
anfci,              # Chicago Fed NFCI
nfci,               # National Financial Conditions
spy_spy,            # S&P 500 ETF
qqq_qqq,            # NASDAQ ETF
iwm_iwm,            # Russell 2000 ETF
gld_gld,            # Gold ETF
slv_slv,            # Silver ETF
uso_uso,            # Oil ETF
bnd_bnd,            # Total Bond ETF
tlt_tlt,            # 20+ Year Treasury ETF
shy_shy,            # Short-term Treasury ETF
ief_ief,            # 7-10 Year Treasury ETF
tip_tip,            # TIPS ETF
lqd_lqd,            # IG Corporate ETF
hyg_hyg,            # HY Corporate ETF
xlu_xlu             # Utilities Sector ETF
```

### Date Range

- **Start**: 1913-01-01 (earliest available data)
- **End**: Present
- **Note**: Many indicators have NaN before their inception dates

## Usage in Analysis Engine

The analysis engine expects panels in this format:

```python
from 05_engine.orchestration.indicator_engine import IndicatorEngine

# Load panel
panel = pd.read_csv('data/panels/master_panel.csv', parse_dates=['date'])

# Run analysis
engine = IndicatorEngine()
results = engine.analyze(panel, mode='full')
```

### Panel Validation

Before analysis, panels are validated for:

1. **Date column exists**: Must have 'date' column
2. **Numeric columns**: All non-date columns must be numeric
3. **Minimum rows**: At least 100 data points
4. **No empty columns**: Each indicator has some valid values

## Best Practices

### Panel Building

1. **Start with clean data**: Apply cleaning before panel building
2. **Use consistent frequency**: Align all to same frequency
3. **Handle missing values**: Don't leave NaN gaps
4. **Document indicators**: Track what each column represents

### Panel Maintenance

1. **Version panels**: Use timestamps in filenames for versioning
2. **Validate after updates**: Run validation after adding data
3. **Track metadata**: Maintain documentation of indicator sources
4. **Test with subset**: Build small test panels first

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Misaligned dates | Different start dates | Use common date range |
| NaN gaps | Missing data | Forward-fill or interpolate |
| Duplicate columns | Same ticker from multiple sources | Use unique column names |
| Wrong frequency | Mixed frequencies | Align to target frequency |

## Example: Building a Custom Panel

```python
import pandas as pd
from 01_fetch.fetcher_yahoo import YahooFetcher
from 01_fetch.fetcher_fred import FREDFetcher
from 03_cleaning.alignment import DataAligner

# 1. Fetch data
yahoo = YahooFetcher()
fred = FREDFetcher(api_key="...")

market = yahoo.fetch_batch(['SPY', 'TLT', 'GLD'])
econ = fred.fetch_batch(['DGS10', 'CPIAUCSL'])

# 2. Create aligner
aligner = DataAligner()

# 3. Build panel
all_data = {**market, **econ}
panel = aligner.create_master_panel(
    dfs=all_data,
    target_freq='D',
    start_date='2015-01-01'
)

# 4. Handle remaining NaN
panel = panel.ffill().bfill()

# 5. Validate
assert 'date' in panel.columns
assert len(panel) >= 100
assert panel.select_dtypes(include='number').notna().any().all()

# 6. Save
panel.to_csv('data/panels/custom_panel.csv', index=False)
print(f"Panel created: {len(panel)} rows, {len(panel.columns)} columns")
```
