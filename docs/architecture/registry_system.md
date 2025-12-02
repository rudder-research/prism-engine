# Registry System

This document describes the registry configuration files that define available data sources and indicators in PRISM.

## Overview

PRISM uses two types of configuration:

1. **Metric Registry** (`data/registry/metric_registry.json`): JSON array of indicator definitions
2. **Source Configs** (`01_fetch/configs/*.yaml`): YAML files defining fetcher configurations

## Metric Registry

### Location

```
data/registry/metric_registry.json
```

### Schema

The metric registry is a JSON array where each entry defines an indicator:

```json
[
    {
        "key": "string",      // Required: Unique internal identifier
        "source": "string",   // Required: Data source ("fred" | "yahoo")
        "ticker": "string"    // Required: Source-specific identifier
    }
]
```

### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `key` | string | Yes | Unique internal identifier used in panels (e.g., "dgs10", "spy") |
| `source` | string | Yes | Data source type: "fred" or "yahoo" |
| `ticker` | string | Yes | Source-specific ticker/series ID (e.g., "DGS10", "SPY") |

### Example Entries

```json
[
    {
        "key": "dgs10",
        "source": "fred",
        "ticker": "DGS10"
    },
    {
        "key": "cpi",
        "source": "fred",
        "ticker": "CPIAUCSL"
    },
    {
        "key": "spy",
        "source": "yahoo",
        "ticker": "SPY"
    },
    {
        "key": "vix",
        "source": "yahoo",
        "ticker": "VIX"
    }
]
```

### Current Indicators

The registry includes 35+ indicators across categories:

**FRED Indicators (Economic)**:
| Key | Ticker | Description |
|-----|--------|-------------|
| dgs10 | DGS10 | 10-Year Treasury Yield |
| dgs2 | DGS2 | 2-Year Treasury Yield |
| dgs3mo | DGS3MO | 3-Month Treasury Yield |
| t10y2y | T10Y2Y | 10Y-2Y Spread |
| t10y3m | T10Y3M | 10Y-3M Spread |
| cpi | CPIAUCSL | Consumer Price Index |
| cpi_core | CPILFESL | Core CPI |
| ppi | PPIACO | Producer Price Index |
| unrate | UNRATE | Unemployment Rate |
| payrolls | PAYEMS | Nonfarm Payrolls |
| industrial_production | INDPRO | Industrial Production |
| housing_starts | HOUST | Housing Starts |
| permits | PERMIT | Building Permits |
| m2 | M2SL | M2 Money Supply |
| fed_balance_sheet | WALCL | Fed Balance Sheet |
| anfci | ANFCI | Chicago Fed NFCI |
| nfci | NFCI | National Financial Conditions |

**Yahoo Indicators (Market)**:
| Key | Ticker | Description |
|-----|--------|-------------|
| spy | SPY | S&P 500 ETF |
| qqq | QQQ | NASDAQ 100 ETF |
| iwm | IWM | Russell 2000 ETF |
| vix | VIX | Volatility Index |
| gld | GLD | Gold ETF |
| slv | SLV | Silver ETF |
| uso | USO | Oil ETF |
| bnd | BND | Total Bond Market |
| tlt | TLT | 20+ Year Treasury |
| shy | SHY | 1-3 Year Treasury |
| ief | IEF | 7-10 Year Treasury |
| tip | TIP | TIPS ETF |
| lqd | LQD | Investment Grade Corporate |
| hyg | HYG | High Yield Corporate |
| xlu | XLU | Utilities Sector |

## Source Configuration Files

### Financial Sources

**Location**: `01_fetch/configs/financial_sources.yaml`

**Schema**:
```yaml
sources:
  fred:
    - ticker: string        # FRED series ID
      name: string          # Human-readable name
      category: string      # Category (rates, inflation, employment, etc.)

  yahoo:
    - ticker: string        # Yahoo ticker symbol
      name: string          # Human-readable name
      category: string      # Category (equities, volatility, sectors, etc.)

settings:
  start_date: "YYYY-MM-DD"  # Default fetch start date
  end_date: null            # null = today
  frequency: string         # daily, weekly, monthly
  retry_attempts: int       # Number of retry attempts
  retry_delay_seconds: int  # Delay between retries
```

**Categories**:

FRED Categories:
- `rates`: Interest rates and spreads
- `inflation`: Price indices (CPI, PCE, PPI)
- `employment`: Labor market indicators
- `activity`: Economic activity (GDP, Industrial Production)
- `money`: Money supply and monetary base
- `credit`: Credit spreads
- `housing`: Housing market indicators
- `consumer`: Consumer sentiment and spending

Yahoo Categories:
- `equities`: Stock indices and ETFs
- `volatility`: VIX and volatility products
- `sectors`: Sector ETFs (XLF, XLK, XLE, etc.)
- `commodities`: Commodity futures and ETFs
- `currency`: Dollar index
- `bonds`: Bond ETFs

### Climate Sources

**Location**: `01_fetch/configs/climate_sources.yaml`

**Schema**:
```yaml
sources:
  - id: string              # Source identifier
    name: string            # Human-readable name
    url: string             # Data URL
    frequency: string       # Data frequency

settings:
  start_date: "YYYY-MM-DD"
  frequency: monthly
```

## Validation Rules

When loading registries, the following validations apply:

### Required Fields
- All entries must have `key`, `source`, and `ticker`
- Keys must be unique across the registry

### Source Validation
- `source` must be one of: "fred", "yahoo", "climate", "custom"
- Each source has specific ticker format requirements

### Error Examples

**Missing Required Field**:
```json
{
    "key": "spy",
    "source": "yahoo"
    // ERROR: missing "ticker" field
}
```

**Duplicate Key**:
```json
[
    {"key": "spy", "source": "yahoo", "ticker": "SPY"},
    {"key": "spy", "source": "yahoo", "ticker": "SPY"}  // ERROR: duplicate key
]
```

**Invalid Source**:
```json
{
    "key": "test",
    "source": "invalid",  // ERROR: unknown source
    "ticker": "TEST"
}
```

## Usage in Code

### Loading the Registry

```python
import json
from pathlib import Path

# Load registry
registry_path = Path("data/registry/metric_registry.json")
with open(registry_path) as f:
    registry = json.load(f)

# Access entries
for entry in registry:
    print(f"{entry['key']}: {entry['source']}/{entry['ticker']}")
```

### Filtering by Source

```python
# Get all FRED indicators
fred_indicators = [e for e in registry if e['source'] == 'fred']

# Get all Yahoo indicators
yahoo_indicators = [e for e in registry if e['source'] == 'yahoo']
```

### Validating Registry

```python
def validate_registry(registry):
    """Validate registry entries."""
    required_fields = ['key', 'source', 'ticker']
    valid_sources = ['fred', 'yahoo', 'climate', 'custom']
    seen_keys = set()

    for i, entry in enumerate(registry):
        # Check required fields
        for field in required_fields:
            if field not in entry:
                raise ValueError(f"Entry {i}: missing required field '{field}'")

        # Check source
        if entry['source'] not in valid_sources:
            raise ValueError(f"Entry {i}: invalid source '{entry['source']}'")

        # Check uniqueness
        if entry['key'] in seen_keys:
            raise ValueError(f"Entry {i}: duplicate key '{entry['key']}'")
        seen_keys.add(entry['key'])

    return True
```

## Extending the Registry

### Adding a New Indicator

1. Identify the source and ticker
2. Add entry to `metric_registry.json`:
   ```json
   {
       "key": "new_indicator",
       "source": "fred",
       "ticker": "NEW_SERIES_ID"
   }
   ```
3. Optionally add to source config for batch fetching

### Adding a New Source

1. Create a new fetcher class extending `BaseFetcher`
2. Add source type to validation
3. Create corresponding config YAML
4. Add entries to registry with new source type
