# PRISM Architecture Documentation

This documentation provides a comprehensive overview of the PRISM data pipeline system, designed for multi-domain time series analysis.

## Documentation Index

| Document | Description |
|----------|-------------|
| [Data Pipeline](prism_data_pipeline.md) | Overview of market vs economic pipelines, data flow, and processing stages |
| [Registry System](registry_system.md) | Schema and usage of metric and source registries |
| [Database Schema](db_schema.md) | SQLite schema, tables, constraints, and Python API |
| [Panel Architecture](panel_architecture.md) | Panel building, series alignment, and master panel construction |

## Quick Start

### Understanding the System

PRISM (Probabilistic Risk Intelligence System for Markets) is a modular data pipeline that:

1. **Fetches** data from multiple sources (FRED, Yahoo Finance, Climate APIs)
2. **Cleans** and normalizes the data (date alignment, NaN handling, outlier detection)
3. **Stores** data in SQLite with a flexible schema supporting multiple domains
4. **Builds** analysis-ready panels from heterogeneous time series
5. **Analyzes** data through 14+ analytical lenses
6. **Generates** consensus rankings and reports

### Key Concepts

- **System**: A domain classification (finance, climate, chemistry, etc.)
- **Indicator**: A single time series (e.g., SPY, GDP, CPI)
- **Panel**: A consolidated DataFrame with multiple indicators aligned on a common date index
- **Lens**: An analytical perspective that ranks indicators by importance

### Directory Structure

```
prism-engine/
├── 01_fetch/           # Data fetchers (Yahoo, FRED, Climate, Custom)
├── 03_cleaning/        # Data cleaning & normalization
├── 05_engine/          # Analysis engine with 14+ lenses
│   ├── lenses/         # Individual lens implementations
│   └── orchestration/  # Lens coordination & consensus
├── data/
│   ├── sql/            # SQLite database schema & API
│   ├── panels/         # Analysis-ready CSV panels
│   ├── raw/            # Fetched raw data
│   └── registry/       # Metric registry JSON
├── tests/              # Test suite
└── docs/               # This documentation
```

## For AI Assistants

When working with this codebase:

1. **Registry First**: Check `data/registry/metric_registry.json` for available indicators
2. **Schema Awareness**: Review `data/sql/prism_schema.sql` before database operations
3. **Panel Structure**: Panels have date as first column, indicators as remaining columns
4. **System Validation**: Only use valid system types: finance, climate, chemistry, anthropology, biology, physics
5. **Frequency Handling**: Economic data is often monthly/quarterly; market data is daily

## Version History

- **v1.0** - Initial documentation set for PR #7
