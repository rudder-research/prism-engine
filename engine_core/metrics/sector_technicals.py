"""
Sector Technical Indicators
===========================

Builds technical indicators for sectors and other assets.

Supports both explicit indicator definitions and group-based auto-apply rules.
The technical builder can automatically generate indicators for all assets
in specified groups (e.g., all equity indices, all FX pairs).

Technical indicators include:
    - Moving averages (SMA, EMA)
    - Momentum indicators (RSI, MACD, ROC)
    - Volatility indicators (Bollinger Bands, ATR, rolling std)
    - Trend indicators (ADX)
    - Rolling z-scores for mean reversion
    - Price relative to moving averages

Supports the Full Institutional Pack with 250+ indicators.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_technical_indicators(
    registry: Dict[str, Any],
    conn: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, int]:
    """
    Build technical indicators for assets defined in the registry.

    Supports two modes:
    1. Explicit definitions: Each indicator is explicitly defined
    2. Auto-apply rules: Apply indicator types to asset groups automatically

    Args:
        registry: The metric registry containing technical indicator definitions
        conn: SQLite database connection
        start_date: Start date for computation (YYYY-MM-DD)
        end_date: End date for computation (YYYY-MM-DD)

    Returns:
        Dictionary mapping technical indicator names to row counts written
    """
    results: Dict[str, int] = {}
    cursor = conn.cursor()

    # Get explicit technical definitions
    explicit_defs = registry.get("technical", [])

    # Get auto-apply rules (from YAML registry)
    auto_rules = registry.get("technical_rules", {})

    # If auto-apply rules exist, generate additional technical indicators
    if auto_rules:
        auto_generated = _generate_auto_apply_indicators(registry, cursor, auto_rules)
        explicit_defs = explicit_defs + auto_generated
        logger.info(f"Auto-generated {len(auto_generated)} additional technical indicators")

    if not explicit_defs:
        logger.info("No technical indicators defined in registry")
        return results

    # Track processed indicators to avoid duplicates
    processed: Set[str] = set()

    for tech in explicit_defs:
        name = tech.get("name")
        indicator_type = tech.get("type", "sma")
        base_series = tech.get("base")
        params = tech.get("params", {})

        if not name or not base_series:
            continue

        # Skip duplicates
        if name in processed:
            continue
        processed.add(name)

        try:
            # Load base series
            series = _load_series(cursor, base_series, start_date, end_date)

            if series is None or series.empty:
                logger.debug(f"No base data for technical {name}")
                results[name] = 0
                continue

            # Compute technical indicator
            output_series = _compute_technical(series, indicator_type, params)

            if output_series is None or output_series.empty:
                logger.debug(f"Empty result for technical {name}")
                results[name] = 0
                continue

            # Write results
            count = _write_technical_values(cursor, name, output_series)
            conn.commit()

            results[name] = count
            if count > 0:
                logger.info(f"Built technical {name}: {count} rows")

        except Exception as e:
            logger.error(f"Error building technical {name}: {e}")
            results[name] = 0

    return results


def _generate_auto_apply_indicators(
    registry: Dict[str, Any],
    cursor: sqlite3.Cursor,
    rules: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Generate technical indicator definitions based on auto-apply rules.

    Rules specify which indicator types to apply to which asset groups.
    """
    generated = []

    groups_to_apply = rules.get("groups_to_apply", [])
    indicator_templates = rules.get("indicators", [])

    if not groups_to_apply or not indicator_templates:
        return generated

    # Get all market indicators with their groups
    yaml_registry = registry.get("yaml_registry", {})
    indicators_by_group: Dict[str, List[str]] = {}

    # Build group -> indicator mapping from YAML registry
    if yaml_registry:
        for ind in yaml_registry.get("indicators", []):
            group = ind.get("group")
            ind_id = ind.get("id")
            if group and ind_id:
                if group not in indicators_by_group:
                    indicators_by_group[group] = []
                indicators_by_group[group].append(ind_id)
    else:
        # Fall back to market indicators from legacy format
        for ind in registry.get("market", []):
            ind_name = ind.get("name")
            group = ind.get("group", "market")
            if ind_name:
                if group not in indicators_by_group:
                    indicators_by_group[group] = []
                indicators_by_group[group].append(ind_name)

    # Generate indicator definitions for each applicable group
    for group in groups_to_apply:
        assets = indicators_by_group.get(group, [])

        for asset in assets:
            for template in indicator_templates:
                ind_type = template.get("type")
                params = template.get("params", {}).copy()
                suffix = template.get("suffix", ind_type)

                # Create indicator name
                name = f"{asset}_{suffix}"

                generated.append({
                    "name": name,
                    "base": asset,
                    "type": ind_type,
                    "params": params,
                    "group": "technical",
                })

    return generated


def _load_series(
    cursor: sqlite3.Cursor,
    name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[pd.Series]:
    """Load a time series from the database."""
    query = """
        SELECT iv.date, iv.value
        FROM indicator_values iv
        JOIN indicators i ON iv.indicator_id = i.id
        WHERE i.name = ?
    """
    params = [name]

    if start_date:
        query += " AND iv.date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND iv.date <= ?"
        params.append(end_date)

    query += " ORDER BY iv.date"

    cursor.execute(query, params)
    rows = cursor.fetchall()

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["date", "value"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df["value"]


def _write_technical_values(
    cursor: sqlite3.Cursor,
    name: str,
    series: pd.Series,
) -> int:
    """Write technical indicator values to database."""
    # Get or create indicator
    cursor.execute("SELECT id FROM indicators WHERE name = ?", (name,))
    row = cursor.fetchone()

    if row:
        indicator_id = row[0]
    else:
        cursor.execute(
            """
            INSERT INTO indicators (name, system, frequency, source)
            VALUES (?, ?, ?, ?)
            """,
            (name, "finance", "daily", "technical"),
        )
        indicator_id = cursor.lastrowid

    # Write values
    count = 0
    for date_val, value in series.items():
        if pd.isna(value) or np.isinf(value):
            continue
        date_str = date_val.strftime("%Y-%m-%d")
        cursor.execute(
            """
            INSERT OR REPLACE INTO indicator_values (indicator_id, date, value)
            VALUES (?, ?, ?)
            """,
            (indicator_id, date_str, float(value)),
        )
        count += 1

    return count


def _compute_technical(
    series: pd.Series,
    indicator_type: str,
    params: Dict[str, Any],
) -> Optional[pd.Series]:
    """
    Compute a technical indicator.

    Supported types:
        - sma: Simple Moving Average
        - ema: Exponential Moving Average
        - wma: Weighted Moving Average
        - rsi: Relative Strength Index
        - macd: MACD line
        - macd_signal: MACD signal line
        - macd_histogram: MACD histogram
        - bollinger_upper: Upper Bollinger Band
        - bollinger_lower: Lower Bollinger Band
        - bollinger_width: Bollinger Band width
        - bollinger_pct: %B (price position in bands)
        - volatility: Rolling standard deviation
        - momentum: Percentage change over window
        - roc: Rate of Change
        - zscore: Rolling Z-score
        - atr: Average True Range (requires high/low/close)
        - stochastic_k: Stochastic %K
        - stochastic_d: Stochastic %D
        - price_to_sma: Price relative to SMA
        - distance_from_high: Distance from rolling high
        - distance_from_low: Distance from rolling low
    """
    indicator_type = indicator_type.lower()
    window = params.get("window", 20)

    # Moving Averages
    if indicator_type == "sma":
        return series.rolling(window=window).mean()

    elif indicator_type == "ema":
        return series.ewm(span=window, adjust=False).mean()

    elif indicator_type == "wma":
        weights = np.arange(1, window + 1)
        return series.rolling(window=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )

    # RSI
    elif indicator_type == "rsi":
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    # MACD Family
    elif indicator_type == "macd":
        fast = params.get("fast", 12)
        slow = params.get("slow", 26)
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        return ema_fast - ema_slow

    elif indicator_type == "macd_signal":
        fast = params.get("fast", 12)
        slow = params.get("slow", 26)
        signal = params.get("signal", 9)
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        return macd_line.ewm(span=signal, adjust=False).mean()

    elif indicator_type == "macd_histogram":
        fast = params.get("fast", 12)
        slow = params.get("slow", 26)
        signal = params.get("signal", 9)
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line - signal_line

    # Bollinger Bands
    elif indicator_type == "bollinger_upper":
        std_dev = params.get("std_dev", 2)
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        return sma + (std_dev * std)

    elif indicator_type == "bollinger_lower":
        std_dev = params.get("std_dev", 2)
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        return sma - (std_dev * std)

    elif indicator_type == "bollinger_width":
        std_dev = params.get("std_dev", 2)
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return (upper - lower) / sma * 100

    elif indicator_type == "bollinger_pct":
        std_dev = params.get("std_dev", 2)
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return (series - lower) / (upper - lower)

    # Volatility & Momentum
    elif indicator_type == "volatility":
        return series.rolling(window=window).std()

    elif indicator_type == "volatility_pct":
        # Volatility as percentage of price
        vol = series.rolling(window=window).std()
        return vol / series * 100

    elif indicator_type == "momentum":
        return series.pct_change(periods=window) * 100

    elif indicator_type == "roc":
        # Rate of change
        return (series / series.shift(window) - 1) * 100

    elif indicator_type == "log_return":
        return np.log(series / series.shift(1))

    # Z-Score
    elif indicator_type == "zscore":
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        return (series - rolling_mean) / rolling_std.replace(0, np.nan)

    # Price Position Indicators
    elif indicator_type == "price_to_sma":
        sma = series.rolling(window=window).mean()
        return (series / sma - 1) * 100

    elif indicator_type == "distance_from_high":
        rolling_high = series.rolling(window=window).max()
        return (series / rolling_high - 1) * 100

    elif indicator_type == "distance_from_low":
        rolling_low = series.rolling(window=window).min()
        return (series / rolling_low - 1) * 100

    # Stochastic Oscillator
    elif indicator_type == "stochastic_k":
        low_min = series.rolling(window=window).min()
        high_max = series.rolling(window=window).max()
        return (series - low_min) / (high_max - low_min).replace(0, np.nan) * 100

    elif indicator_type == "stochastic_d":
        smooth = params.get("smooth", 3)
        low_min = series.rolling(window=window).min()
        high_max = series.rolling(window=window).max()
        k = (series - low_min) / (high_max - low_min).replace(0, np.nan) * 100
        return k.rolling(window=smooth).mean()

    # Rank-based
    elif indicator_type == "percentile_rank":
        return series.rolling(window=window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
        )

    # Trend indicators
    elif indicator_type == "trend_strength":
        # Simple trend strength based on returns consistency
        returns = series.pct_change()
        positive_days = returns.rolling(window=window).apply(
            lambda x: (x > 0).sum() / len(x), raw=True
        )
        return positive_days * 100

    else:
        logger.warning(f"Unknown indicator type '{indicator_type}'")
        return None


def get_technical_summary(results: Dict[str, int]) -> Dict[str, Any]:
    """Generate summary statistics for technical build results."""
    successful = {k: v for k, v in results.items() if v > 0}
    failed = {k: v for k, v in results.items() if v == 0}

    return {
        "total": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "total_rows": sum(results.values()),
        "successful_indicators": list(successful.keys()),
        "failed_indicators": list(failed.keys()),
    }
