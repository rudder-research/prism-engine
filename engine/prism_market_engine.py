"""
PRISM Market Engine - Market Indicator Analysis

Analyzes market-based indicators (equities, commodities, fixed income, volatility)
using registry-driven panel loading and the PRISM lens framework.

This engine focuses on market dynamics, cross-asset relationships, and price-based signals.

Usage:
    from engine.prism_market_engine import PrismMarketEngine

    engine = PrismMarketEngine()
    results = engine.analyze()
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path for imports
import sys
_ENGINE_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _ENGINE_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.panel_loader import (
    load_panel,
    get_registry,
    get_engine_indicators,
    get_panel_path,
    PanelLoadError,
    RegistryError
)

logger = logging.getLogger(__name__)


class PrismMarketEngine:
    """
    PRISM Engine for Market Analysis.

    Analyzes market indicators to identify:
    - Cross-asset correlations
    - Risk-on/risk-off regimes
    - Market momentum and reversals
    - Sector leadership
    """

    # Engine metadata
    name = "market"
    description = "Market indicator analysis engine"
    version = "1.0.0"

    # Asset class groupings
    ASSET_GROUPS = {
        "equity": ["spy", "qqq", "iwm"],
        "volatility": ["vix"],
        "currency": ["dxy"],
        "commodities": ["gld", "slv", "uso", "bcom"],
        "fixed_income": ["bnd", "tlt", "shy", "ief", "tip"],
        "credit": ["lqd", "hyg"],
        "sectors": ["xlu"]
    }

    def __init__(
        self,
        panel_name: str = "default",
        custom_indicators: Optional[List[str]] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize the Market Engine.

        Args:
            panel_name: Panel to load from registry ("default", "climate", etc.)
            custom_indicators: Override default market indicators
            config: Additional configuration options
        """
        self.panel_name = panel_name
        self.config = config or {}
        self._panel: Optional[pd.DataFrame] = None
        self._results: Optional[Dict] = None

        # Get indicators from registry or use custom
        if custom_indicators is not None:
            self.indicators = custom_indicators
        else:
            self.indicators = get_engine_indicators("market")

        logger.info(f"PrismMarketEngine initialized with {len(self.indicators)} indicators")

    def load_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        reload: bool = False
    ) -> pd.DataFrame:
        """
        Load panel data using registry configuration.

        Args:
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            reload: Force reload even if already loaded

        Returns:
            DataFrame with market indicators
        """
        if self._panel is not None and not reload:
            return self._panel

        try:
            self._panel = load_panel(
                panel_name=self.panel_name,
                columns=self.indicators,
                start_date=start_date,
                end_date=end_date,
                fill_na=True
            )
            logger.info(f"Loaded market panel: {self._panel.shape}")
            return self._panel

        except (PanelLoadError, RegistryError) as e:
            logger.error(f"Failed to load market panel: {e}")
            raise

    def analyze(
        self,
        df: Optional[pd.DataFrame] = None,
        lenses: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run market analysis using PRISM lenses.

        Args:
            df: Optional DataFrame (loads from registry if None)
            lenses: List of lenses to run (uses defaults if None)
            **kwargs: Additional parameters for lenses

        Returns:
            Dictionary with analysis results
        """
        # Load data if not provided
        if df is None:
            df = self.load_data()

        if df is None or df.empty:
            return {"error": "No data available for analysis"}

        # Default lenses for market analysis
        if lenses is None:
            lenses = ["magnitude", "pca", "clustering", "influence", "network"]

        # Import analysis components
        try:
            from loader import run_lens, compute_consensus
        except ImportError:
            logger.warning("Loader not available, using basic analysis")
            return self._basic_analysis(df)

        # Run lenses
        results = {}
        for lens_name in lenses:
            try:
                value_cols = [c for c in df.columns if c != "date"]
                panel_data = df[value_cols]
                results[lens_name] = run_lens(lens_name, panel_data)
                logger.info(f"Completed {lens_name} lens")
            except Exception as e:
                logger.warning(f"Lens {lens_name} failed: {e}")
                results[lens_name] = {"error": str(e)}

        # Compute consensus if multiple lenses succeeded
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        if len(valid_results) > 1:
            try:
                consensus = compute_consensus(valid_results)
                results["consensus"] = consensus.to_dict() if hasattr(consensus, "to_dict") else consensus
            except Exception as e:
                logger.warning(f"Consensus computation failed: {e}")

        # Add market-specific metrics
        results["market_metrics"] = self._compute_market_metrics(df)
        results["cross_asset"] = self._compute_cross_asset_metrics(df)

        self._results = results
        return results

    def _basic_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Basic analysis when full lens framework is unavailable."""
        value_cols = [c for c in df.columns if c != "date"]
        panel = df[value_cols]

        # Compute basic statistics
        stats = {
            "mean": panel.mean().to_dict(),
            "std": panel.std().to_dict(),
            "correlation": panel.corr().to_dict(),
        }

        # L2 norm importance
        normalized = (panel - panel.mean()) / panel.std()
        magnitude = np.sqrt((normalized ** 2).sum())
        stats["magnitude_importance"] = magnitude.to_dict()

        return {"basic_stats": stats}

    def _compute_market_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute market-specific metrics."""
        value_cols = [c for c in df.columns if c != "date"]
        panel = df[value_cols]

        metrics = {}

        # Returns analysis for equity indicators
        equity_cols = [c for c in self.ASSET_GROUPS.get("equity", []) if c in panel.columns]
        for col in equity_cols:
            if len(panel[col].dropna()) > 20:
                returns = panel[col].pct_change()
                metrics[f"{col}_performance"] = {
                    "return_1m": float(panel[col].iloc[-1] / panel[col].iloc[-21] - 1) if len(panel) > 21 else None,
                    "return_3m": float(panel[col].iloc[-1] / panel[col].iloc[-63] - 1) if len(panel) > 63 else None,
                    "return_ytd": float(panel[col].iloc[-1] / panel[col].iloc[0] - 1) if len(panel) > 1 else None,
                    "volatility_20d": float(returns.tail(20).std() * np.sqrt(252)),
                    "sharpe_20d": float(returns.tail(20).mean() / returns.tail(20).std() * np.sqrt(252)) if returns.tail(20).std() > 0 else None
                }

        # VIX metrics
        if "vix" in panel.columns:
            vix = panel["vix"]
            metrics["volatility_regime"] = {
                "current_vix": float(vix.iloc[-1]) if not pd.isna(vix.iloc[-1]) else None,
                "vix_percentile": float((vix < vix.iloc[-1]).mean() * 100) if not pd.isna(vix.iloc[-1]) else None,
                "vix_20d_avg": float(vix.tail(20).mean()),
                "regime": "high" if vix.iloc[-1] > 25 else ("elevated" if vix.iloc[-1] > 18 else "low")
            }

        # Dollar strength
        if "dxy" in panel.columns:
            dxy = panel["dxy"]
            metrics["dollar"] = {
                "current": float(dxy.iloc[-1]) if not pd.isna(dxy.iloc[-1]) else None,
                "change_1m": float(dxy.iloc[-1] / dxy.iloc[-21] - 1) if len(dxy) > 21 else None,
                "percentile": float((dxy < dxy.iloc[-1]).mean() * 100) if not pd.isna(dxy.iloc[-1]) else None
            }

        return metrics

    def _compute_cross_asset_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute cross-asset correlation and relationship metrics."""
        value_cols = [c for c in df.columns if c != "date"]
        panel = df[value_cols]

        metrics = {}

        # Rolling correlation between SPY and key assets
        if "spy" in panel.columns:
            spy_returns = panel["spy"].pct_change()

            for col in panel.columns:
                if col != "spy" and len(panel[col].dropna()) > 60:
                    col_returns = panel[col].pct_change()
                    # 60-day rolling correlation
                    rolling_corr = spy_returns.rolling(60).corr(col_returns)
                    metrics[f"spy_{col}_corr"] = {
                        "current": float(rolling_corr.iloc[-1]) if not pd.isna(rolling_corr.iloc[-1]) else None,
                        "mean": float(rolling_corr.mean()),
                        "regime": "positive" if rolling_corr.iloc[-1] > 0.3 else ("negative" if rolling_corr.iloc[-1] < -0.3 else "neutral")
                    }

        # Risk-on/Risk-off indicator
        risk_on_assets = [c for c in ["spy", "qqq", "hyg"] if c in panel.columns]
        risk_off_assets = [c for c in ["tlt", "gld", "vix"] if c in panel.columns]

        if risk_on_assets and risk_off_assets:
            risk_on_returns = panel[risk_on_assets].pct_change().mean(axis=1)
            risk_off_returns = panel[risk_off_assets].pct_change().mean(axis=1)
            risk_sentiment = risk_on_returns.rolling(20).mean() - risk_off_returns.rolling(20).mean()

            metrics["risk_sentiment"] = {
                "current": float(risk_sentiment.iloc[-1]) if not pd.isna(risk_sentiment.iloc[-1]) else None,
                "regime": "risk_on" if risk_sentiment.iloc[-1] > 0.001 else ("risk_off" if risk_sentiment.iloc[-1] < -0.001 else "neutral")
            }

        return metrics

    def get_asset_group_performance(self, lookback_days: int = 20) -> Dict[str, float]:
        """
        Get performance by asset group.

        Args:
            lookback_days: Number of days for return calculation

        Returns:
            Dictionary mapping asset group to return
        """
        if self._panel is None:
            self.load_data()

        if self._panel is None:
            return {}

        performance = {}
        for group_name, tickers in self.ASSET_GROUPS.items():
            available = [t for t in tickers if t in self._panel.columns]
            if available:
                group_data = self._panel[available]
                if len(group_data) > lookback_days:
                    group_return = (group_data.iloc[-1] / group_data.iloc[-lookback_days] - 1).mean()
                    performance[group_name] = float(group_return)

        return performance

    def get_panel_info(self) -> Dict[str, Any]:
        """Get information about the loaded panel."""
        registry = get_registry("system")

        info = {
            "panel_name": self.panel_name,
            "panel_path": str(get_panel_path(self.panel_name)),
            "indicators": self.indicators,
            "engine_type": self.name,
            "asset_groups": self.ASSET_GROUPS
        }

        if self._panel is not None:
            info["loaded"] = True
            info["shape"] = self._panel.shape
            info["date_range"] = [
                str(self._panel["date"].min()) if "date" in self._panel.columns else "N/A",
                str(self._panel["date"].max()) if "date" in self._panel.columns else "N/A"
            ]
        else:
            info["loaded"] = False

        return info

    def __repr__(self) -> str:
        return f"PrismMarketEngine(panel='{self.panel_name}', indicators={len(self.indicators)})"
