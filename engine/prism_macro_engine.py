"""
PRISM Macro Engine - Macroeconomic Indicator Analysis

Analyzes macroeconomic indicators (GDP, employment, inflation, etc.) using
registry-driven panel loading and the PRISM lens framework.

This engine focuses on business cycle analysis and economic regime detection.

Usage:
    from engine.prism_macro_engine import PrismMacroEngine

    engine = PrismMacroEngine()
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


class PrismMacroEngine:
    """
    PRISM Engine for Macroeconomic Analysis.

    Analyzes economic indicators to identify:
    - Business cycle phases
    - Leading indicators
    - Economic regime changes
    - Cross-indicator relationships
    """

    # Engine metadata
    name = "macro"
    description = "Macroeconomic indicator analysis engine"
    version = "1.0.0"

    def __init__(
        self,
        panel_name: str = "default",
        custom_indicators: Optional[List[str]] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize the Macro Engine.

        Args:
            panel_name: Panel to load from registry ("default", "climate", etc.)
            custom_indicators: Override default macro indicators
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
            self.indicators = get_engine_indicators("macro")

        logger.info(f"PrismMacroEngine initialized with {len(self.indicators)} indicators")

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
            DataFrame with macro indicators
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
            logger.info(f"Loaded macro panel: {self._panel.shape}")
            return self._panel

        except (PanelLoadError, RegistryError) as e:
            logger.error(f"Failed to load macro panel: {e}")
            raise

    def analyze(
        self,
        df: Optional[pd.DataFrame] = None,
        lenses: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run macro analysis using PRISM lenses.

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

        # Default lenses for macro analysis
        if lenses is None:
            lenses = ["magnitude", "pca", "decomposition", "regime"]

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

        # Add macro-specific metrics
        results["macro_metrics"] = self._compute_macro_metrics(df)

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

    def _compute_macro_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute macroeconomic-specific metrics."""
        value_cols = [c for c in df.columns if c != "date"]
        panel = df[value_cols]

        metrics = {}

        # YoY changes for inflation indicators
        inflation_indicators = [c for c in ["cpi", "cpi_core", "ppi"] if c in panel.columns]
        if inflation_indicators:
            for ind in inflation_indicators:
                if len(panel[ind].dropna()) > 12:
                    yoy_change = panel[ind].pct_change(periods=12) * 100
                    metrics[f"{ind}_yoy"] = {
                        "latest": float(yoy_change.iloc[-1]) if not pd.isna(yoy_change.iloc[-1]) else None,
                        "mean": float(yoy_change.mean()),
                        "std": float(yoy_change.std())
                    }

        # Employment metrics
        if "unrate" in panel.columns:
            unrate = panel["unrate"]
            metrics["unemployment"] = {
                "latest": float(unrate.iloc[-1]) if not pd.isna(unrate.iloc[-1]) else None,
                "trend_12m": float(unrate.iloc[-1] - unrate.iloc[-12]) if len(unrate) > 12 else None
            }

        # Liquidity metrics
        if "m2" in panel.columns:
            m2 = panel["m2"]
            metrics["m2_growth"] = {
                "yoy": float(m2.pct_change(periods=12).iloc[-1] * 100) if len(m2) > 12 else None
            }

        return metrics

    def get_leading_indicators(self, top_n: int = 5) -> List[str]:
        """
        Get the top leading indicators from the last analysis.

        Args:
            top_n: Number of indicators to return

        Returns:
            List of indicator names
        """
        if self._results is None:
            logger.warning("No analysis results available. Run analyze() first.")
            return []

        # Try to extract from consensus
        if "consensus" in self._results:
            consensus = self._results["consensus"]
            if isinstance(consensus, dict) and "avg_rank" in str(consensus):
                # Extract top indicators
                pass

        # Fall back to magnitude importance
        if "magnitude" in self._results and "importance" in self._results["magnitude"]:
            importance = self._results["magnitude"]["importance"]
            if isinstance(importance, pd.Series):
                return list(importance.sort_values(ascending=False).head(top_n).index)
            elif isinstance(importance, dict):
                sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                return [k for k, v in sorted_imp[:top_n]]

        return []

    def get_panel_info(self) -> Dict[str, Any]:
        """Get information about the loaded panel."""
        registry = get_registry("system")

        info = {
            "panel_name": self.panel_name,
            "panel_path": str(get_panel_path(self.panel_name)),
            "indicators": self.indicators,
            "engine_type": self.name,
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
        return f"PrismMacroEngine(panel='{self.panel_name}', indicators={len(self.indicators)})"
