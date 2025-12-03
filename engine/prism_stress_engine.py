"""
PRISM Stress Engine - Financial Stress Analysis

Analyzes financial stress indicators (VIX, credit spreads, yield curve, etc.)
using registry-driven panel loading and the PRISM lens framework.

This engine focuses on identifying financial stress conditions and systemic risk.

Usage:
    from engine.prism_stress_engine import PrismStressEngine

    engine = PrismStressEngine()
    results = engine.analyze()
    stress_level = engine.get_stress_score()
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
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


class PrismStressEngine:
    """
    PRISM Engine for Financial Stress Analysis.

    Analyzes stress indicators to identify:
    - Overall financial stress levels
    - Yield curve inversions
    - Credit spread widening
    - Volatility regimes
    - Systemic risk signals
    """

    # Engine metadata
    name = "stress"
    description = "Financial stress analysis engine"
    version = "1.0.0"

    # Stress indicator categories
    STRESS_CATEGORIES = {
        "volatility": ["vix"],
        "yield_curve": ["t10y2y", "t10y3m"],
        "financial_conditions": ["nfci", "anfci"],
        "credit": ["hyg", "lqd"]
    }

    # Thresholds for stress levels
    THRESHOLDS = {
        "vix": {"low": 15, "medium": 20, "high": 30, "extreme": 40},
        "t10y2y": {"inverted": 0, "flat": 0.25},
        "nfci": {"tight": 0, "very_tight": 0.5},
        "anfci": {"tight": 0, "very_tight": 0.5}
    }

    def __init__(
        self,
        panel_name: str = "default",
        custom_indicators: Optional[List[str]] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize the Stress Engine.

        Args:
            panel_name: Panel to load from registry ("default", "climate", etc.)
            custom_indicators: Override default stress indicators
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
            self.indicators = get_engine_indicators("stress")

        logger.info(f"PrismStressEngine initialized with {len(self.indicators)} indicators")

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
            DataFrame with stress indicators
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
            logger.info(f"Loaded stress panel: {self._panel.shape}")
            return self._panel

        except (PanelLoadError, RegistryError) as e:
            logger.error(f"Failed to load stress panel: {e}")
            raise

    def analyze(
        self,
        df: Optional[pd.DataFrame] = None,
        lenses: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run stress analysis using PRISM lenses.

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

        # Default lenses for stress analysis
        if lenses is None:
            lenses = ["magnitude", "anomaly", "regime"]

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

        # Add stress-specific metrics
        results["stress_metrics"] = self._compute_stress_metrics(df)
        results["stress_score"] = self._compute_composite_stress_score(df)
        results["alerts"] = self._generate_stress_alerts(df)

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
            "current": panel.iloc[-1].to_dict() if len(panel) > 0 else {},
            "percentile": {}
        }

        # Compute percentiles
        for col in panel.columns:
            current = panel[col].iloc[-1]
            if not pd.isna(current):
                stats["percentile"][col] = float((panel[col] < current).mean() * 100)

        return {"basic_stats": stats}

    def _compute_stress_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute stress-specific metrics for each category."""
        value_cols = [c for c in df.columns if c != "date"]
        panel = df[value_cols]

        metrics = {}

        # Volatility stress
        if "vix" in panel.columns:
            vix = panel["vix"]
            current_vix = vix.iloc[-1]
            thresholds = self.THRESHOLDS["vix"]

            if current_vix >= thresholds["extreme"]:
                vix_level = "extreme"
            elif current_vix >= thresholds["high"]:
                vix_level = "high"
            elif current_vix >= thresholds["medium"]:
                vix_level = "medium"
            else:
                vix_level = "low"

            metrics["volatility"] = {
                "current_vix": float(current_vix) if not pd.isna(current_vix) else None,
                "stress_level": vix_level,
                "percentile": float((vix < current_vix).mean() * 100) if not pd.isna(current_vix) else None,
                "20d_avg": float(vix.tail(20).mean()),
                "change_1w": float(current_vix - vix.iloc[-5]) if len(vix) > 5 else None
            }

        # Yield curve stress
        yield_curve_metrics = {}
        for spread_name in ["t10y2y", "t10y3m"]:
            if spread_name in panel.columns:
                spread = panel[spread_name]
                current = spread.iloc[-1]
                is_inverted = current < 0 if not pd.isna(current) else None

                yield_curve_metrics[spread_name] = {
                    "current": float(current) if not pd.isna(current) else None,
                    "inverted": is_inverted,
                    "percentile": float((spread < current).mean() * 100) if not pd.isna(current) else None
                }

        if yield_curve_metrics:
            # Overall yield curve assessment
            inversions = [v["inverted"] for v in yield_curve_metrics.values() if v["inverted"] is not None]
            yield_curve_metrics["overall"] = {
                "any_inverted": any(inversions) if inversions else None,
                "all_inverted": all(inversions) if inversions else None
            }
            metrics["yield_curve"] = yield_curve_metrics

        # Financial conditions stress
        fin_cond_metrics = {}
        for indicator in ["nfci", "anfci"]:
            if indicator in panel.columns:
                series = panel[indicator]
                current = series.iloc[-1]

                if not pd.isna(current):
                    if current > self.THRESHOLDS.get(indicator, {}).get("very_tight", 0.5):
                        level = "very_tight"
                    elif current > self.THRESHOLDS.get(indicator, {}).get("tight", 0):
                        level = "tight"
                    else:
                        level = "loose"

                    fin_cond_metrics[indicator] = {
                        "current": float(current),
                        "level": level,
                        "percentile": float((series < current).mean() * 100),
                        "trend_4w": float(current - series.iloc[-20]) if len(series) > 20 else None
                    }

        if fin_cond_metrics:
            metrics["financial_conditions"] = fin_cond_metrics

        # Credit stress (HYG/LQD spread as proxy)
        if "hyg" in panel.columns and "lqd" in panel.columns:
            hyg_ret = panel["hyg"].pct_change()
            lqd_ret = panel["lqd"].pct_change()
            credit_spread_proxy = lqd_ret - hyg_ret  # HY underperformance = stress

            metrics["credit"] = {
                "hyg_current": float(panel["hyg"].iloc[-1]) if not pd.isna(panel["hyg"].iloc[-1]) else None,
                "lqd_current": float(panel["lqd"].iloc[-1]) if not pd.isna(panel["lqd"].iloc[-1]) else None,
                "spread_proxy_20d": float(credit_spread_proxy.tail(20).sum()),
                "hy_underperforming": credit_spread_proxy.tail(20).sum() > 0
            }

        return metrics

    def _compute_composite_stress_score(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute a composite stress score (0-100).

        Higher scores indicate higher stress levels.
        """
        value_cols = [c for c in df.columns if c != "date"]
        panel = df[value_cols]

        scores = []
        weights = []

        # VIX component (weight: 30%)
        if "vix" in panel.columns:
            vix = panel["vix"]
            current_vix = vix.iloc[-1]
            if not pd.isna(current_vix):
                vix_percentile = (vix < current_vix).mean() * 100
                scores.append(vix_percentile)
                weights.append(0.30)

        # Yield curve component (weight: 25%)
        yield_scores = []
        for spread_name in ["t10y2y", "t10y3m"]:
            if spread_name in panel.columns:
                spread = panel[spread_name]
                current = spread.iloc[-1]
                if not pd.isna(current):
                    # Inversion = high stress
                    percentile = (spread > current).mean() * 100  # Lower spread = higher stress
                    yield_scores.append(percentile)

        if yield_scores:
            scores.append(np.mean(yield_scores))
            weights.append(0.25)

        # Financial conditions component (weight: 25%)
        fc_scores = []
        for indicator in ["nfci", "anfci"]:
            if indicator in panel.columns:
                series = panel[indicator]
                current = series.iloc[-1]
                if not pd.isna(current):
                    percentile = (series < current).mean() * 100  # Higher = tighter = more stress
                    fc_scores.append(percentile)

        if fc_scores:
            scores.append(np.mean(fc_scores))
            weights.append(0.25)

        # Credit component (weight: 20%)
        if "hyg" in panel.columns:
            hyg = panel["hyg"]
            hyg_ret_20d = hyg.iloc[-1] / hyg.iloc[-20] - 1 if len(hyg) > 20 else 0
            # Negative HYG returns = stress
            hyg_stress = max(0, -hyg_ret_20d * 1000)  # Scale and cap
            scores.append(min(100, hyg_stress))
            weights.append(0.20)

        if not scores:
            return {"score": None, "level": "unknown", "components": {}}

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Compute weighted score
        composite_score = sum(s * w for s, w in zip(scores, weights))

        # Determine level
        if composite_score >= 75:
            level = "extreme"
        elif composite_score >= 60:
            level = "high"
        elif composite_score >= 40:
            level = "elevated"
        elif composite_score >= 25:
            level = "moderate"
        else:
            level = "low"

        return {
            "score": float(composite_score),
            "level": level,
            "components": dict(zip(
                ["volatility", "yield_curve", "financial_conditions", "credit"][:len(scores)],
                scores
            ))
        }

    def _generate_stress_alerts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate alerts based on stress conditions."""
        value_cols = [c for c in df.columns if c != "date"]
        panel = df[value_cols]

        alerts = []

        # VIX alerts
        if "vix" in panel.columns:
            vix = panel["vix"]
            current_vix = vix.iloc[-1]
            if not pd.isna(current_vix):
                if current_vix >= 40:
                    alerts.append({
                        "type": "extreme_volatility",
                        "severity": "critical",
                        "message": f"VIX at extreme level: {current_vix:.1f}",
                        "indicator": "vix"
                    })
                elif current_vix >= 30:
                    alerts.append({
                        "type": "high_volatility",
                        "severity": "warning",
                        "message": f"VIX elevated: {current_vix:.1f}",
                        "indicator": "vix"
                    })

                # VIX spike detection
                if len(vix) > 5:
                    vix_change = current_vix - vix.iloc[-5]
                    if vix_change > 10:
                        alerts.append({
                            "type": "vix_spike",
                            "severity": "warning",
                            "message": f"VIX spiked {vix_change:.1f} points in 5 days",
                            "indicator": "vix"
                        })

        # Yield curve alerts
        for spread_name in ["t10y2y", "t10y3m"]:
            if spread_name in panel.columns:
                spread = panel[spread_name]
                current = spread.iloc[-1]
                if not pd.isna(current) and current < 0:
                    alerts.append({
                        "type": "yield_curve_inversion",
                        "severity": "warning",
                        "message": f"{spread_name.upper()} inverted: {current:.2f}%",
                        "indicator": spread_name
                    })

        # Financial conditions alerts
        for indicator in ["nfci", "anfci"]:
            if indicator in panel.columns:
                series = panel[indicator]
                current = series.iloc[-1]
                if not pd.isna(current) and current > 0.5:
                    alerts.append({
                        "type": "tight_financial_conditions",
                        "severity": "warning",
                        "message": f"{indicator.upper()} indicates tight conditions: {current:.2f}",
                        "indicator": indicator
                    })

        return alerts

    def get_stress_score(self) -> Tuple[float, str]:
        """
        Get the current composite stress score.

        Returns:
            Tuple of (score, level)
        """
        if self._results is None:
            self.analyze()

        if self._results and "stress_score" in self._results:
            score_data = self._results["stress_score"]
            return (score_data.get("score"), score_data.get("level"))

        return (None, "unknown")

    def get_panel_info(self) -> Dict[str, Any]:
        """Get information about the loaded panel."""
        registry = get_registry("system")

        info = {
            "panel_name": self.panel_name,
            "panel_path": str(get_panel_path(self.panel_name)),
            "indicators": self.indicators,
            "engine_type": self.name,
            "stress_categories": self.STRESS_CATEGORIES
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
        return f"PrismStressEngine(panel='{self.panel_name}', indicators={len(self.indicators)})"
