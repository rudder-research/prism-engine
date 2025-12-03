"""
PRISM ML Engine - Machine Learning Analysis

Provides machine learning capabilities for indicator analysis using
registry-driven panel loading and the PRISM lens framework.

This engine focuses on pattern recognition, anomaly detection, and predictive analysis.

Usage:
    from engine.prism_ml_engine import PrismMLEngine

    engine = PrismMLEngine()
    results = engine.analyze()
    anomalies = engine.detect_anomalies()
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


class PrismMLEngine:
    """
    PRISM Engine for Machine Learning Analysis.

    Provides ML-based analysis including:
    - Anomaly detection
    - Pattern recognition
    - Regime classification
    - Feature importance ranking
    - Dimensionality reduction
    """

    # Engine metadata
    name = "ml"
    description = "Machine learning analysis engine"
    version = "1.0.0"

    def __init__(
        self,
        panel_name: str = "default",
        custom_indicators: Optional[List[str]] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize the ML Engine.

        Args:
            panel_name: Panel to load from registry ("default", "climate", etc.)
            custom_indicators: Override default indicators (uses all by default)
            config: Additional configuration options
        """
        self.panel_name = panel_name
        self.config = config or {}
        self._panel: Optional[pd.DataFrame] = None
        self._results: Optional[Dict] = None

        # ML engine uses all available indicators by default
        if custom_indicators is not None:
            self.indicators = custom_indicators
        else:
            # Combine all indicator types
            self.indicators = None  # Will load all columns

        logger.info("PrismMLEngine initialized")

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
            DataFrame with indicators
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
            logger.info(f"Loaded ML panel: {self._panel.shape}")
            return self._panel

        except (PanelLoadError, RegistryError) as e:
            logger.error(f"Failed to load ML panel: {e}")
            raise

    def analyze(
        self,
        df: Optional[pd.DataFrame] = None,
        lenses: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run ML analysis using PRISM lenses.

        Args:
            df: Optional DataFrame (loads from registry if None)
            lenses: List of lenses to run (uses ML-focused defaults if None)
            **kwargs: Additional parameters for lenses

        Returns:
            Dictionary with analysis results
        """
        # Load data if not provided
        if df is None:
            df = self.load_data()

        if df is None or df.empty:
            return {"error": "No data available for analysis"}

        # Default lenses for ML analysis
        if lenses is None:
            lenses = ["pca", "clustering", "anomaly", "mutual_info"]

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

        # Add ML-specific analysis
        results["feature_analysis"] = self._compute_feature_analysis(df)
        results["anomaly_detection"] = self._detect_anomalies_internal(df)
        results["regime_analysis"] = self._analyze_regimes(df)

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

        # PCA-based feature importance
        normalized = (panel - panel.mean()) / panel.std()
        normalized = normalized.fillna(0)

        try:
            U, S, Vt = np.linalg.svd(normalized.values, full_matrices=False)
            explained_var = (S ** 2) / (len(normalized) - 1)
            explained_var_ratio = explained_var / explained_var.sum()

            # Feature importance from top 3 components
            loadings = Vt[:3].T * S[:3]
            importance = np.abs(loadings).sum(axis=1)
            stats["pca_importance"] = dict(zip(panel.columns, importance))
            stats["explained_variance"] = explained_var_ratio[:5].tolist()
        except Exception as e:
            logger.warning(f"PCA failed: {e}")

        return {"basic_stats": stats}

    def _compute_feature_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute feature importance and relationships."""
        value_cols = [c for c in df.columns if c != "date"]
        panel = df[value_cols]

        analysis = {}

        # Normalize data
        normalized = (panel - panel.mean()) / panel.std()
        normalized = normalized.fillna(0)

        # PCA-based importance
        try:
            U, S, Vt = np.linalg.svd(normalized.values, full_matrices=False)

            # Explained variance
            explained_var = (S ** 2) / (len(normalized) - 1)
            total_var = explained_var.sum()
            explained_var_ratio = explained_var / total_var

            analysis["pca"] = {
                "n_components_90pct": int(np.searchsorted(np.cumsum(explained_var_ratio), 0.9) + 1),
                "top_5_variance": explained_var_ratio[:5].tolist(),
                "cumulative_variance": np.cumsum(explained_var_ratio)[:10].tolist()
            }

            # Feature importance from loadings
            n_components = min(5, len(S))
            loadings = Vt[:n_components].T * S[:n_components]
            importance = np.abs(loadings).sum(axis=1)
            importance_normalized = importance / importance.sum()

            analysis["feature_importance"] = {
                col: float(imp)
                for col, imp in sorted(
                    zip(panel.columns, importance_normalized),
                    key=lambda x: x[1],
                    reverse=True
                )
            }

        except Exception as e:
            logger.warning(f"Feature analysis failed: {e}")
            analysis["error"] = str(e)

        # Correlation clusters
        try:
            corr_matrix = panel.corr()
            high_corr_pairs = []

            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j:
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            high_corr_pairs.append({
                                "pair": [col1, col2],
                                "correlation": float(corr_val)
                            })

            analysis["high_correlation_pairs"] = sorted(
                high_corr_pairs,
                key=lambda x: abs(x["correlation"]),
                reverse=True
            )[:20]

        except Exception as e:
            logger.warning(f"Correlation analysis failed: {e}")

        return analysis

    def _detect_anomalies_internal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Internal anomaly detection using statistical methods."""
        value_cols = [c for c in df.columns if c != "date"]
        panel = df[value_cols]

        anomalies = {
            "zscore_anomalies": {},
            "iqr_anomalies": {},
            "summary": {}
        }

        total_anomalies = 0

        for col in panel.columns:
            series = panel[col].dropna()
            if len(series) < 20:
                continue

            # Z-score anomalies (|z| > 3)
            mean = series.mean()
            std = series.std()
            if std > 0:
                z_scores = (series - mean) / std
                zscore_anomaly_idx = z_scores[abs(z_scores) > 3].index.tolist()

                if zscore_anomaly_idx:
                    anomalies["zscore_anomalies"][col] = {
                        "count": len(zscore_anomaly_idx),
                        "latest_anomaly": str(zscore_anomaly_idx[-1]) if zscore_anomaly_idx else None,
                        "max_zscore": float(abs(z_scores).max())
                    }
                    total_anomalies += len(zscore_anomaly_idx)

            # IQR anomalies
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            if iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                iqr_anomaly_idx = series[(series < lower_bound) | (series > upper_bound)].index.tolist()

                if iqr_anomaly_idx:
                    anomalies["iqr_anomalies"][col] = {
                        "count": len(iqr_anomaly_idx),
                        "latest_anomaly": str(iqr_anomaly_idx[-1]) if iqr_anomaly_idx else None,
                        "bounds": [float(lower_bound), float(upper_bound)]
                    }

        anomalies["summary"] = {
            "total_zscore_anomalies": total_anomalies,
            "indicators_with_anomalies": len(anomalies["zscore_anomalies"]),
            "total_indicators": len(panel.columns)
        }

        return anomalies

    def _analyze_regimes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market regimes using simple statistical methods."""
        value_cols = [c for c in df.columns if c != "date"]
        panel = df[value_cols]

        regimes = {}

        # Rolling volatility regime
        window = min(60, len(panel) // 4)
        if window > 10:
            rolling_vol = panel.std(axis=1).rolling(window).mean()

            if len(rolling_vol.dropna()) > 0:
                current_vol = rolling_vol.iloc[-1]
                vol_percentile = (rolling_vol < current_vol).mean() * 100

                regimes["volatility_regime"] = {
                    "current_level": float(current_vol) if not pd.isna(current_vol) else None,
                    "percentile": float(vol_percentile) if not pd.isna(vol_percentile) else None,
                    "regime": "high" if vol_percentile > 70 else ("low" if vol_percentile < 30 else "normal")
                }

        # Correlation regime
        if len(panel) > window * 2:
            # Recent correlation vs historical
            recent_corr = panel.tail(window).corr().values
            historical_corr = panel.head(len(panel) - window).corr().values

            # Average correlation (excluding diagonal)
            mask = ~np.eye(recent_corr.shape[0], dtype=bool)
            recent_avg_corr = recent_corr[mask].mean()
            historical_avg_corr = historical_corr[mask].mean()

            regimes["correlation_regime"] = {
                "recent_avg_correlation": float(recent_avg_corr),
                "historical_avg_correlation": float(historical_avg_corr),
                "regime": "high_correlation" if recent_avg_corr > historical_avg_corr + 0.1 else (
                    "low_correlation" if recent_avg_corr < historical_avg_corr - 0.1 else "normal"
                )
            }

        # Trend regime (based on first principal component)
        normalized = (panel - panel.mean()) / panel.std()
        normalized = normalized.fillna(0)

        try:
            U, S, Vt = np.linalg.svd(normalized.values, full_matrices=False)
            pc1 = U[:, 0] * S[0]

            # Recent vs historical trend
            recent_pc1 = pc1[-window:].mean() if len(pc1) > window else pc1.mean()
            historical_pc1 = pc1[:-window].mean() if len(pc1) > window else 0

            regimes["trend_regime"] = {
                "recent_pc1": float(recent_pc1),
                "historical_pc1": float(historical_pc1),
                "regime": "uptrend" if recent_pc1 > historical_pc1 + 0.5 else (
                    "downtrend" if recent_pc1 < historical_pc1 - 0.5 else "sideways"
                )
            }
        except Exception as e:
            logger.warning(f"Trend regime analysis failed: {e}")

        return regimes

    def detect_anomalies(
        self,
        method: str = "zscore",
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Detect anomalies in the panel data.

        Args:
            method: "zscore" or "iqr"
            threshold: Threshold for anomaly detection (z-score or IQR multiplier)

        Returns:
            DataFrame with anomaly flags
        """
        if self._panel is None:
            self.load_data()

        if self._panel is None:
            return pd.DataFrame()

        value_cols = [c for c in self._panel.columns if c != "date"]
        panel = self._panel[value_cols].copy()

        if method == "zscore":
            normalized = (panel - panel.mean()) / panel.std()
            anomaly_mask = abs(normalized) > threshold
        elif method == "iqr":
            q1 = panel.quantile(0.25)
            q3 = panel.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            anomaly_mask = (panel < lower) | (panel > upper)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Add date back
        result = anomaly_mask.copy()
        if "date" in self._panel.columns:
            result.insert(0, "date", self._panel["date"])

        return result

    def get_feature_importance(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top features by importance.

        Args:
            top_n: Number of features to return

        Returns:
            List of (feature_name, importance_score) tuples
        """
        if self._results is None:
            self.analyze()

        if self._results and "feature_analysis" in self._results:
            fa = self._results["feature_analysis"]
            if "feature_importance" in fa:
                imp = fa["feature_importance"]
                sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
                return sorted_imp[:top_n]

        return []

    def get_panel_info(self) -> Dict[str, Any]:
        """Get information about the loaded panel."""
        registry = get_registry("system")

        info = {
            "panel_name": self.panel_name,
            "panel_path": str(get_panel_path(self.panel_name)),
            "indicators": self.indicators if self.indicators else "all",
            "engine_type": self.name,
        }

        if self._panel is not None:
            info["loaded"] = True
            info["shape"] = self._panel.shape
            info["columns"] = [c for c in self._panel.columns if c != "date"]
            info["date_range"] = [
                str(self._panel["date"].min()) if "date" in self._panel.columns else "N/A",
                str(self._panel["date"].max()) if "date" in self._panel.columns else "N/A"
            ]
        else:
            info["loaded"] = False

        return info

    def __repr__(self) -> str:
        n_indicators = len(self.indicators) if self.indicators else "all"
        return f"PrismMLEngine(panel='{self.panel_name}', indicators={n_indicators})"
