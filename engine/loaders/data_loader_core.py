"""
Prism Engine — Core Data Loader
Fully portable loader for FRED + Yahoo Finance + registry-driven series.

Features:
- No absolute paths
- Uses PATHS resolver for all directories
- Works on Chromebook Linux, Mac, Windows, Colab
- Registry-based series fetching (FRED + Yahoo)
- Builds raw CSVs + master panel
"""

import os
from pathlib import Path
import pandas as pd
import yfinance as yf
from fredapi import Fred

# ------------------------------------------------------------
# PATH RESOLUTION (portable across every device)
# ------------------------------------------------------------
from engine.config.path_resolver import PATHS

DATA_DIR = PATHS.data_raw            # e.g. prism_engine/working_copy/data/raw
REGISTRY_PATH = PATHS.data_registry  # e.g. prism_engine/working_copy/data/registry/prism_metric_registry.json

# Ensure raw data folder exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# OPTIONAL: Colab secrets support
# ------------------------------------------------------------
try:
    from google.colab import userdata
except ImportError:
    userdata = None


# ============================================================
#                    CORE DATA LOADER
# ============================================================
class CoreDataLoader:

    def __init__(self):
        self.fred = None

        # Load registry JSON (array of dicts)
        if REGISTRY_PATH.exists():
            self.registry = pd.read_json(REGISTRY_PATH)
        else:
            print(f"⚠️ Registry not found at {REGISTRY_PATH}")
            self.registry = None

    # --------------------------------------------------------
    # Initialize FRED API
    # --------------------------------------------------------
    def init_fred(self):
        api_key = None

        # (optional) Colab secrets
        if userdata:
            api_key = userdata.get("FRED_API")

        # fallback: environment variable
        if not api_key:
            api_key = os.environ.get("FRED_API")

        if not api_key:
            raise ValueError(
                "❌ FRED_API not set. Store it in Colab (userdata) or as an environment variable."
            )

        self.fred = Fred(api_key)

    # --------------------------------------------------------
    # Internal sanitization: flatten columns, enforce date
    # --------------------------------------------------------
    def _sanitize(self, df, ticker):
        if df is None or df.empty:
            print(f"⚠️ {ticker} returned EMPTY dataset — placeholder created.")
            return pd.DataFrame({"date": [], ticker.lower(): []})

        # flatten MultiIndex columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join([str(part) for part in col if part]) for col in df.columns
            ]

        # normalize column names
        df.columns = [c.lower() for c in df.columns]

        if "date" not in df.columns:
            raise ValueError(f"❌ '{ticker}' missing date column")

        return df

    # --------------------------------------------------------
    # Fetch FRED series
    # --------------------------------------------------------
    def fetch_fred(self, ticker):
        try:
            series = self.fred.get_series(ticker)
            df = pd.DataFrame({"date": series.index, ticker.lower(): series.values})
            df["date"] = pd.to_datetime(df["date"])
            return self._sanitize(df, ticker)
        except Exception as e:
            print(f"❌ FRED error for {ticker}: {e}")
            return None

    # --------------------------------------------------------
    # Fetch Yahoo Finance ticker
    # --------------------------------------------------------
    def fetch_yahoo(self, ticker):
        try:
            df = yf.download(ticker, auto_adjust=True, progress=False)

            if df is None or df.empty:
                return self._sanitize(None, ticker)

            df = df.reset_index()
            df = df[["Date", "Close"]].rename(
                columns={"Date": "date", "Close": ticker.lower()}
            )
            df["date"] = pd.to_datetime(df["date"])

            return self._sanitize(df, ticker)

        except Exception as e:
            print(f"❌ Yahoo error for {ticker}: {e}")
            return self._sanitize(None, ticker)

    # --------------------------------------------------------
    # Save single dataframe to CSV
    # --------------------------------------------------------
    def save(self, df, name):
        out = DATA_DIR / f"{name.lower()}.csv"
        df.to_csv(out, index=False)
        print(f"✔ Saved {name} → {out}")

    # --------------------------------------------------------
    # Merge all dataframes into master panel
    # --------------------------------------------------------
    def build_master_panel(self, dfs):
        dfs = [df for df in dfs if not df.empty]

        if not dfs:
            print("⚠️ No dataframes to merge → master panel skipped.")
            return pd.DataFrame()

        panel = dfs[0]

        for df in dfs[1:]:
            df.columns = df.columns.astype(str)
            panel.columns = panel.columns.astype(str)
            panel = pd.merge(panel, df, on="date", how="outer")

        panel = panel.sort_values("date")
        panel.columns = [c.lower() for c in panel.columns]
        return panel

    # --------------------------------------------------------
    # MAIN — Fetch all series from registry
    # --------------------------------------------------------
    def fetch_all(self):
        if self.registry is None:
            raise ValueError("❌ Registry missing! Cannot fetch data.")

        self.init_fred()

        dfs = []

        for _, row in self.registry.iterrows():
            ticker = row["ticker"]
            source = row["source"].lower()

            if source == "fred":
                df = self.fetch_fred(ticker)
            elif source == "yahoo":
                df = self.fetch_yahoo(ticker)
            else:
                print(f"⚠️ Unknown source '{source}' for {ticker}")
                continue

            if df is not None and not df.empty:
                self.save(df, ticker)
                dfs.append(df)

        if not dfs:
            print("⚠️ No data fetched → master panel NOT created.")
            return pd.DataFrame()

        master = self.build_master_panel(dfs)
        out = DATA_DIR / "master_panel.csv"
        master.to_csv(out, index=False)

        print(f"\n📁 MASTER PANEL CREATED → {out}\n")
        return master
