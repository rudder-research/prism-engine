import pandas as pd
import requests
from io import StringIO

from data.sql.db import add_indicator, write_dataframe, load_indicator

print("\n=== PRISM STOOQ FETCH TEST (SPY.US) ===\n")

indicator = "spy"
system = "market"
ticker = "SPY.US"

# Register indicator
indicator_id = add_indicator(
    name=indicator,
    system=system,
    frequency="daily",
    source="stooq",
    units="price",
    description="SPY daily close via Stooq"
)
print(f"Registered indicator '{indicator}' with ID {indicator_id}\n")

# Fetch CSV from Stooq
url = f"https://stooq.com/q/d/l/?s={ticker}&i=d"
print(f"Fetching: {url}")

response = requests.get(url)
if response.status_code != 200:
    print("ERROR fetching from Stooq:", response.status_code)
    exit()

print("Download OK, parsing CSV...")

df = pd.read_csv(StringIO(response.text))

print("\nRaw head:")
print(df.head())

# Must contain at least: Date, Close
if "Date" not in df.columns or "Close" not in df.columns:
    print("\nERROR: Missing expected columns")
    print("Columns found:", df.columns)
    exit()

# Convert to prism dataframe
df["date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
df = df.rename(columns={"Close": "value"})
mini = df[["date", "value"]].dropna()

print("\nPrepared for DB:")
print(mini.head())

# Write to DB
rows = write_dataframe(mini, indicator, system)
print(f"\nInserted {rows} rows into indicator_values")

# Read back to confirm
loaded = load_indicator(indicator, system)
print("\nLoaded back from DB:")
print(loaded.head())

print("\n=== STOOQ TEST COMPLETE ===\n")
