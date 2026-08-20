"""
Download NIFTY50 training data — FREE, no API key needed.

Run: python scripts/download_nifty.py

Downloads 2 years of 1h OHLCV data for NIFTY50 + top stocks.
Saves to data/nifty50_1h.csv (used as training input by data_pipeline.py).
Takes ~2 minutes.
"""
import pandas as pd
import yfinance as yf
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Primary training stock → NIFTY50 index (most data, cleanest patterns)
# Fallback stocks if NIFTY is unavailable
SYMBOLS = {
    "nifty50":  "^NSEI",        # NIFTY 50 index  ← BEST for training
    "reliance": "RELIANCE.NS",  # Reliance Industries
    "tcs":      "TCS.NS",       # TCS (IT sector, clean trends)
}


def download(name: str, ticker: str) -> bool:
    print(f"\nDownloading {name} ({ticker})...")
    try:
        df = yf.download(
            ticker,
            period="2y",
            interval="1h",
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        print(f"  [FAIL] Download failed: {e}")
        return False

    if df.empty:
        print(f"  [FAIL] No data returned for {ticker}")
        return False

    # Flatten MultiIndex columns (yfinance sometimes returns these)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Rename to pipeline format
    df = df.rename(columns={
        "Open": "open", "High": "high",
        "Low": "low",  "Close": "close", "Volume": "volume",
    })

    # Add unix timestamp
    df["timestamp"] = df.index.astype("int64") // 1_000_000_000
    df = df.reset_index(drop=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna()

    out = DATA_DIR / f"{name}_1h.csv"
    df.to_csv(out, index=False)
    start = pd.to_datetime(df["timestamp"].iloc[0],  unit="s").date()
    end   = pd.to_datetime(df["timestamp"].iloc[-1], unit="s").date()
    print(f"  [OK] {len(df)} bars  |  {start} -> {end}  |  saved: {out}")
    return True


def main():
    print("=" * 55)
    print("  NIFTY50 Training Data Downloader")
    print("  Source : Yahoo Finance (free, no key)")
    print("  Output : data/nifty50_1h.csv  (primary training file)")
    print("=" * 55)

    success = []
    for name, ticker in SYMBOLS.items():
        ok = download(name, ticker)
        if ok:
            success.append(name)

    print("\n" + "=" * 55)
    if success:
        print(f"  Downloaded: {', '.join(success)}")
        print()
        print("  Next step - train the model:")
        print("    python -m forecast.train_lstm")
        print()
        print("  Make sure config.yaml points to the right CSV:")
        print("    backtest.data_path: data/nifty50_1h.csv")
    else:
        print("  [FAIL] All downloads failed. Check your internet connection.")
    print("=" * 55)


if __name__ == "__main__":
    main()
