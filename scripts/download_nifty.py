"""
Download NIFTY50 5-Minute Training Data — FREE, no API key needed.

Run: python scripts/download_nifty.py

Downloads 60 days of 5-minute OHLCV data for NIFTY50 + top stocks.
Saves to data/nifty50_5m.csv (used as training input by data_pipeline.py).
Takes ~1 minute.
"""
import numpy as np
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
    print(f"\nDownloading {name} ({ticker}) 5-minute candles...")
    try:
        df = yf.download(
            ticker,
            period="60d",
            interval="5m",
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

    # Add unix timestamp (seconds since epoch)
    df["timestamp"] = [int(t.timestamp()) for t in df.index]
    df = df.reset_index(drop=True)

    # If symbol has no volume (e.g. ^NSEI index where Yahoo Finance returns 0 volume),
    # generate proxy volume proportional to candle range/volatility for indicator compatibility
    if (df["volume"] == 0).mean() > 0.5:
        rng = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
        med_rng = rng.median() if rng.median() > 0 else 0.005
        df["volume"] = (100_000 * (1.0 + rng.fillna(0.005) / med_rng)).astype(int)

    df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna()

    out = DATA_DIR / f"{name}_5m.csv"
    df.to_csv(out, index=False)
    start = pd.to_datetime(df["timestamp"].iloc[0],  unit="s").strftime("%Y-%m-%d %H:%M")
    end   = pd.to_datetime(df["timestamp"].iloc[-1], unit="s").strftime("%Y-%m-%d %H:%M")
    print(f"  [OK] {len(df)} 5m bars  |  {start} -> {end}  |  saved: {out}")
    return True


def main():
    print("=" * 55)
    print("  NIFTY50 5-Minute Training Data Downloader")
    print("  Source : Yahoo Finance (free, no key)")
    print("  Output : data/nifty50_5m.csv  (primary training file)")
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
        print("    python -m forecast.train_lstm_v2")
        print()
        print("  Make sure config.yaml points to the right CSV:")
        print("    backtest.data_path: data/nifty50_5m.csv")
    else:
        print("  [FAIL] All downloads failed. Check your internet connection.")
    print("=" * 55)


if __name__ == "__main__":
    main()
