"""
scraper_price.py
================
Downloads HDFCBANK.NS historical OHLCV data from Yahoo Finance.
6 years: 2020-01-01 to 2026-03-14

Output: hdfcbank_prices.csv
Columns: Date, Open, High, Low, Close, Volume, Adj Close
"""

import yfinance as yf
import pandas as pd

TICKER = "HDFCBANK.NS"
START = "2020-01-01"
END   = "2026-03-14"
OUTPUT_CSV = "hdfcbank_prices.csv"

def main():
    print("=" * 60)
    print(f"  Price Scraper — {TICKER}")
    print(f"  Range: {START} → {END}")
    print("=" * 60)

    df = yf.download(TICKER, start=START, end=END, auto_adjust=True)

    if df.empty:
        print("  [!] No data returned. Check ticker or internet connection.")
        return

    df.reset_index(inplace=True)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df = df.dropna()
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n  Done! {len(df)} trading days saved to {OUTPUT_CSV}")
    print(f"  Date range: {df['Date'].min()} → {df['Date'].max()}")
    print(f"\n  Price summary:")
    print(f"    Open  : {df['Open'].min():.2f} → {df['Open'].max():.2f}")
    print(f"    Close : {df['Close'].min():.2f} → {df['Close'].max():.2f}")
    print(f"    Volume: avg {df['Volume'].mean():,.0f}")

if __name__ == "__main__":
    main()