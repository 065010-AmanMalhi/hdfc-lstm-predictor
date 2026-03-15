"""
preprocess.py v3
================
Enhanced feature set:
  - Volume MA (5d, 10d) + volume ratio
  - Price momentum (5d, 10d, 20d)
  - Overnight gap (open vs prev close)
  - All previous features retained
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

PRICE_CSV = "hdfcbank_prices.csv"
NEWS_CSV  = "hdfcbank_news.csv"
OUTPUT_CSV   = "features.csv"
SEQUENCE_LEN = 30
TEST_SPLIT   = 0.2


def compute_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd     = ema_fast - ema_slow
    return macd, macd.ewm(span=signal, adjust=False).mean()

def compute_bollinger(series, period=20, std=2):
    ma    = series.rolling(period).mean()
    sd    = series.rolling(period).std()
    upper = ma + std * sd
    lower = ma - std * sd
    return upper, lower, (upper - lower) / (ma + 1e-10)


def main():
    print("=" * 60)
    print("  Preprocessing v3 — Enhanced Feature Set")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n  [1] Loading data...")
    price = pd.read_csv(PRICE_CSV, parse_dates=["Date"])
    price = price.sort_values("Date").reset_index(drop=True)

    news = pd.read_csv(NEWS_CSV, parse_dates=["date"])
    news["date"] = pd.to_datetime(news["date"]).dt.normalize()
    daily_sent = news.groupby("date")["compound"].mean().reset_index()
    daily_sent.columns = ["Date", "daily_compound"]

    print(f"      Price: {len(price)} trading days")
    print(f"      News : {len(daily_sent)} days with coverage")

    # ── Merge + sentiment fill ─────────────────────────────────────────────────
    df = price.merge(daily_sent, on="Date", how="left")
    df["daily_compound"] = df["daily_compound"].ffill(limit=5)
    df["daily_compound"] = df["daily_compound"].bfill(limit=3)
    df["daily_compound"] = df["daily_compound"].fillna(0.0)

    # ── Feature Engineering ───────────────────────────────────────────────────
    print("\n  [2] Engineering features...")

    # Basic price features
    df["returns"]        = df["Close"].pct_change()
    df["log_returns"]    = np.log(df["Close"] / df["Close"].shift(1))
    df["volatility_5d"]  = df["returns"].rolling(5).std()
    df["volatility_10d"] = df["returns"].rolling(10).std()
    df["hl_ratio"]       = (df["High"] - df["Low"]) / df["Close"]
    df["co_ratio"]       = (df["Close"] - df["Open"]) / df["Open"]

    # Lag features
    df["lag1_return"]    = df["returns"].shift(1)
    df["lag2_return"]    = df["returns"].shift(2)
    df["lag3_return"]    = df["returns"].shift(3)
    df["lag1_direction"] = (df["returns"].shift(1) > 0).astype(float)

    # Technical indicators
    df["rsi_14"]               = compute_rsi(df["Close"])
    df["rsi_7"]                = compute_rsi(df["Close"], period=7)
    df["macd"], df["macd_sig"] = compute_macd(df["Close"])
    df["macd_hist"]            = df["macd"] - df["macd_sig"]
    df["bb_upper"], df["bb_lower"], df["bb_width"] = compute_bollinger(df["Close"])

    # ── NEW: Volume features ──────────────────────────────────────────────────
    df["volume_ma5"]    = df["Volume"].rolling(5).mean()
    df["volume_ma10"]   = df["Volume"].rolling(10).mean()
    df["volume_ratio"]  = df["Volume"] / (df["volume_ma5"] + 1e-10)  # today vs avg

    # ── NEW: Price momentum ───────────────────────────────────────────────────
    df["momentum_5d"]   = df["Close"] / df["Close"].shift(5) - 1
    df["momentum_10d"]  = df["Close"] / df["Close"].shift(10) - 1
    df["momentum_20d"]  = df["Close"] / df["Close"].shift(20) - 1

    # ── NEW: Overnight gap ────────────────────────────────────────────────────
    df["overnight_gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

    # ── NEW: Price vs moving averages ─────────────────────────────────────────
    df["close_ma10"]    = df["Close"].rolling(10).mean()
    df["close_ma20"]    = df["Close"].rolling(20).mean()
    df["price_vs_ma10"] = df["Close"] / (df["close_ma10"] + 1e-10) - 1
    df["price_vs_ma20"] = df["Close"] / (df["close_ma20"] + 1e-10) - 1

    # Sentiment features
    df["sentiment_ma3"]  = df["daily_compound"].rolling(3, min_periods=1).mean()
    df["sentiment_ma7"]  = df["daily_compound"].rolling(7, min_periods=1).mean()
    df["sentiment_ma14"] = df["daily_compound"].rolling(14, min_periods=1).mean()

    # ── Targets ───────────────────────────────────────────────────────────────
    df["next_close"]     = df["Close"].shift(-1)       # next day for regression
    df["next_direction"] = (df["next_close"] > df["Close"]).astype(int) # next day up/down for classification
    df = df.dropna().reset_index(drop=True)

    up   = df["next_direction"].sum()
    down = len(df) - up
    print(f"      {len(df)} clean rows")
    print(f"      Up: {up} ({up/len(df)*100:.1f}%)  Down: {down} ({down/len(df)*100:.1f}%)")

    df.to_csv(OUTPUT_CSV, index=False)

    # ── Feature columns ───────────────────────────────────────────────────────
    FEATURE_COLS = [
    # Price & volume
    "Open", "High", "Low", "Close", "Volume",
    # Returns & volatility
    "returns", "log_returns", "volatility_5d", "volatility_10d",
    "hl_ratio", "co_ratio", "overnight_gap",
    # Lag features
    "lag1_return", "lag2_return", "lag1_direction",
    # Technical indicators
    "rsi_14", "macd", "macd_sig", "macd_hist",
    "bb_upper", "bb_lower", "bb_width",
    # Momentum
    "momentum_5d", "momentum_10d",
    # Sentiment
    "daily_compound", "sentiment_ma7",
]

    print(f"\n  [3] {len(FEATURE_COLS)} features → scaling...")
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(df[FEATURE_COLS].values)

    scaler_close = MinMaxScaler()
    scaler_close.fit(df[["Close"]].values)

    print(f"\n  [4] Building sequences (window={SEQUENCE_LEN})...")
    y_reg = df["next_close"].values
    y_cls = df["next_direction"].values
    X_seq, y_reg_seq, y_cls_seq = [], [], []

    for i in range(SEQUENCE_LEN, len(X_scaled)):
        X_seq.append(X_scaled[i - SEQUENCE_LEN:i])
        y_reg_seq.append(y_reg[i])
        y_cls_seq.append(y_cls[i])

    X_seq     = np.array(X_seq)
    y_reg_seq = np.array(y_reg_seq)
    y_cls_seq = np.array(y_cls_seq)
    print(f"      Shape: {X_seq.shape}")

    split = int(len(X_seq) * (1 - TEST_SPLIT))
    X_train, X_test         = X_seq[:split],     X_seq[split:]
    y_reg_train, y_reg_test = y_reg_seq[:split],  y_reg_seq[split:]
    y_cls_train, y_cls_test = y_cls_seq[:split],  y_cls_seq[split:]

    print(f"\n  [5] Split — Train: {len(X_train)}  Test: {len(X_test)}")
    print(f"      Test Up  : {y_cls_test.sum()} ({y_cls_test.mean()*100:.1f}%)")
    print(f"      Test Down: {(y_cls_test==0).sum()} ({(1-y_cls_test.mean())*100:.1f}%)")

    np.save("X_train.npy", X_train);   np.save("X_test.npy", X_test)
    np.save("y_reg_train.npy", y_reg_train); np.save("y_reg_test.npy", y_reg_test)
    np.save("y_cls_train.npy", y_cls_train); np.save("y_cls_test.npy", y_cls_test)
    with open("scaler_features.pkl","wb") as f: pickle.dump(scaler_X, f)
    with open("scaler_close.pkl","wb") as f:    pickle.dump(scaler_close, f)
    with open("feature_cols.pkl","wb") as f:    pickle.dump(FEATURE_COLS, f)

    print(f"\n{'='*60}")
    print(f"  Done! {len(FEATURE_COLS)} features, {len(X_train)} train sequences")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()