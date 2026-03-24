# 🏦 HDFC Bank LSTM Stock Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![R²](https://img.shields.io/badge/R²-0.9768-00C851?style=for-the-badge)
![MAPE](https://img.shields.io/badge/MAPE-0.76%25-004C8F?style=for-the-badge)

<br/>

**A dual LSTM architecture trained on 6 years of NSE price data and 5,422 news headlines to predict HDFC Bank's next trading day closing price. Achieved R² = 0.9768 and MAPE = 0.76% on the held-out test set.**

<br/>

<a href="https://hdfc-lstm-predictor-6gsrdp3df4xxjm4ferufcj.streamlit.app">
  <img src="https://img.shields.io/badge/🚀%20Live%20Demo-Click%20Here-004C8F?style=for-the-badge&logoColor=white" alt="Live Demo"/>
</a>
&nbsp;
<a href="#-results">
  <img src="https://img.shields.io/badge/📊%20Results-View-00C851?style=for-the-badge&logoColor=white" alt="Results"/>
</a>
&nbsp;
<a href="#-run-locally">
  <img src="https://img.shields.io/badge/🛠%20Run%20Locally-Guide-FF6F00?style=for-the-badge&logoColor=white" alt="Run Locally"/>
</a>

</div>

---

## 🎯 Overview

| Step | What happens |
|------|-------------|
| **Scrape** | 1,537 trading days of HDFCBANK.NS OHLCV from Yahoo Finance |
| **News** | 5,422 headlines from Google News scored with VADER sentiment |
| **Features** | 26 engineered features — price, technical indicators, momentum, sentiment |
| **Train** | Dual LSTM — regression for price, classifier for direction |
| **Deploy** | Live Streamlit app with DCF/CAPM valuation + real-time yfinance data |

---

## 🏗 Architecture

```
HDFCBANK.NS (yfinance)          Google News RSS
       ↓                              ↓
scraper_price.py            scraper_news.py (VADER)
       ↓                              ↓
           preprocess.py (26 features, 30-day sequences)
                        ↓
         ┌──────────────┴──────────────┐
         ↓                             ↓
   LSTM Regression              LSTM Classifier
   (next-day price)            (Up/Down direction)
         ↓                             ↓
              Streamlit Dashboard
    (Live Prediction · Performance · Sentiment · Valuation · Architecture)
```

### Model Architecture

```
Input (30 days × 26 features)
        ↓
LSTM(128, return_sequences=True)  →  Dropout(0.2)
        ↓
LSTM(64, return_sequences=True)   →  Dropout(0.2)
        ↓
LSTM(32, return_sequences=False)  →  Dropout(0.2)
        ↓
BatchNormalization
        ↓
Dense(64, relu)  →  Dropout(0.1)  →  Dense(32, relu)
        ↓
Dense(1)  ←── Regression: next-day return → price
Dense(1, sigmoid)  ←── Classifier: Up/Down probability
```

---

## 📊 Results

### Regression — Price Prediction

| Metric | Value | Notes |
|--------|-------|-------|
| **R² Score** | **0.9768** | Excellent — above 0.95 threshold |
| **MAPE** | **0.76%** | Sub-1% mean absolute percentage error |
| **MAE** | **₹7.00** | Mean absolute error on ₹363–₹1,012 range |
| **RMSE** | **₹9.22** | Root mean squared error |

### Classification — Direction Prediction

| Metric | Value | Notes |
|--------|-------|-------|
| **Accuracy** | **52.7%** | Marginally above random (50%) |
| **Finding** | **EMH** | Consistent with Efficient Market Hypothesis |

> **Key Insight:** LSTM outputs probabilities in [0.477, 0.484] with std=0.002 — the model outputs only the dataset prior. HDFC Bank is a large-cap, heavily traded stock where all public information is already priced in. Daily direction is near-random walk and is not learnable from price + sentiment alone. This is consistent with **Weak Form EMH** (Fama, 1970).

---

## 🔬 Feature Engineering — 26 Features

| Category | Features |
|----------|----------|
| **Price & Volume (5)** | Open, High, Low, Close, Volume |
| **Returns & Volatility (7)** | returns, log_returns, vol_5d, vol_10d, hl_ratio, co_ratio, overnight_gap |
| **Lag Features (3)** | lag1_return, lag2_return, lag1_direction |
| **Technical Indicators (7)** | RSI-14, MACD, MACD Signal, MACD Hist, BB Upper, BB Lower, BB Width |
| **Momentum (2)** | momentum_5d, momentum_10d |
| **Sentiment (4)** | daily_compound, sentiment_ma3, sentiment_ma7, sentiment_ma14 |

---

## 🗂 Project Structure

```
hdfc-lstm-predictor/
├── scraper_price.py     # Yahoo Finance OHLCV scraper — 6 years of HDFCBANK.NS
├── scraper_news.py      # Google News RSS scraper + VADER sentiment scoring
├── preprocess.py        # Feature engineering, sequence building, train/test split
├── train.py             # Dual LSTM training — regression + classifier
├── app.py               # Streamlit dashboard — 6 pages
├── hdfc_logo.png        # HDFC Bank logo for nav bar
├── hdfcbank_prices.csv  # Raw price data
├── hdfcbank_news.csv    # Raw news headlines with sentiment scores
└── requirements.txt     # Python dependencies
```

---

## 🌐 Streamlit App — 6 Pages

| Page | Features |
|------|---------|
| 🏠 **Home** | Hero landing · live price · metric strip · model summary |
| 📈 **Live Prediction** | Next NSE trading day forecast · confidence · direction signal |
| 📊 **Model Performance** | Predicted vs actual chart · scatter · R² · training curves |
| 📰 **Sentiment** | Price vs sentiment overlay · yearly breakdown · recent headlines |
| 💰 **Valuation** | DCF + CAPM · live financials · 5Y chart · income statement · balance sheet |
| 🧠 **Architecture** | Layer-by-layer model cards · feature pipeline · EMH finding |

---

## 🛠 Run Locally

```bash
# 1. Clone
git clone https://github.com/065010-AmanMalhi/hdfc-lstm-predictor.git
cd hdfc-lstm-predictor

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install
pip install -r requirements.txt

# 4. Run pipeline in order
python scraper_price.py      # → hdfcbank_prices.csv
python scraper_news.py       # → hdfcbank_news.csv
python preprocess.py         # → X_train.npy, feature_cols.pkl, scalers
python train.py              # → model_regression.keras, model_classifier.keras

# 5. Launch
streamlit run app.py
```

> **Note:** Model files (`.keras`, `.pkl`, `.npy`) are not included in the repo due to size. Run the pipeline above to generate them. The app auto-fetches live prices and financials via yfinance.

---

## 🧠 Key Technical Decisions

**Why predict returns instead of raw price?**
Raw price is non-stationary — it trends upward over 6 years. LSTM learns patterns better on stationary data (returns oscillate around 0%). We predict the return then convert back to ₹ price using `price × (1 + return)`.

**Why VADER for news sentiment?**
VADER is specifically tuned for short financial headlines — handles ALL CAPS, negations, and financial terminology without needing a trained model. Fast, interpretable, and well-validated on financial text.

**Why document the direction failure?**
A model that honestly reports its limitations is more credible than one that cherry-picks metrics. The EMH finding is academically grounded and demonstrates understanding of market microstructure — a stronger signal than a suspiciously perfect classifier.

---

## 📦 Tech Stack

`Python 3.12` · `TensorFlow 2.20` · `Keras 3.13` · `Streamlit` · `Plotly` · `yfinance` · `VADER Sentiment` · `scikit-learn` · `pandas` · `feedparser`

---

<div align="center">
Project 2 of 2 in the RNN Series<br/>
<b>Project 1:</b> <a href="https://github.com/065010-AmanMalhi/groww-sentiment-analyser">Groww Review Sentiment Analyser</a> — 92.17% accuracy on 3,000 Play Store reviews
</div>
