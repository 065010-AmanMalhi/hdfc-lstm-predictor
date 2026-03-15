"""
app.py v4 — HDFC Bank LSTM Stock Analyser
Dark navy theme · Bloomberg-terminal aesthetic
"""

import json, pickle, warnings, os, base64
from download_model import download_if_missing
download_if_missing()
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
import yfinance as yf

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="HDFC Bank LSTM Analyser",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown("""<style>
.block-container{padding-top:0!important;margin-top:0!important}
[data-testid="stAppViewBlockContainer"]{padding-top:0!important}
.stMainBlockContainer{padding-top:0!important}
</style>""", unsafe_allow_html=True)

# Must be first st call — kills ALL Streamlit default spacing
st.markdown("""
<style>
    .block-container {padding:0 !important; margin:0 !important;}
    .main > div {padding:0 !important; margin:0 !important;}
    section[data-testid="stMain"] > div {padding:0 !important;}
    div[data-testid="stVerticalBlock"] {gap:0 !important;}
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
        padding:0 !important; margin:0 !important;
    }
    iframe {border:none !important;}
</style>
""", unsafe_allow_html=True)



# ── HDFC Off-White Theme ──────────────────────────────────────────────────────
D_BG      = "#f5f6fa"       # HDFC off-white background
D_SURFACE = "#ffffff"       # card surface white
D_SURFACE2= "#eef1f8"       # elevated surface
D_BORDER  = "#d8dff0"       # border
D_BLUE    = "#004C8F"       # HDFC official blue
D_BLUE2   = "#0066CC"       # HDFC lighter blue
D_RED     = "#EE3124"       # HDFC official red
D_GREEN   = "#16a34a"       # green
D_GOLD    = "#d97706"       # amber
D_TEAL    = "#0891b2"       # teal
D_TEXT    = "#0d1b2e"       # dark navy text
D_MUTED   = "#6b7280"       # muted text
D_MUTED2  = "#374151"       # slightly brighter muted
D_HDFC    = "#004C8F"       # HDFC official blue
D_HDFC_R  = "#EE3124"       # HDFC official red
TEMPLATE  = "plotly_white"
MAX_LEN   = 30

def get_logo_b64():
    logo_path = os.path.join(os.path.dirname(__file__), "hdfc_logo.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

LOGO_B64  = get_logo_b64()
LOGO_HTML = f'<img src="data:image/png;base64,{LOGO_B64}" style="height:36px;width:auto;object-fit:contain;">' if LOGO_B64 else '<div style="width:36px;height:36px;background:{D_BLUE};border-radius:6px;display:flex;align-items:center;justify-content:center;font-weight:900;color:white;font-size:1rem;">H</div>'

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after {{
    font-family: 'Inter', sans-serif !important;
    box-sizing: border-box;
}}
html, body, [class*="css"], .stApp {{
    background: {D_BG} !important;
    color: {D_TEXT} !important;
}}
section[data-testid="stSidebar"] {{ display: none !important; }}mportant;
}}
header[data-testid="stHeader"]   {{ display: none !important; }}
.block-container {{ padding: 0 !important; max-width: 100% !important; }}
div[data-testid="stAppViewContainer"] > section > div.block-container {{
    padding: 0 !important;
    max-width: 100% !important;
}}
div[data-testid="stAppViewContainer"] {{
    background: {D_BG} !important;
}}
/* Remove ALL default Streamlit padding/margins */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    padding: 0 !important;
    margin: 0 !important;
    background: {D_BG} !important;
}}
div[data-testid="stHorizontalBlock"] > div {{
    padding: 0 !important;
    margin: 0 !important;
    gap: 0 !important;
    flex: 1 !important;
}}
div[data-testid="stVerticalBlock"] > div {{
    padding: 0 !important;
    margin: 0 !important;
    margin-top: 0 !important;
}}
div[data-testid="stVerticalBlock"] {{
    gap: 0 !important;
}}
.main .block-container > div {{
    padding: 0 !important;
    margin: 0 !important;
}}
footer {{ display: none !important; }}

/* ── TOP NAV ── */
.d-nav {{
    background: {D_SURFACE};
    border-bottom: 2px solid {D_BORDER};
    padding: 20 40px;
    display: flex; align-items: center; height: 64px;
    position: sticky; top: 0; z-index: 1000;
    box-shadow: 0 1px 30px rgba(0,0,0,0.5);
}}
.d-brand {{
    display: flex; align-items: center; gap: 12px;
    padding-right: 40px; margin-right: 8px;
    border-right: 1px solid {D_BORDER};
}}
.d-brand-name {{ font-size: 1.1rem; font-weight: 700; color: {D_TEXT}; letter-spacing: -0.2px; }}
.d-brand-sub  {{ font-size: 0.75rem; color: {D_MUTED}; letter-spacing: 1.5px; text-transform: uppercase; }}

.d-live {{
    margin-left: auto;
    display: flex; align-items: center; gap: 10px;
    background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.25);
    padding: 6px 16px; border-radius: 99px;
}}
.d-live-dot {{
    width: 7px; height: 7px; border-radius: 50%; background: {D_GREEN};
    animation: blink 1.5s infinite;
}}
.d-live-price {{
    font-size: 0.9rem; font-weight: 800; color: {D_TEXT};
    font-family: 'JetBrains Mono', monospace !important;
}}
.d-live-change {{ font-size: 0.78rem; font-weight: 600; }}
@keyframes blink {{ 0%,100%{{opacity:1;}} 50%{{opacity:0.2;}} }}

/* Hero CSS removed */

/* Hero metric strip */
.d-metric-strip {{
    display: flex; gap: 1px; flex-wrap: wrap;
    background: {D_BORDER}; border-radius: 14px;
    overflow: hidden; width: fit-content;
    border: 1px solid {D_BORDER};
}}
.d-metric-item {{
    background: {D_SURFACE}; padding: 20px 28px; min-width: 150px;
    border-right: 1px solid {D_BORDER};
}}
.d-metric-item:first-child {{ border-radius: 13px 0 0 13px; }}
.d-metric-item:last-child  {{ border-radius: 0 13px 13px 0; }}
.d-metric-val {{
    font-size: 1.9rem; font-weight: 800; letter-spacing: -1px;
    font-family: 'JetBrains Mono', monospace !important;
    line-height: 1.1; color: inherit;
}}
.d-metric-lbl {{
    font-size: 0.66rem; color: {D_MUTED}; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1.2px; margin-top: 6px;
}}

/* Hero CTA buttons */
.d-hero-cta {{ display: flex; gap: 12px; margin-top: 48px; flex-wrap: wrap; }}
.d-cta-primary {{
    background: {D_BLUE}; color: white; font-weight: 700; font-size: 0.95rem;
    padding: 12px 28px; border-radius: 10px; border: none; cursor: pointer;
    box-shadow: 0 4px 20px rgba(59,130,246,0.35); letter-spacing: 0.2px;
}}
.d-cta-secondary {{
    background: {D_SURFACE2}; color: {D_TEXT}; font-weight: 600; font-size: 0.95rem;
    padding: 12px 28px; border-radius: 10px; border: 1px solid {D_BORDER}; cursor: pointer;
}}



/* ── PAGE CONTENT ── */
.d-page {{ padding: 40px 56px; max-width: 1200px; }}
.d-page-header {{ margin-bottom: 32px; }}
.d-page-title {{
    font-size: 1.8rem; font-weight: 800; color: {D_TEXT};
    letter-spacing: -0.5px; margin-bottom: 6px;
    font-family: 'Georgia', serif;
}}
.d-page-sub {{ font-size: 0.9rem; color: {D_MUTED2}; font-weight: 400; line-height: 1.5; }}
.d-divider {{ border: none; border-top: 1px solid {D_BORDER}; margin: 24px 0; }}

/* ── CARDS ── */
.d-card {{
    background: {D_SURFACE}; border: 1px solid {D_BORDER};
    border-radius: 12px; padding: 24px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.2);
    margin-bottom: 20px;
}}
.d-card-title {{
    font-size: 0.75rem; font-weight: 700; color: {D_MUTED};
    text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 12px;
}}
.pred-pos {{
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 2px solid #16a34a; border-radius: 14px;
    padding: 32px; text-align: center;
}}
.pred-neg {{
    background: linear-gradient(135deg, #fef2f2, #fee2e2);
    border: 2px solid #EE3124; border-radius: 14px;
    padding: 32px; text-align: center;
}}

/* ── BUTTONS ── */
.stButton > button {{
    background: transparent !important; color: {D_MUTED2} !important;
    border: none !important; border-radius: 8px !important;
    padding: 8px 16px !important; font-size: 0.88rem !important; font-weight: 500 !important;
    transition: background 0.15s !important; box-shadow: none !important;
    white-space: nowrap !important; outline: none !important;
}}
.stButton > button:hover {{
    background: rgba(0,76,143,0.08) !important;
    color: {D_BLUE} !important;
    transform: none !important;
    box-shadow: none !important;
}}
.predict-btn .stButton > button {{
    background: {D_BLUE} !important; color: white !important;
    border: 1px solid rgba(96,165,250,0.5) !important;
    border-radius: 10px !important; padding: 14px 24px !important;
    font-size: 1rem !important; font-weight: 800 !important;
    box-shadow: 0 4px 20px rgba(59,130,246,0.3) !important;
    letter-spacing: 0.3px !important;
}}
.predict-btn .stButton > button:hover {{
    background: #2563eb !important; transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(59,130,246,0.4) !important; color: white !important;
}}

/* ── STREAMLIT OVERRIDES ── */
div[data-testid="stMetric"] {{
    background: {D_SURFACE} !important; border: 1px solid {D_BORDER} !important;
    border-radius: 10px !important; padding: 16px !important;
}}
div[data-testid="stMetricLabel"] p {{ color: {D_MUTED2} !important; font-size: 0.8rem !important; }}
div[data-testid="stMetricValue"] {{ color: {D_BLUE2} !important; font-size: 1.5rem !important; font-weight: 800 !important; font-family: 'JetBrains Mono', monospace !important; }}
div[data-testid="stMetricDelta"] {{ font-size: 0.78rem !important; }}
.stDataFrame {{ border-radius: 10px !important; overflow: hidden !important; }}
.stDataFrame [data-testid="stDataFrameResizable"] {{ background: {D_SURFACE} !important; }}
.stAlert {{ border-radius: 10px !important; background: {D_SURFACE2} !important; border: 1px solid {D_BORDER} !important; }}
textarea, input {{ background: {D_SURFACE2} !important; color: {D_TEXT} !important; border: 1px solid {D_BORDER} !important; border-radius: 8px !important; }}

/* Kill white gaps */
.main, section.main, .block-container,
[data-testid="stAppViewContainer"],
[data-testid="stVerticalBlock"],
[data-testid="column"] {{
    background: transparent !important;
}}
.stApp {{ background: {D_BG} !important; }}

/* Nav tab area - transparent bg */
[data-testid="stHorizontalBlock"] {{
    background: transparent !important;
    gap: 0 !important;
}}

/* Spinner transparent */
[data-testid="stSpinner"] > div {{ background: transparent !important; }}

/* Caption */
.stCaption p {{ color: {D_MUTED} !important; font-size: 0.75rem !important; }}

/* Metric delta color fix */
[data-testid="stMetricDelta"] svg {{ display: none !important; }}

/* Architecture cards */
.arch-card {{
    background: {D_SURFACE}; border: 1px solid {D_BORDER};
    border-radius: 12px; overflow: hidden;
}}
.arch-header {{
    padding: 16px 20px; font-weight: 700; font-size: 0.95rem;
    display: flex; align-items: center; gap: 10px;
}}
.arch-layer {{
    display: flex; align-items: center; padding: 10px 20px;
    border-top: 1px solid {D_BORDER}; gap: 12px;
    transition: background 0.15s;
}}
.arch-layer:hover {{ background: #f0f4ff; }}
.arch-layer-icon {{
    width: 32px; height: 32px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; font-weight: 700; flex-shrink: 0;
}}
.arch-layer-name {{ font-size: 0.88rem; font-weight: 600; color: {D_TEXT}; }}
.arch-layer-sub  {{ font-size: 0.75rem; color: {D_MUTED}; margin-top: 1px; }}
.arch-arrow {{
    text-align: center; padding: 4px; color: {D_MUTED};
    font-size: 0.8rem; border-left: 2px dashed {D_BORDER};
    margin-left: 35px; padding-left: 26px; height: 18px;
    display: flex; align-items: center;
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    from tensorflow.keras.models import load_model
    return load_model("model_regression.keras"), load_model("model_classifier.keras")

@st.cache_data
def load_artifacts():
    with open("scaler_ret.pkl","rb") as f:      sr = pickle.load(f)
    with open("scaler_features.pkl","rb") as f: sx = pickle.load(f)
    with open("feature_cols.pkl","rb") as f:    fc = pickle.load(f)
    with open("cls_threshold.pkl","rb") as f:   th = pickle.load(f)
    return sr, sx, fc, th

@st.cache_data
def load_data():
    df   = pd.read_csv("features.csv", parse_dates=["Date"])
    pred = pd.read_csv("test_predictions.csv")
    with open("training_history.json") as f: hist = json.load(f)
    news = pd.read_csv("hdfcbank_news.csv", parse_dates=["date"])
    return df, pred, hist, news

@st.cache_data(ttl=300)
def fetch_live():
    try:
        t = yf.Ticker("HDFCBANK.NS")
        h = t.history(period="5d")
        if h.empty: raise ValueError("empty")
        price   = float(h["Close"].iloc[-1])
        dt      = h.index[-1]
        change  = float(h["Close"].iloc[-1] - h["Close"].iloc[-2])
        chg_pct = change / float(h["Close"].iloc[-2]) * 100
        return price, dt, change, chg_pct, True
    except:
        return None, None, None, None, False

@st.cache_data(ttl=300)
def fetch_full():
    try:
        t   = yf.Ticker("HDFCBANK.NS")
        inf = t.info
        h   = t.history(period="5y")
        fin = t.financials
        bs  = t.balance_sheet
        return inf, h, fin, bs, True
    except:
        return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), False


# ══════════════════════════════════════════════════════════════════════════════
# DATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def next_nse_day():
    today   = date.today()
    wd      = today.weekday()
    delta   = 3 if wd == 4 else (2 if wd == 5 else (1 if wd == 6 else 1))
    return (today + timedelta(days=delta)).strftime("%A, %d %b %Y")

def last_trading_label(dt):
    """Return 'Today' if dt was today's weekday, else 'Last Trading Day'."""
    today = date.today()
    if hasattr(dt, 'date'):
        dt_date = dt.date()
    else:
        dt_date = dt
    return "Today" if dt_date == today else "Last Trading Day"


# ══════════════════════════════════════════════════════════════════════════════
# INIT
# ══════════════════════════════════════════════════════════════════════════════

if "page" not in st.session_state:
    st.session_state.page = "home"

reg_model, cls_model               = load_models()
scaler_ret, scaler_X, feat_cols, thresh = load_artifacts()
df, pred_df, hist_data, news_df    = load_data()
live_price, live_dt, live_chg, live_chg_pct, live_ok = fetch_live()

display_price = live_price if live_ok else float(df["Close"].iloc[-1])
display_label = last_trading_label(live_dt) if live_ok else "Last Trading Day"
display_date  = live_dt.strftime("%d %b %Y") if live_ok and live_dt else pd.to_datetime(df["Date"].iloc[-1]).strftime("%d %b %Y")
next_day      = next_nse_day()

chg_color = D_GREEN if (live_chg or 0) >= 0 else D_RED
chg_str   = f"{'▲' if (live_chg or 0)>=0 else '▼'} {abs(live_chg or 0):.2f} ({live_chg_pct or 0:+.2f}%)" if live_ok else ""


# ══════════════════════════════════════════════════════════════════════════════
# TOP NAV
# ══════════════════════════════════════════════════════════════════════════════




# ── TOP NAV ──────────────────────────────────────────────────────────────────
pages = [
    ("home",        "🏠  Home"),
    ("prediction",  "📈  Live Prediction"),
    ("performance", "📊  Model Performance"),
    ("sentiment",   "📰  Sentiment"),
    ("valuation",   "💰  Valuation"),
    ("architecture","🧠  Architecture"),
]

st.markdown(f"""
<div style="background:{D_SURFACE};border-bottom:1px solid {D_BORDER};
            padding:0 32px;display:flex;align-items:center;
            gap:16px;height:60px;
            box-shadow:0 1px 8px rgba(0,76,143,0.08);">
  {LOGO_HTML}
  <div>
    <div style="font-size:1rem;font-weight:800;color:{D_TEXT};
                font-family:'Georgia',serif;line-height:1.1;">HDFC Bank</div>
    <div style="font-size:0.58rem;color:{D_MUTED};text-transform:uppercase;
                letter-spacing:1.5px;font-weight:600;">NSE: HDFCBANK · LSTM AI</div>
  </div>
  <div style="margin-left:auto;display:flex;align-items:center;gap:8px;
              background:rgba(22,163,74,0.08);border:1px solid rgba(22,163,74,0.2);
              padding:7px 16px;border-radius:99px;">
    <div style="width:7px;height:7px;border-radius:50%;background:{D_GREEN};
                animation:blink 1.5s infinite;"></div>
    <span style="font-size:0.95rem;font-weight:800;color:{D_TEXT};
                 font-family:'JetBrains Mono',monospace;">₹{display_price:,.2f}</span>
    <span style="font-size:0.8rem;font-weight:700;color:{chg_color};">{chg_str}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Tab row
tab_cols = st.columns(len(pages))
for col, (pid, label) in zip(tab_cols, pages):
    with col:
        if st.session_state.page == pid:
            st.markdown(f"""<div style="background:{D_SURFACE};
                border-bottom:3px solid {D_BLUE};padding:9px 4px;text-align:center;
                font-size:0.85rem;font-weight:700;color:{D_BLUE};">
                {label}</div>""", unsafe_allow_html=True)
        else:
            if st.button(label, key=f"nav_{pid}", use_container_width=True):
                st.session_state.page = pid
                st.rerun()

st.markdown(f'<div style="height:1px;background:{D_BORDER};"></div>',
            unsafe_allow_html=True)

page = st.session_state.page


# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════

if page == "home":
    st.markdown(f"""
    <div style="background:linear-gradient(160deg,#eef2fb 0%,#f5f6fa 60%,#fef9f9 100%);
                padding:56px 56px 52px;position:relative;overflow:hidden;">

      <div style="position:absolute;inset:0;pointer-events:none;
                  background-image:linear-gradient(rgba(0,76,143,0.03) 1px,transparent 1px),
                  linear-gradient(90deg,rgba(0,76,143,0.03) 1px,transparent 1px);
                  background-size:60px 60px;"></div>

      <div style="display:inline-flex;align-items:center;gap:8px;
                  background:rgba(0,76,143,0.08);border:1px solid rgba(0,76,143,0.18);
                  color:{D_BLUE2};font-size:0.72rem;font-weight:600;
                  padding:5px 14px;border-radius:99px;text-transform:uppercase;
                  letter-spacing:1.5px;margin-bottom:20px;">
        LSTM · Deep Learning · Quantitative Finance · NSE India
      </div>

      <div style="font-size:4rem;font-weight:900;line-height:1.06;
                  letter-spacing:-2px;margin-bottom:18px;font-family:'Georgia',serif;">
        <span style="color:{D_BLUE};">HDFC Bank</span><br>
        Stock <span style="color:{D_RED};">Intelligence</span>
      </div>

      <div style="display:inline-flex;align-items:center;gap:12px;
                  background:rgba(0,76,143,0.05);border:1px solid rgba(0,76,143,0.15);
                  padding:10px 20px;border-radius:99px;margin-bottom:20px;">
        <div style="width:8px;height:8px;border-radius:50%;background:{D_GREEN};
                    animation:blink 1.5s infinite;"></div>
        <span style="font-size:1.3rem;font-weight:900;color:{D_TEXT};
                     font-family:'JetBrains Mono',monospace;">₹{display_price:,.2f}</span>
        <span style="font-size:0.88rem;font-weight:700;color:{chg_color};">{chg_str}</span>
        <span style="font-size:0.75rem;color:{D_MUTED};font-weight:600;">
          {display_label} · {display_date}
        </span>
      </div>

      <div style="font-size:1.05rem;color:{D_MUTED2};max-width:620px;line-height:1.7;margin-bottom:44px;">
        A dual LSTM architecture trained on <b style="color:{D_TEXT};">6 years</b> of NSE price data
        and <b style="color:{D_TEXT};">5,422 news headlines</b> to predict HDFC Bank's next trading day
        closing price. Achieved <b style="color:{D_TEXT};">R² = 0.9768</b> and
        <b style="color:{D_TEXT};">MAPE = 0.76%</b> on held-out test data.
      </div>

      <div style="display:flex;gap:1px;background:{D_BORDER};border-radius:14px;
                  overflow:hidden;border:1px solid {D_BORDER};width:fit-content;margin-bottom:44px;">
        <div style="background:{D_SURFACE};padding:18px 26px;min-width:130px;border-right:1px solid {D_BORDER};">
          <div style="font-size:1.7rem;font-weight:800;color:{D_GREEN};font-family:'JetBrains Mono',monospace;letter-spacing:-1px;">0.9768</div>
          <div style="font-size:0.64rem;color:{D_MUTED};font-weight:600;text-transform:uppercase;letter-spacing:1.2px;margin-top:5px;">Regression R²</div>
        </div>
        <div style="background:{D_SURFACE};padding:18px 26px;min-width:130px;border-right:1px solid {D_BORDER};">
          <div style="font-size:1.7rem;font-weight:800;color:{D_GOLD};font-family:'JetBrains Mono',monospace;letter-spacing:-1px;">0.76%</div>
          <div style="font-size:0.64rem;color:{D_MUTED};font-weight:600;text-transform:uppercase;letter-spacing:1.2px;margin-top:5px;">Price MAPE</div>
        </div>
        <div style="background:{D_SURFACE};padding:18px 26px;min-width:130px;border-right:1px solid {D_BORDER};">
          <div style="font-size:1.7rem;font-weight:800;color:{D_BLUE};font-family:'JetBrains Mono',monospace;letter-spacing:-1px;">₹7.00</div>
          <div style="font-size:0.64rem;color:{D_MUTED};font-weight:600;text-transform:uppercase;letter-spacing:1.2px;margin-top:5px;">Mean Abs Error</div>
        </div>
        <div style="background:{D_SURFACE};padding:18px 26px;min-width:130px;border-right:1px solid {D_BORDER};">
          <div style="font-size:1.7rem;font-weight:800;color:{D_TEAL};font-family:'JetBrains Mono',monospace;letter-spacing:-1px;">5,422</div>
          <div style="font-size:0.64rem;color:{D_MUTED};font-weight:600;text-transform:uppercase;letter-spacing:1.2px;margin-top:5px;">News Headlines</div>
        </div>
        <div style="background:{D_SURFACE};padding:18px 26px;min-width:130px;">
          <div style="font-size:1.7rem;font-weight:800;color:{D_MUTED2};font-family:'JetBrains Mono',monospace;letter-spacing:-1px;">1,537</div>
          <div style="font-size:0.64rem;color:{D_MUTED};font-weight:600;text-transform:uppercase;letter-spacing:1.2px;margin-top:5px;">Trading Days</div>
        </div>
      </div>

      <div style="display:flex;gap:14px;flex-wrap:wrap;">
        <div style="background:{D_SURFACE};border:1px solid {D_BORDER};border-left:3px solid {D_BLUE};
                    border-radius:10px;padding:16px 20px;min-width:180px;">
          <div style="font-size:0.66rem;color:{D_MUTED};text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:6px;">Next Prediction</div>
          <div style="font-size:0.92rem;font-weight:700;color:{D_TEXT};">{next_day}</div>
          <div style="font-size:0.74rem;color:{D_MUTED};margin-top:3px;">NSE next trading day</div>
        </div>
        <div style="background:{D_SURFACE};border:1px solid {D_BORDER};border-left:3px solid {D_RED};
                    border-radius:10px;padding:16px 20px;min-width:180px;">
          <div style="font-size:0.66rem;color:{D_MUTED};text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:6px;">Architecture</div>
          <div style="font-size:0.92rem;font-weight:700;color:{D_TEXT};">LSTM 128 → 64 → 32</div>
          <div style="font-size:0.74rem;color:{D_MUTED};margin-top:3px;">3-layer stacked + BatchNorm</div>
        </div>
        <div style="background:{D_SURFACE};border:1px solid {D_BORDER};border-left:3px solid {D_GOLD};
                    border-radius:10px;padding:16px 20px;min-width:180px;">
          <div style="font-size:0.66rem;color:{D_MUTED};text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:6px;">Feature Set</div>
          <div style="font-size:0.92rem;font-weight:700;color:{D_TEXT};">26 Features</div>
          <div style="font-size:0.74rem;color:{D_MUTED};margin-top:3px;">Price · Technical · Sentiment</div>
        </div>
        <div style="background:{D_SURFACE};border:1px solid {D_BORDER};border-left:3px solid {D_TEAL};
                    border-radius:10px;padding:16px 20px;min-width:180px;">
          <div style="font-size:0.66rem;color:{D_MUTED};text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:6px;">Lookback Window</div>
          <div style="font-size:0.92rem;font-weight:700;color:{D_TEXT};">30 Trading Days</div>
          <div style="font-size:0.74rem;color:{D_MUTED};margin-top:3px;">Sequence length fed to LSTM</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "prediction":
    st.markdown('<div class="d-page">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="d-page-header">
      <div class="d-page-title">📈 Live Price Prediction</div>
      <div class="d-page-sub">LSTM forecasting next NSE trading day close &nbsp;·&nbsp;
        <span style="color:{D_BLUE2};font-weight:600;">{next_day}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([1.1, 1], gap="large")

    with col_l:
        arrow2 = "▲" if (live_chg or 0) >= 0 else "▼"
        c2     = D_GREEN if (live_chg or 0) >= 0 else D_RED
        st.markdown(f"""
        <div class="d-card" style="border-left:3px solid {D_BLUE}; margin-bottom:20px;">
          <div class="d-card-title">{display_label} · {display_date}</div>
          <div style="display:flex;align-items:baseline;gap:14px;">
            <span style="font-size:2.8rem;font-weight:900;color:{D_TEXT};font-family:'JetBrains Mono',monospace;">₹{display_price:,.2f}</span>
            <span style="font-size:1rem;font-weight:700;color:{c2};">{arrow2} ₹{abs(live_chg or 0):.2f} ({live_chg_pct or 0:+.2f}%)</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div style="font-size:0.8rem;color:{D_MUTED};font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">Last 10 Trading Days (30-day window fed to LSTM)</div>', unsafe_allow_html=True)
        recent = df.tail(10)[["Date","Open","High","Low","Close","returns","daily_compound"]].copy()
        recent["Date"]           = pd.to_datetime(recent["Date"]).dt.strftime("%d %b %Y")
        recent[["Open","High","Low","Close"]] = recent[["Open","High","Low","Close"]].round(2)
        recent["returns"]        = (recent["returns"]*100).round(3)
        recent["daily_compound"] = recent["daily_compound"].round(4)
        recent.columns           = ["Date","Open","High","Low","Close (₹)","Return (%)","Sentiment"]
        st.dataframe(recent, use_container_width=True, hide_index=True)

        st.markdown('<div class="predict-btn" style="margin-top:16px;">', unsafe_allow_html=True)
        predict_btn = st.button(f"🔮  Predict Close — {next_day}", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        if predict_btn:
            with st.spinner("Running LSTM inference..."):
                seq  = scaler_X.transform(df[feat_cols].values[-MAX_LEN:])
                X_in = seq.reshape(1, MAX_LEN, len(feat_cols))
                base = display_price
                rs   = float(reg_model.predict(X_in, verbose=0)[0][0])
                ret  = float(scaler_ret.inverse_transform([[rs]])[0][0])
                pp   = base * (1 + ret)
                prob = float(cls_model.predict(X_in, verbose=0)[0][0])
                up   = prob > thresh

            chg   = pp - base
            chgp  = chg / base * 100
            css   = "pred-pos" if up else "pred-neg"
            col   = D_GREEN if up else D_RED
            arr   = "▲" if up else "▼"

            st.markdown(f"""
            <div class="{css}">
              <div style="font-size:0.7rem;color:{D_MUTED};font-weight:700;text-transform:uppercase;
                          letter-spacing:1.5px;margin-bottom:12px;">
                Predicted Close · {next_day}
              </div>
              <div style="font-size:3.4rem;font-weight:900;color:{col};letter-spacing:-2px;
                          margin-bottom:8px;font-family:'JetBrains Mono',monospace;">
                ₹{pp:,.2f}
              </div>
              <div style="font-size:1.2rem;font-weight:700;color:{col};">
                {arr} ₹{abs(chg):,.2f} &nbsp; ({chgp:+.2f}%)
              </div>
              <div style="font-size:0.82rem;color:{D_MUTED};margin-top:12px;font-family:'JetBrains Mono',monospace;">
                from {display_label.lower()} close ₹{base:,.2f}
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("Predicted Return", f"{ret*100:+.4f}%")
            c2.metric("Direction Signal", "↑ Up" if up else "↓ Down")
            st.info("**R² = 0.9768** on test set · Direction ~52.7% (EMH — large-cap efficient market)", icon="ℹ️")
        else:
            st.markdown(f"""
            <div style="text-align:center;padding:80px 24px;border:1px dashed {D_BORDER};
                        border-radius:14px;background:{D_SURFACE};">
              <div style="font-size:3rem;margin-bottom:16px;opacity:0.4;">🔮</div>
              <div style="color:{D_MUTED2};font-size:0.95rem;font-weight:500;line-height:1.6;">
                Click <b style="color:{D_TEXT};">Predict Close — {next_day}</b><br>
                to run the LSTM forecast
              </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "performance":
    st.markdown('<div class="d-page">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="d-page-header">
      <div class="d-page-title">📊 Model Performance</div>
      <div class="d-page-sub">Held-out chronological test set · most recent 20% of data · 298 trading days.</div>
    </div>
    """, unsafe_allow_html=True)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae  = mean_absolute_error(pred_df["actual_price"], pred_df["predicted_price"])
    rmse = np.sqrt(mean_squared_error(pred_df["actual_price"], pred_df["predicted_price"]))
    r2   = r2_score(pred_df["actual_price"], pred_df["predicted_price"])
    mape = np.mean(np.abs((pred_df["actual_price"]-pred_df["predicted_price"])/pred_df["actual_price"]))*100

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("R² Score", f"{r2:.4f}", delta="Excellent — >0.95 threshold")
    c2.metric("MAPE",     f"{mape:.2f}%", delta="Sub-1% error rate")
    c3.metric("MAE",      f"₹{mae:.2f}")
    c4.metric("RMSE",     f"₹{rmse:.2f}")
    st.markdown("<br>", unsafe_allow_html=True)

    plotly_bg = D_SURFACE; plotly_grid = D_BORDER

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=pred_df["actual_price"], name="Actual",
                              line=dict(color=D_BLUE, width=2.5),
                              hovertemplate="Day %{x}<br>Actual: ₹%{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(y=pred_df["predicted_price"], name="Predicted",
                              line=dict(color=D_RED, width=2, dash="dash"),
                              hovertemplate="Day %{x}<br>Predicted: ₹%{y:,.2f}<extra></extra>"))
    n = len(pred_df)
    fig.add_trace(go.Scatter(
        y=list(pred_df["predicted_price"]*1.01)+list(pred_df["predicted_price"]*0.99)[::-1],
        x=list(range(n))+list(range(n))[::-1],
        fill="toself", fillcolor=f"rgba(239,68,68,0.06)",
        line=dict(color="rgba(0,0,0,0)"), name="±1% Band"
    ))
    fig.update_layout(
        title=dict(text="Predicted vs Actual Closing Price — Test Set (298 days)", font=dict(size=15, family="Inter")),
        xaxis=dict(title="Trading Day", showgrid=True, gridcolor=plotly_grid),
        yaxis=dict(title="Price (₹)", showgrid=True, gridcolor=plotly_grid, tickprefix="₹", tickformat=",.0f"),
        template=TEMPLATE, height=440,
        paper_bgcolor=D_SURFACE, plot_bgcolor='#f8f9ff',
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=D_BORDER),
        font=dict(family="Inter"), hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    mn = pred_df["actual_price"].min(); mx = pred_df["actual_price"].max()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=pred_df["actual_price"], y=pred_df["predicted_price"],
                               mode="markers", name="Predictions",
                               marker=dict(color=D_BLUE, size=5, opacity=0.6),
                               hovertemplate="Actual: ₹%{x:,.2f}<br>Predicted: ₹%{y:,.2f}<extra></extra>"))
    fig2.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines", name="Perfect Fit",
                               line=dict(color=D_RED, dash="dash", width=1.5)))
    fig2.add_annotation(x=mn+(mx-mn)*0.07, y=mx-(mx-mn)*0.07,
                         text=f"<b>R² = {r2:.4f}</b>", showarrow=False,
                         font=dict(size=14, color=D_BLUE2, family="Inter"),
                         bgcolor="white", bordercolor=D_BORDER, borderwidth=1)
    fig2.update_layout(
        title=dict(text="Actual vs Predicted Scatter (R² = 0.9768)", font=dict(size=15, family="Inter")),
        xaxis=dict(title="Actual (₹)", tickprefix="₹", tickformat=",.0f", showgrid=True, gridcolor=plotly_grid),
        yaxis=dict(title="Predicted (₹)", tickprefix="₹", tickformat=",.0f", showgrid=True, gridcolor=plotly_grid),
        template=TEMPLATE, height=420, paper_bgcolor=D_SURFACE, plot_bgcolor='#f8f9ff',
        font=dict(family="Inter"), legend=dict(bgcolor="rgba(255,255,255,0.9)"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    if "reg_loss" in hist_data:
        epochs     = list(range(1, len(hist_data["reg_loss"])+1))
        best_epoch = int(np.argmin(hist_data["reg_val_loss"]))+1
        fig3 = make_subplots(rows=1, cols=2, subplot_titles=("Loss (MSE)", "MAE"))
        for ci,(tk,vk) in enumerate([("reg_loss","reg_val_loss"),("reg_mae","reg_val_mae")],1):
            fig3.add_trace(go.Scatter(x=epochs, y=hist_data[tk], name="Train",
                                       line=dict(color=D_BLUE, width=2)), row=1, col=ci)
            fig3.add_trace(go.Scatter(x=epochs, y=hist_data[vk], name="Validation",
                                       line=dict(color=D_RED, width=2, dash="dash")), row=1, col=ci)
            fig3.add_vline(x=best_epoch, line_dash="dot", line_color=D_GOLD,
                           annotation_text=f"Best ep.{best_epoch}",
                           annotation_font_color=D_GOLD, row=1, col=ci)
        fig3.update_layout(template=TEMPLATE, height=380,
                            paper_bgcolor=D_SURFACE, plot_bgcolor='#f8f9ff',
                            font=dict(family="Inter"),
                            title=dict(text="Training History — Regression Model", font=dict(size=15)),
                            showlegend=True, legend=dict(bgcolor="rgba(255,255,255,0.9)"))
        fig3.update_xaxes(title_text="Epoch")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown(f"""
    <div style="background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.2);
                border-left:3px solid {D_RED};border-radius:10px;padding:20px 24px;">
      <div style="font-weight:700;font-size:0.9rem;color:{D_TEXT};margin-bottom:8px;">📌 Direction Classifier — EMH Finding</div>
      <div style="font-size:0.86rem;color:{D_MUTED2};line-height:1.8;">
        Accuracy <b style="color:{D_TEXT}">52.7%</b> · LSTM sigmoid outputs in [0.477, 0.484] · std = 0.002 — outputting dataset prior only.<br>
        Consistent with <b style="color:{D_TEXT}">Weak Form EMH</b> (Fama, 1970): HDFC Bank is a large-cap stock where all historical information is already priced in.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "sentiment":
    st.markdown('<div class="d-page">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="d-page-header">
      <div class="d-page-title">📰 News Sentiment Analysis</div>
      <div class="d-page-sub">5,422 HDFC Bank headlines · VADER scored · overlaid on live NSE price data.</div>
    </div>
    """, unsafe_allow_html=True)

    pos_n = (news_df["sentiment"]=="Positive").sum()
    neu_n = (news_df["sentiment"]=="Neutral").sum()
    neg_n = (news_df["sentiment"]=="Negative").sum()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Headlines", f"{len(news_df):,}")
    c2.metric("Positive 🟢",     f"{pos_n:,} ({pos_n/len(news_df)*100:.0f}%)")
    c3.metric("Neutral 🟡",      f"{neu_n:,} ({neu_n/len(news_df)*100:.0f}%)")
    c4.metric("Negative 🔴",     f"{neg_n:,} ({neg_n/len(news_df)*100:.0f}%)")
    st.markdown("<br>", unsafe_allow_html=True)

    _, lh2, _, _, ok2 = fetch_full()
    if ok2 and not lh2.empty:
        price_m = lh2["Close"].resample("ME").last().reset_index()
        price_m.columns = ["Date","Close"]
    else:
        price_m = df.set_index("Date").resample("ME")["Close"].last().reset_index()

    news_m = news_df.set_index("date").resample("ME")["compound"].mean().reset_index()
    news_m.columns = ["Date","sentiment"]

    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Scatter(
        x=price_m["Date"], y=price_m["Close"], name="HDFCBANK",
        line=dict(color=D_BLUE, width=2.5), fill="tozeroy", fillcolor="rgba(59,130,246,0.06)",
        hovertemplate="%{x|%b %Y}<br>₹%{y:,.2f}<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=news_m["Date"], y=news_m["sentiment"], name="Sentiment",
        marker=dict(color=news_m["sentiment"].apply(
            lambda x: D_GREEN if x>0.05 else (D_RED if x<-0.05 else D_GOLD)), opacity=0.7),
        hovertemplate="%{x|%b %Y}<br>Score: %{y:.3f}<extra></extra>",
    ), secondary_y=True)
    fig.add_hline(y=0, line_dash="dot", line_color=D_MUTED, opacity=0.5, secondary_y=True)
    fig.update_layout(
        title=dict(text="Price vs News Sentiment — Monthly Average", font=dict(size=15, family="Inter")),
        template=TEMPLATE, height=500, paper_bgcolor=D_SURFACE, plot_bgcolor='#f8f9ff',
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=D_BORDER),
        font=dict(family="Inter"), xaxis=dict(title="Month", tickangle=-30), hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price (₹)", secondary_y=False, tickprefix="₹", tickformat=",.0f")
    fig.update_yaxes(title_text="VADER Score", secondary_y=True, range=[-0.4,0.4])
    st.plotly_chart(fig, use_container_width=True)

    news_df["year"] = pd.to_datetime(news_df["date"]).dt.year
    yearly = news_df.groupby(["year","sentiment"]).size().reset_index(name="count")
    fig2 = px.bar(yearly, x="year", y="count", color="sentiment",
                   color_discrete_map={"Positive":D_GREEN,"Neutral":D_GOLD,"Negative":D_RED},
                   title="Headlines by Sentiment & Year", barmode="stack",
                   template=TEMPLATE, labels={"count":"Headlines","year":"Year"}, text_auto=True)
    fig2.update_layout(height=360, paper_bgcolor=D_SURFACE, plot_bgcolor='#f8f9ff',
                        font=dict(family="Inter"), legend_title="Sentiment")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f'<div style="font-size:0.78rem;color:{D_MUTED};font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">Recent Headlines</div>', unsafe_allow_html=True)
    rn = news_df.sort_values("date",ascending=False).head(20)[["date","headline","sentiment","compound"]].copy()
    rn["date"] = pd.to_datetime(rn["date"]).dt.strftime("%d %b %Y")
    rn.columns = ["Date","Headline","Sentiment","Score"]
    st.dataframe(rn, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# VALUATION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "valuation":
    st.markdown('<div class="d-page">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="d-page-header">
      <div class="d-page-title">💰 Valuation — DCF + CAPM</div>
      <div class="d-page-sub">Live fundamental analysis from Yahoo Finance · refreshes every 5 minutes.</div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Fetching live financials..."):
        info, lh, fin, bs, ok = fetch_full()

    if not ok:
        st.error("Could not fetch live data.")
    else:
        st.markdown("#### Key Statistics")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Market Cap",  f"₹{info.get('marketCap',0)/1e12:.2f}T")
        c2.metric("P/E (TTM)",   f"{info.get('trailingPE','N/A'):.1f}" if info.get('trailingPE') else "N/A")
        c3.metric("EPS (TTM)",   f"₹{info.get('trailingEps',0):.2f}")
        c4.metric("Book Value",  f"₹{info.get('bookValue',0):.2f}")
        c5,c6,c7,c8 = st.columns(4)
        c5.metric("P/B Ratio",   f"{info.get('priceToBook','N/A'):.2f}" if info.get('priceToBook') else "N/A")
        c6.metric("Div Yield",   f"{info.get('dividendYield',0)*100:.2f}%" if info.get('dividendYield') else "N/A")
        c7.metric("Beta (β)",    f"{info.get('beta',0):.2f}" if info.get('beta') else "N/A")
        c8.metric("52W Range",   f"₹{info.get('fiftyTwoWeekLow',0):,.0f}–₹{info.get('fiftyTwoWeekHigh',0):,.0f}")
        st.markdown("<br>", unsafe_allow_html=True)

        beta = info.get("beta",0.85) or 0.85
        rf, rm = 0.068, 0.13
        capm = rf + beta*(rm-rf)
        curr = info.get("currentPrice", display_price)
        eps  = info.get("trailingEps",0) or 0
        g, g2 = 0.10, 0.05

        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown(f"""
            <div class="d-card">
              <div style="font-weight:700;font-size:0.95rem;color:{D_BLUE2};margin-bottom:18px;">📐 CAPM — Cost of Equity</div>
              <table style="width:100%;font-size:0.88rem;border-collapse:collapse;">
                <tr style="border-bottom:1px solid {D_BORDER};"><td style="color:{D_MUTED};padding:9px 0;">Risk-Free Rate (10Y Gsec)</td><td style="text-align:right;font-weight:600;color:{D_TEXT};">{rf*100:.1f}%</td></tr>
                <tr style="border-bottom:1px solid {D_BORDER};"><td style="color:{D_MUTED};padding:9px 0;">Nifty 50 Long-run Return</td><td style="text-align:right;font-weight:600;color:{D_TEXT};">{rm*100:.1f}%</td></tr>
                <tr style="border-bottom:1px solid {D_BORDER};"><td style="color:{D_MUTED};padding:9px 0;">Beta (β) — live</td><td style="text-align:right;font-weight:600;color:{D_TEXT};">{beta:.2f}</td></tr>
                <tr style="border-bottom:1px solid {D_BORDER};"><td style="color:{D_MUTED};padding:9px 0;">Equity Risk Premium</td><td style="text-align:right;font-weight:600;color:{D_TEXT};">{(rm-rf)*100:.1f}%</td></tr>
                <tr><td style="font-weight:700;padding:10px 0;color:{D_TEXT};">CAPM Expected Return</td>
                    <td style="text-align:right;font-weight:900;color:{D_BLUE2};font-size:1.3rem;font-family:'JetBrains Mono',monospace;">{capm*100:.2f}%</td></tr>
              </table>
              <div style="margin-top:12px;padding-top:10px;border-top:1px solid {D_BORDER};font-size:0.75rem;color:{D_MUTED};">E(R) = Rf + β × (Rm − Rf)</div>
            </div>
            """, unsafe_allow_html=True)

        with col_d:
            if eps > 0:
                dcf = eps*(1+g)/(capm-g2) if capm>g2 else 0
                mos = ((dcf-curr)/curr)*100 if dcf>0 else 0
                v   = "Undervalued 🟢" if mos>10 else ("Overvalued 🔴" if mos<-10 else "Fair Value 🟡")
                vc  = D_GREEN if mos>10 else (D_RED if mos<-10 else D_GOLD)
            else:
                dcf,mos,v,vc = 0,0,"N/A",D_MUTED

            st.markdown(f"""
            <div class="d-card">
              <div style="font-weight:700;font-size:0.95rem;color:{D_RED};margin-bottom:18px;">💹 DCF — Gordon Growth Model</div>
              <table style="width:100%;font-size:0.88rem;border-collapse:collapse;">
                <tr style="border-bottom:1px solid {D_BORDER};"><td style="color:{D_MUTED};padding:9px 0;">EPS (TTM)</td><td style="text-align:right;font-weight:600;color:{D_TEXT};">₹{eps:.2f}</td></tr>
                <tr style="border-bottom:1px solid {D_BORDER};"><td style="color:{D_MUTED};padding:9px 0;">Near-term Growth</td><td style="text-align:right;font-weight:600;color:{D_TEXT};">{g*100:.0f}%</td></tr>
                <tr style="border-bottom:1px solid {D_BORDER};"><td style="color:{D_MUTED};padding:9px 0;">Terminal Growth</td><td style="text-align:right;font-weight:600;color:{D_TEXT};">{g2*100:.0f}%</td></tr>
                <tr style="border-bottom:1px solid {D_BORDER};"><td style="color:{D_MUTED};padding:9px 0;">Discount Rate (CAPM)</td><td style="text-align:right;font-weight:600;color:{D_TEXT};">{capm*100:.2f}%</td></tr>
                <tr style="border-bottom:1px solid {D_BORDER};"><td style="color:{D_MUTED};padding:9px 0;">DCF Intrinsic Value</td><td style="text-align:right;font-weight:600;color:{D_TEXT};">₹{dcf:,.0f}</td></tr>
                <tr style="border-bottom:1px solid {D_BORDER};"><td style="color:{D_MUTED};padding:9px 0;">Current Price</td><td style="text-align:right;font-weight:600;color:{D_TEXT};">₹{curr:,.0f}</td></tr>
                <tr><td style="font-weight:700;padding:10px 0;color:{D_TEXT};">Verdict</td>
                    <td style="text-align:right;font-weight:900;color:{vc};font-size:1.1rem;">{v}<br><span style="font-size:0.8rem;font-family:'JetBrains Mono',monospace;">({mos:+.1f}% margin)</span></td></tr>
              </table>
              <div style="margin-top:12px;padding-top:10px;border-top:1px solid {D_BORDER};font-size:0.75rem;color:{D_MUTED};">V = EPS×(1+g) / (r − g∞)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 5-Year Price History")
        if not lh.empty:
            fh = go.Figure(go.Scatter(
                x=lh.index, y=lh["Close"], name="HDFCBANK",
                line=dict(color=D_BLUE, width=2), fill="tozeroy",
                fillcolor="rgba(59,130,246,0.06)",
                hovertemplate="%{x|%d %b %Y}<br>₹%{y:,.2f}<extra></extra>"
            ))
            fh.update_layout(template=TEMPLATE, height=360,
                              paper_bgcolor=D_SURFACE, plot_bgcolor='#f8f9ff',
                              font=dict(family="Inter"),
                              xaxis_title="Date",
                              yaxis=dict(title="Price (₹)", tickprefix="₹", tickformat=",.0f"))
            st.plotly_chart(fh, use_container_width=True)

        col_fs, col_bs = st.columns(2)
        with col_fs:
            st.markdown("#### Income Statement")
            if not fin.empty:
                fd = (fin.iloc[:7]/1e7).round(2)
                fd.columns = [str(c)[:10] for c in fd.columns]
                st.dataframe(fd, use_container_width=True)
                st.caption("₹ Crore")
            else:
                st.info("Unavailable.")
        with col_bs:
            st.markdown("#### Balance Sheet")
            if not bs.empty:
                bd = (bs.iloc[:10]/1e7).round(2)
                bd.columns = [str(c)[:10] for c in bd.columns]
                st.dataframe(bd, use_container_width=True)
                st.caption("₹ Crore")
            else:
                st.info("Unavailable.")
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "architecture":
    st.markdown('<div class="d-page">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="d-page-header">
      <div class="d-page-title">🧠 Model Architecture</div>
      <div class="d-page-sub">Two dedicated LSTM models — one for price regression, one for direction classification.</div>
    </div>
    """, unsafe_allow_html=True)

    def arch_layer(icon, icon_bg, name, detail, border_color):
        return f"""
        <div class="arch-layer">
          <div class="arch-layer-icon" style="background:{icon_bg};color:white;">{icon}</div>
          <div>
            <div class="arch-layer-name">{name}</div>
            {"<div class='arch-layer-sub'>" + detail + "</div>" if detail else ""}
          </div>
          <div style="margin-left:auto;width:8px;height:8px;border-radius:50%;background:{border_color};opacity:0.6;"></div>
        </div>
        """

    def arch_arrow_html():
        return f'<div class="arch-arrow" style="border-left-color:{D_BORDER};">↓</div>'

    def build_model_card(title, subtitle, color, layers, result_label, result_val, result_color):
        layers_html = ""
        for i, (icon, icon_bg, name, detail) in enumerate(layers):
            layers_html += arch_layer(icon, icon_bg, name, detail, color)
            if i < len(layers)-1:
                layers_html += arch_arrow_html()
        return f"""
        <div class="arch-card">
          <div class="arch-header" style="background:linear-gradient(135deg,{color}15,{color}08);border-bottom:1px solid {D_BORDER};">
            <div style="width:10px;height:10px;border-radius:50%;background:{color};"></div>
            <div>
              <div style="color:{color};">{title}</div>
              <div style="font-size:0.72rem;color:{D_MUTED};font-weight:400;margin-top:1px;">{subtitle}</div>
            </div>
          </div>
          {layers_html}
          <div style="padding:14px 20px;background:linear-gradient(135deg,{result_color}12,{result_color}06);
                      border-top:1px solid {D_BORDER};display:flex;justify-content:space-between;align-items:center;">
            <span style="font-size:0.78rem;color:{D_MUTED};font-weight:600;text-transform:uppercase;letter-spacing:1px;">{result_label}</span>
            <span style="font-size:1.1rem;font-weight:800;color:{result_color};font-family:'JetBrains Mono',monospace;">{result_val}</span>
          </div>
        </div>
        """

    DB = D_SURFACE2

    reg_layers = [
        ("→",  D_BLUE,  "Input",              "30 days × 26 features"),
        ("⟳",  D_BLUE,  "LSTM Layer 1",       "128 units · return_sequences=True"),
        ("◌",  D_MUTED, "Dropout 0.2",         ""),
        ("⟳",  D_BLUE,  "LSTM Layer 2",       "64 units · return_sequences=True"),
        ("◌",  D_MUTED, "Dropout 0.2",         ""),
        ("⟳",  D_BLUE,  "LSTM Layer 3",       "32 units · return_sequences=False"),
        ("◌",  D_MUTED, "Dropout 0.2",         ""),
        ("≡",  D_GOLD,  "BatchNormalization",  "Stabilise activations"),
        ("▣",  D_TEAL,  "Dense (64, ReLU)",    "Feature combination"),
        ("◌",  D_MUTED, "Dropout 0.1",         ""),
        ("▣",  D_TEAL,  "Dense (32, ReLU)",    "Feature refinement"),
        ("★",  D_GREEN, "Output Dense (1)",    "→ next-day return (scaled)"),
    ]
    cls_layers = [
        ("→",  D_RED,   "Input",              "30 days × 26 features"),
        ("⟳",  D_RED,   "LSTM Layer 1",       "128 units · return_sequences=True"),
        ("◌",  D_MUTED, "Dropout 0.4",         "Heavy regularisation"),
        ("⟳",  D_RED,   "LSTM Layer 2",       "64 units · return_sequences=True"),
        ("◌",  D_MUTED, "Dropout 0.3",         ""),
        ("⟳",  D_RED,   "LSTM Layer 3",       "32 units · return_sequences=False"),
        ("◌",  D_MUTED, "Dropout 0.3",         ""),
        ("≡",  D_GOLD,  "BatchNormalization",  "Stabilise activations"),
        ("▣",  D_TEAL,  "Dense (64, ReLU)",    "Feature combination"),
        ("◌",  D_MUTED, "Dropout 0.3",         ""),
        ("▣",  D_TEAL,  "Dense (32, ReLU)",    "Feature refinement"),
        ("★",  D_RED,   "Output Dense (1, σ)", "→ Up/Down probability"),
    ]

    col_r, col_c = st.columns(2, gap="large")
    with col_r:
        st.markdown(build_model_card(
            "Model 1 — Price Regression",
            "MSE Loss · Adam lr=0.001 · ~145K parameters",
            D_BLUE, reg_layers,
            "Test R² = 0.9768 · MAPE = 0.76% · MAE = ₹7.00",
            "R² 0.9768", D_GREEN
        ), unsafe_allow_html=True)
    with col_c:
        st.markdown(build_model_card(
            "Model 2 — Direction Classifier",
            "BCE Loss · Adam lr=0.0003 · ~145K parameters",
            D_RED, cls_layers,
            "Accuracy ≈ 52.7% — EMH limitation (large-cap stock)",
            "52.7%", D_GOLD
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:1.1rem;font-weight:800;color:{D_TEXT};margin-bottom:20px;">Feature Engineering Pipeline — 26 Features</div>', unsafe_allow_html=True)

    cats = {
        "Price & Volume":    (["Open","High","Low","Close","Volume"], D_BLUE, "5 features"),
        "Returns & Vol":     (["returns","log_returns","vol_5d","vol_10d","hl_ratio","co_ratio","overnight_gap"], D_TEAL, "7 features"),
        "Lag Features":      (["lag1_return","lag2_return","lag1_direction"], D_GOLD, "3 features"),
        "Technical Ind.":    (["RSI-14","MACD","MACD Signal","MACD Hist","BB Upper","BB Lower","BB Width"], D_RED, "7 features"),
        "Momentum":          (["momentum_5d","momentum_10d"], D_BLUE2, "2 features"),
        "Sentiment":         (["daily_compound","sentiment_ma3","sentiment_ma7","sentiment_ma14"], D_GREEN, "4 features"),
    }
    cols3 = st.columns(3)
    for i,(cat,(feats,col,cnt)) in enumerate(cats.items()):
        with cols3[i%3]:
            st.markdown(f"""
            <div class="d-card" style="border-top:2px solid {col};margin-bottom:14px;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                <div style="font-weight:700;font-size:0.85rem;color:{col};">{cat}</div>
                <div style="font-size:0.7rem;color:{D_MUTED};background:{D_SURFACE2};padding:2px 8px;border-radius:99px;">{cnt}</div>
              </div>
              <div style="font-size:0.78rem;color:{D_MUTED2};line-height:2;">{"<br>".join(feats)}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{D_SURFACE};border:1px solid {D_BORDER};border-left:3px solid {D_BLUE};
                border-radius:10px;padding:22px 28px;">
      <div style="font-weight:700;font-size:0.95rem;margin-bottom:14px;color:{D_TEXT};">Model Summary</div>
      <div style="display:flex;gap:40px;flex-wrap:wrap;font-size:0.88rem;">
        <div><div style="color:{D_MUTED};font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;">Parameters</div><div style="font-size:1.2rem;font-weight:800;color:{D_BLUE};font-family:'JetBrains Mono',monospace;">~145,537</div></div>
        <div><div style="color:{D_MUTED};font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;">Sequence</div><div style="font-size:1.2rem;font-weight:800;color:{D_BLUE};font-family:'JetBrains Mono',monospace;">30 days</div></div>
        <div><div style="color:{D_MUTED};font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;">Features</div><div style="font-size:1.2rem;font-weight:800;color:{D_BLUE};font-family:'JetBrains Mono',monospace;">26</div></div>
        <div><div style="color:{D_MUTED};font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;">Train</div><div style="font-size:1.2rem;font-weight:800;color:{D_TEAL};font-family:'JetBrains Mono',monospace;">1,188</div></div>
        <div><div style="color:{D_MUTED};font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;">Test</div><div style="font-size:1.2rem;font-weight:800;color:{D_GOLD};font-family:'JetBrains Mono',monospace;">298</div></div>
        <div><div style="color:{D_MUTED};font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;">Best R²</div><div style="font-size:1.2rem;font-weight:800;color:{D_GREEN};font-family:'JetBrains Mono',monospace;">0.9768</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)