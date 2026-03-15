"""
scraper_news.py
===============
Scrapes HDFC Bank news headlines from Google News
for 2020-2026 — recent news has the strongest price signal.

Queries Google News in monthly chunks to maximise headline count.
Scores each headline with VADER sentiment (compound score -1 to +1).

Output: hdfcbank_news.csv
Columns: date, headline, source, compound, sentiment
"""

import time
import pandas as pd
from datetime import date, timedelta
from dateutil import parser as dp
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from pygooglenews import GoogleNews
except ImportError:
    print("Install pygooglenews: pip install pygooglenews")
    exit()

OUTPUT_CSV = "hdfcbank_news.csv"

QUERIES = [
    "HDFC Bank stock",
    "HDFC Bank NSE share price",
    "HDFCBANK stock market",
    "HDFC Bank results earnings",
]

def generate_months(start_year=2020, end_year=2026):
    months = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            m_start = date(year, month, 1)
            # last day of month
            if month == 12:
                m_end = date(year, 12, 31)
            else:
                m_end = date(year, month + 1, 1) - timedelta(days=1)
            if m_start <= date(2026, 3, 14):
                months.append((m_start, m_end))
    return months

def score_sentiment(text, analyzer):
    compound = round(analyzer.polarity_scores(text)["compound"], 4)
    if compound >= 0.05:   return compound, "Positive"
    elif compound <= -0.05: return compound, "Negative"
    else:                   return compound, "Neutral"

def main():
    print("=" * 60)
    print("  News Scraper — HDFC Bank (2020-2026)")
    print("  Method: Google News monthly queries")
    print("=" * 60)

    gn        = GoogleNews(lang="en", country="IN")
    analyzer  = SentimentIntensityAnalyzer()
    months    = generate_months(2020, 2026)
    all_items = []
    seen      = set()

    for m_start, m_end in months:
        from_str = m_start.strftime("%Y-%m-%d")
        to_str   = m_end.strftime("%Y-%m-%d")
        print(f"\n  Month: {from_str} → {to_str}")
        month_count = 0

        for query in QUERIES:
            try:
                results = gn.search(query, from_=from_str, to_=to_str)
                entries = results.get("entries", [])

                for entry in entries:
                    title = entry.get("title", "").strip()
                    if not title or len(title) < 10:
                        continue

                    try:
                        art_date = dp.parse(entry.get("published", "")).date()
                    except Exception:
                        art_date = m_start

                    key = title[:80]
                    if key in seen:
                        continue
                    seen.add(key)

                    compound, sentiment = score_sentiment(title, analyzer)
                    all_items.append({
                        "date":      art_date,
                        "headline":  title,
                        "source":    entry.get("source", {}).get("title", "Google News"),
                        "compound":  compound,
                        "sentiment": sentiment,
                    })
                    month_count += 1

                time.sleep(1.2)

            except Exception as e:
                print(f"    [!] '{query}': {e}")
                time.sleep(3)

        print(f"    Total new this month: {month_count}")

    if not all_items:
        print("\n  [!] No headlines scraped.")
        return

    df = pd.DataFrame(all_items)
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'=' * 60}")
    print(f"  Done! {len(df)} headlines saved to {OUTPUT_CSV}")
    print(f"{'=' * 60}")
    print(f"\n  Date range : {df['date'].min()} → {df['date'].max()}")
    print(f"\n  Sentiment breakdown:")
    for sent, cnt in df["sentiment"].value_counts().items():
        bar = "█" * int(cnt / len(df) * 30)
        print(f"    {sent:12s} {bar} {cnt} ({cnt/len(df)*100:.1f}%)")
    print(f"\n  By year:")
    df["year"] = pd.to_datetime(df["date"]).dt.year
    for year, cnt in df.groupby("year").size().items():
        print(f"    {year}: {cnt} headlines")

if __name__ == "__main__":
    main()