# analyzer.py — sentiment-only (no forecast/backtest)
import os
import traceback
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import praw
import yfinance as yf
from dotenv import load_dotenv

# optional sklearn metrics (not required for sentiment-only)
try:
    from sklearn.metrics import accuracy_score  # noqa: F401
except Exception:
    accuracy_score = None  # type: ignore

# helpers expected in your utils.py; fallback will be used if missing
from utils import ASPECT_KEYWORDS, normalize_sentiment  # type: ignore
# extract_label should map various model outputs to 'positive'|'neutral'|'negative'
try:
    from utils import extract_label  # type: ignore
except Exception:
    # simple fallback
    def extract_label(x):
        if isinstance(x, str):
            v = x.strip().lower()
            if "pos" in v:
                return "positive"
            if "neg" in v:
                return "negative"
            if "neu" in v:
                return "neutral"
        if isinstance(x, dict) and "label" in x:
            return extract_label(x["label"])
        return "neutral"


load_dotenv()


def _extract_aspect_sentences(txt: str, max_sentences: int = 3) -> str:
    """
    Keep only the first few sentences that include aspect keywords.
    Fallback to the first ~240 chars if nothing matches.
    """
    txt = (txt or "").strip()
    if not txt:
        return ""
    parts = [p.strip() for p in txt.replace("\n", " ").split(".") if p.strip()]
    keys = {k for words in ASPECT_KEYWORDS.values() for k in words}
    picks = [s for s in parts if any(k in s.lower() for k in keys)]
    if picks:
        return ". ".join(picks[:max_sentences])
    return txt[:240]


def _daily_sentiment_from_posts(posts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate posts into daily average sentiment (using 'raw_score' if present, else 'score').
    Returns DataFrame with columns ['date', 'daily_sentiment'] (date is tz-naive datetime).
    """
    if posts_df is None or posts_df.empty:
        return pd.DataFrame(columns=["date", "daily_sentiment"])

    if posts_df["created_utc"].dtype == object:
        posts_df["created_utc"] = pd.to_datetime(posts_df["created_utc"], utc=True)

    posts_df["date"] = posts_df["created_utc"].dt.date
    score_col = "raw_score" if "raw_score" in posts_df.columns else "score"
    daily = posts_df.groupby("date")[score_col].mean().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.rename(columns={score_col: "daily_sentiment"})
    return daily[["date", "daily_sentiment"]]


class StockSentimentAnalyzer:
    """
    Minimal sentiment analyzer:
      - get_reddit_posts(symbol, limit, use_titles_only) -> DataFrame
      - predict_trend(symbol, limit, use_titles_only, debug) -> (label, posts_df, avg_score)
    """

    def __init__(self):
        # lazy reddit init — won't crash import if env vars missing
        try:
            self.reddit = praw.Reddit(
                client_id=os.getenv("REDDIT_CLIENT_ID"),
                client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
                user_agent=os.getenv("REDDIT_USER_AGENT"),
            )
        except Exception:
            self.reddit = None

        # try to import a robust aspect_sentiment function from utils (FinBERT wrapper)
        try:
            from utils import aspect_sentiment_finbert  # type: ignore
            self.aspect_sentiment_finbert = aspect_sentiment_finbert
        except Exception:
            # fallback naive aspect sentiment (rules) — replace with FinBERT for production
            def _fallback_aspect_sentiment(text: str):
                t = (text or "").lower()
                if any(w in t for w in ("beat", "beats", "good", "strong", "outperform", "upgrade", "profit", "growth", "up")):
                    return {"growth_sentiment": "positive", "employment_sentiment": "neutral", "inflation_sentiment": "neutral"}
                if any(w in t for w in ("miss", "missed", "weak", "down", "loss", "cut", "layoff", "layoffs", "decline")):
                    return {"growth_sentiment": "negative", "employment_sentiment": "neutral", "inflation_sentiment": "neutral"}
                return {"growth_sentiment": "neutral", "employment_sentiment": "neutral", "inflation_sentiment": "neutral"}

            self.aspect_sentiment_finbert = _fallback_aspect_sentiment

    # -------------------------
    # Data fetchers
    # -------------------------
    def _fetch_pushshift(self, stock_symbol: str, days_back: int = 720, limit: int = 1000, subreddits: str = "stocks+investing+wallstreetbets"):
        """
        Fetch submissions from Pushshift for the given query and timeframe.
        Returns a DataFrame similar to get_reddit_posts output.
        """
        # Pushshift endpoint
        base = "https://api.pushshift.io/reddit/search/submission/"
        end_time = int(datetime.now(tz=timezone.utc).timestamp())
        start_time = int((datetime.now(tz=timezone.utc) - timedelta(days=days_back)).timestamp())

        # We'll page backwards using 'before' param; page size up to 500 (Pushshift limit)
        page_size = 500 if limit > 500 else limit
        results = []
        before = end_time
        got = 0

        while got < limit:
            params = {
                "q": f"{stock_symbol} stock",
                "size": page_size,
                "before": before,
                "after": start_time,
                "subreddit": subreddits,  # multiple subreddits comma-separated works
                "sort": "desc",
                "sort_type": "created_utc",
                # you can add "is_self": True to prefer self posts
            }
            try:
                r = requests.get(base, params=params, timeout=20)
                r.raise_for_status()
                data = r.json().get("data", [])
            except Exception as e:
                # Pushshift can be flaky — stop and return what we have (caller may fallback to PRAW)
                # print("Pushshift fetch failed:", e)
                break

            if not data:
                break

            for item in data:
                # map Pushshift fields to your posts shape
                created_utc = item.get("created_utc")
                if created_utc is None:
                    continue
                created_dt = datetime.fromtimestamp(created_utc, tz=timezone.utc)
                title = item.get("title") or ""
                selftext = item.get("selftext") or ""
                base_text = title if False else (title + ". " + selftext)
                focused = title + ". " + base_text[:1000]  # keep reasonable length
                results.append(
                    {
                        "title": title,
                        "text": focused,
                        "score": item.get("score", 0),
                        "upvote_ratio": item.get("upvote_ratio", 1.0),
                        "num_comments": item.get("num_comments", 0),
                        "created_utc": created_dt,
                    }
                )
                got += 1
                if got >= limit:
                    break

            # page earlier than the oldest returned in this batch
            oldest = data[-1].get("created_utc", None)
            if oldest is None or oldest >= before:
                break
            before = oldest - 1
            # be gentle on Pushshift
            time.sleep(0.5)

        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results)

    def get_reddit_posts(self, stock_symbol, limit=500, use_titles_only=False, days_back=720, prefer_pushshift=True):
        """
        Robust post fetcher.
        - prefer_pushshift: try Pushshift (historical archive) first; falls back to PRAW if Pushshift fails.
        - days_back: how many days back to fetch (use big number for 6+ months)
        - returns pandas.DataFrame with columns: title, text, score, upvote_ratio, num_comments, created_utc
        """
        # try Pushshift first (recommended)
        if prefer_pushshift:
            try:
                df = self._fetch_pushshift(stock_symbol, days_back=days_back, limit=limit)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # optionally shorten text to titles only
                    if use_titles_only:
                        df["text"] = df["title"]
                    return df
                # else fall through to PRAW
            except Exception:
                # fallback to PRAW below
                pass

        # fallback: PRAW search (may return less historical coverage)
        posts = []
        try:
            sub = self.reddit.subreddit("stocks+investing+wallstreetbets")
            # PRAW doesn't accept custom after/before easily via search, so we fetch 'limit' newest posts and filter
            for post in sub.search(f"{stock_symbol} stock", sort="new", time_filter="year", limit=limit):
                raw_body = post.selftext or ""
                base_text = post.title if use_titles_only else (post.title + ". " + raw_body)
                focused = post.title + ". " + base_text
                posts.append(
                    {
                        "title": post.title,
                        "text": focused,
                        "score": post.score,
                        "upvote_ratio": getattr(post, "upvote_ratio", 1.0),
                        "num_comments": getattr(post, "num_comments", 0),
                        "created_utc": datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
                    }
                )
        except Exception:
            # if PRAW also fails, return empty DF
            return pd.DataFrame(posts)

        return pd.DataFrame(posts)

    def get_stock_data(self, stock_symbol: str, days: int = 365) -> pd.DataFrame:
        """
        Fetch daily OHLCV using yfinance for the last `days` days.
        Returns a DataFrame (index is Date) as returned by yfinance.history().
        """
        end = datetime.now()
        start = end - timedelta(days=days)
        stock = yf.Ticker(stock_symbol)
        return stock.history(start=start, end=end)

    # -------------------------
    # Sentiment scoring (single-shot)
    # -------------------------
    def predict_trend(self, stock_symbol: str, limit: int = 200, use_titles_only: bool = True, debug: bool = False) -> Tuple[str, Optional[pd.DataFrame], Optional[float]]:
        """
        Analyze last-month posts for `stock_symbol` and return:
            (label, posts_df, avg_weighted_score)

        label: "Bullish (…)" | "Bearish (…)" | "Neutral (…)" | error message
        posts_df: DataFrame with aspect_sentiment, raw_score, score columns added
        avg_weighted_score: float or None

        This function is intentionally simple: it does one FinBERT (or fallback) pass
        per post, computes a weighted label-only score, then aggregates.
        """
        posts_df = self.get_reddit_posts(stock_symbol, limit=limit, use_titles_only=use_titles_only)
        if posts_df is None or posts_df.empty:
            return "No Reddit posts found.", pd.DataFrame(), None

        # aspect labeling (FinBERT wrapper if available)
        posts_df["aspect_sentiment"] = posts_df["text"].apply(lambda t: self.aspect_sentiment_finbert(t))

        # normalize created_utc to tz-aware if possible
        try:
            posts_df["created_utc"] = pd.to_datetime(posts_df["created_utc"], utc=True)
        except Exception:
            pass

        # recency + karma weighting (same shape as earlier pipeline)
        now_utc = pd.Timestamp.now(tz="UTC")
        if "created_utc" in posts_df.columns:
            age_days = (now_utc - posts_df["created_utc"]).dt.total_seconds() / 86400.0
            recency_w = np.exp(-age_days / 7.0)  # ~1-week half-life
        else:
            recency_w = np.ones(len(posts_df))

        karma_w = np.log1p(posts_df["score"].clip(lower=0)) / 5.0 if "score" in posts_df.columns else np.zeros(len(posts_df))
        disc_w = (recency_w * (0.5 + karma_w)).clip(0.1, 3.0)

        # convert aspect labels to signed numeric, then aggregate with weights
        weights = {"growth_sentiment": 0.5, "employment_sentiment": 0.3, "inflation_sentiment": 0.2}

        def _score_row_labelonly(s):
            if not isinstance(s, dict):
                return 0.0
            val = 0.0
            for k, wt in weights.items():
                if k in s:
                    lab = extract_label(s[k])
                    if lab == "positive":
                        val += wt
                    elif lab == "negative":
                        val -= wt
            return float(val)

        posts_df["raw_score"] = posts_df["aspect_sentiment"].apply(_score_row_labelonly)
        try:
            posts_df["score"] = posts_df["raw_score"] * disc_w
        except Exception:
            # if shapes mismatch, fallback to raw_score
            posts_df["score"] = posts_df["raw_score"]

        avg_score = posts_df["score"].replace([np.inf, -np.inf], np.nan).dropna().mean()
        if debug:
            print("[DEBUG] sample aspect_sentiments:", posts_df["aspect_sentiment"].head(5).tolist())
            print("[DEBUG] sample raw/weighted scores:", posts_df[["raw_score", "score"]].head(5).values.tolist())
            print("[DEBUG] avg_score:", avg_score)

        if pd.isna(avg_score):
            return "Unable to calculate sentiment.", posts_df, None
        if avg_score > 0.20:
            return f"Bullish (Positive sentiment detected; avg_score={avg_score:.3f})", posts_df, float(avg_score)
        if avg_score < -0.20:
            return f"Bearish (Negative sentiment detected; avg_score={avg_score:.3f})", posts_df, float(avg_score)
        return f"Neutral (Mixed sentiment; avg_score={avg_score:.3f})", posts_df, float(avg_score)
