# utils.py
import os
import warnings
from typing import Dict, Optional
import numpy as np
import pandas as pd

# -----------------------
# Aspect keywords (same as earlier)
# -----------------------
ASPECT_KEYWORDS = {
    "growth_sentiment": [
        "revenue", "sales", "demand", "growth", "guidance", "earnings", "profit", "profits",
        "margin", "margins", "split", "acquisition", "buyback", "valuation", "market share",
        "outlook", "forecast",
    ],
    "employment_sentiment": [
        "jobs", "employment", "hiring", "headcount", "layoff", "layoffs", "wage", "wages",
        "salary", "salaries", "labor", "union", "strike",
    ],
    "inflation_sentiment": [
        "inflation", "cpi", "ppi", "prices", "price", "cost", "costs", "input costs",
        "raw materials", "commodity", "oil", "energy",
    ],
}

# -----------------------
# Normalizers / label helpers
# -----------------------
def normalize_sentiment(value) -> str:
    """Map various raw values to: 'positive' | 'neutral' | 'negative'."""
    if value is None:
        return "neutral"
    if isinstance(value, (int, float)):
        return "positive" if value > 0 else "negative" if value < 0 else "neutral"
    if isinstance(value, dict) and "polarity" in value:
        v = value["polarity"]
        return "positive" if v > 0 else "negative" if v < 0 else "neutral"
    if isinstance(value, str):
        v = value.strip().lower()
        if "pos" in v:
            return "positive"
        if "neg" in v:
            return "negative"
        if "neu" in v:
            return "neutral"
    return "neutral"


def extract_label(raw) -> str:
    """
    Convert variety of model outputs to canonical 'positive'|'neutral'|'negative'.
    Accepts dictionaries like {'label':'POSITIVE','score':0.9} or plain strings.
    """
    if raw is None:
        return "neutral"
    if isinstance(raw, dict) and "label" in raw:
        return normalize_sentiment(raw["label"])
    if isinstance(raw, str):
        v = raw.strip().lower()
        if v in ("positive", "pos", "p", "bullish"):
            return "positive"
        if v in ("negative", "neg", "n", "bearish"):
            return "negative"
        if "neu" in v or "neutral" in v:
            return "neutral"
        # fuzzy
        if "up" in v or "good" in v or "beat" in v:
            return "positive"
        if "down" in v or "bad" in v or "miss" in v:
            return "negative"
    # numeric
    if isinstance(raw, (int, float)):
        return "positive" if raw > 0 else "negative" if raw < 0 else "neutral"
    return "neutral"


# -----------------------
# FinBERT wrapper (optional)
# -----------------------
# This will try to create a transformers pipeline; if not available, fall back to a rule-based labeler.
_FINBERT_PIPE = None
_FINBERT_MODEL_NAME = os.environ.get("FINBERT_MODEL", "ProsusAI/finbert")  # try common model

def _init_finbert():
    global _FINBERT_PIPE
    if _FINBERT_PIPE is not None:
        return _FINBERT_PIPE
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        # try to instantiate pipeline (this may be slow)
        tok = AutoTokenizer.from_pretrained(_FINBERT_MODEL_NAME, trust_remote_code=False)
        mdl = AutoModelForSequenceClassification.from_pretrained(_FINBERT_MODEL_NAME, trust_remote_code=False)
        _FINBERT_PIPE = pipeline("text-classification", model=mdl, tokenizer=tok, top_k=None)
        return _FINBERT_PIPE
    except Exception as e:
        warnings.warn(f"FinBERT pipeline init failed: {e}. Using fallback sentiment. Install transformers + model to enable.")
        _FINBERT_PIPE = None
        return None

def aspect_sentiment_finbert(text: str) -> Dict[str, str]:
    """
    Return a dict of per-aspect labels (strings): {'growth_sentiment':'positive', ...}
    If transformers & FinBERT are available, use them; otherwise use a heuristic fallback.
    """
    t = (text or "").strip()
    if not t:
        return {"growth_sentiment": "neutral", "employment_sentiment": "neutral", "inflation_sentiment": "neutral"}

    pipe = _init_finbert()
    if pipe is not None:
        try:
            # run model; FinBERT models often output a single label (POSITIVE/NEGATIVE/NEUTRAL)
            out = pipe(t, truncation=True)
            # out is list of dicts for each label score; we'll pick top label if available
            if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
                label = out[0].get("label", "neutral")
                # assign same label to growth aspect (simple)
                lab = extract_label(label)
                return {"growth_sentiment": lab, "employment_sentiment": "neutral", "inflation_sentiment": "neutral"}
        except Exception:
            pass  # fall through to heuristic

    # Heuristic fallback (naive)
    low = t.lower()
    if any(w in low for w in ("beat", "beats", "good", "strong", "outperform", "profit", "growth", "up")):
        return {"growth_sentiment": "positive", "employment_sentiment": "neutral", "inflation_sentiment": "neutral"}
    if any(w in low for w in ("miss", "missed", "weak", "down", "loss", "cut", "decline", "layoff", "layoffs")):
        return {"growth_sentiment": "negative", "employment_sentiment": "neutral", "inflation_sentiment": "neutral"}
    return {"growth_sentiment": "neutral", "employment_sentiment": "neutral", "inflation_sentiment": "neutral"}


# -----------------------
# Feature helper for inference
def to_json_serializable(x):
    """Recursively convert numpy/pandas scalars and arrays to Python builtins suitable for JSON."""
    if x is None or isinstance(x, (str, bool, int, float)):
        return x
    if isinstance(x, (datetime, date)):
        return x.isoformat()
    # numpy scalar
    if isinstance(x, _np.generic):
        return x.item()
    # numpy array -> list
    if isinstance(x, _np.ndarray):
        return x.tolist()
    # pandas Timestamp
    if isinstance(x, _pd.Timestamp):
        return x.isoformat()
    # pandas Series -> list
    if isinstance(x, _pd.Series):
        return [to_json_serializable(v) for v in x.tolist()]
    # pandas DataFrame -> list of row-dicts
    if isinstance(x, _pd.DataFrame):
        return [to_json_serializable(r) for r in x.to_dict(orient="records")]
    if isinstance(x, dict):
        return {str(k): to_json_serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [to_json_serializable(v) for v in x]
    # fallback: try to cast common numeric types
    try:
        if hasattr(x, "tolist"):
            return to_json_serializable(x.tolist())
    except Exception:
        pass
    try:
        return str(x)
    except Exception:
        return None
def apply_scaler_safe(X: np.ndarray, scaler_np_path: str):
    """
    Apply a saved scaler from np.savez file with keys 'mean' and 'scale' to X.
    If scaler length != n_features of X, try to adapt by truncation/padding.
    Returns scaled X and a dict with debug info.
    """
    info = {"ok": False, "msg": ""}
    arr = np.load(scaler_np_path)
    mean = arr.get("mean")
    scale = arr.get("scale")
    if mean is None or scale is None:
        info["msg"] = "scaler file missing mean/scale keys"
        return X, info
    mean = np.asarray(mean, dtype=float)
    scale = np.asarray(scale, dtype=float)
    n_feat = X.shape[1]
    if mean.shape[0] == n_feat and scale.shape[0] == n_feat:
        info["ok"] = True
        return (X - mean) / scale, info
    # mismatch — adapt
    info["msg"] = f"scaler_len={mean.shape[0]} != n_feat={n_feat}; adapting by trunc/pad (temporary)"
    if mean.shape[0] > n_feat:
        # truncate mean/scale
        mean = mean[:n_feat]
        scale = scale[:n_feat]
    else:
        # pad mean with 0s, scale with 1s
        pad = n_feat - mean.shape[0]
        mean = np.concatenate([mean, np.zeros(pad)], axis=0)
        scale = np.concatenate([scale, np.ones(pad)], axis=0)
    info["ok"] = True
    info["adapted"] = True
    return (X - mean) / scale, info

def make_feature_row_from_recent(prices_df: Optional[pd.DataFrame], posts_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Build stable single-row DataFrame of features for inference.
    Columns (fixed order):
      recent_sent, post_count_30d, pos_ratio_30d, avg_raw_score_30d,
      logret_1, sma_5, sma_10, vol_20
    """
    # defaults
    recent_sent = 0.0
    post_count_30d = 0
    pos_ratio_30d = 0.0
    avg_raw_score_30d = 0.0
    logret_1 = 0.0
    sma_5 = 0.0
    sma_10 = 0.0
    vol_20 = 0.0

    # defensive copy
    prices = prices_df.copy() if prices_df is not None else pd.DataFrame()
    posts = posts_df.copy() if posts_df is not None else pd.DataFrame()

    # Normalize posts datetimes and compute counts
    if not posts.empty:
        try:
            posts["created_utc"] = pd.to_datetime(posts["created_utc"], utc=True)
        except Exception:
            # best-effort parse if already tz-aware or string
            posts["created_utc"] = pd.to_datetime(posts["created_utc"], errors="coerce").dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")

        # ensure we have 'raw_score' or compute fallback (from aspect_sentiment)
        if "raw_score" not in posts.columns:
            def _raw_from_aspect(r):
                asp = r.get("aspect_sentiment") if isinstance(r, dict) else None
                if isinstance(asp, dict):
                    val = 0.0
                    w = {"growth_sentiment": 0.5, "employment_sentiment": 0.3, "inflation_sentiment": 0.2}
                    for k, wt in w.items():
                        lab = asp.get(k)
                        if isinstance(lab, str):
                            lab = lab.lower()
                            if lab == "positive": val += wt
                            elif lab == "negative": val -= wt
                    return float(val)
                # fallback to score column if present
                if "score" in r and pd.notna(r["score"]):
                    return float(r["score"])
                return 0.0
            # build raw_score column
            posts = posts.assign(raw_score=[_raw_from_aspect(r) if isinstance(r, (dict, pd.Series)) else 0.0 for _, r in posts.iterrows()])

        # compute windowed subsets
        cutoff_30 = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)
        cutoff_7 = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
        posts_30 = posts[posts["created_utc"] >= cutoff_30]
        posts_7 = posts[posts["created_utc"] >= cutoff_7]

        post_count_30d = int(len(posts_30))

        # positive ratio over 30d (use aspect growth label when available)
        if post_count_30d > 0:
            pos = 0
            for _, row in posts_30.iterrows():
                asp = row.get("aspect_sentiment") if isinstance(row.get("aspect_sentiment"), dict) else None
                if asp and isinstance(asp, dict):
                    if asp.get("growth_sentiment") and str(asp.get("growth_sentiment")).lower().startswith("pos"):
                        pos += 1
                else:
                    # fallback to raw_score > 0
                    if row.get("raw_score", 0.0) > 0:
                        pos += 1
            pos_ratio_30d = float(pos) / post_count_30d
            avg_raw_score_30d = float(posts_30["raw_score"].astype(float).mean())

        # daily_sent for recent 7 days (use raw_score)
        if not posts.empty and "raw_score" in posts.columns:
            posts["date"] = posts["created_utc"].dt.date
            daily = posts.groupby("date")["raw_score"].mean().reset_index()
            daily["date"] = pd.to_datetime(daily["date"])
            if not daily.empty:
                # recent 7-day mean (use available days up to 7)
                last7 = daily.sort_values("date").tail(7)
                recent_sent = float(last7["raw_score"].mean())

    # Price features
    if not prices.empty and "Close" in prices.columns:
        p = prices.sort_values("Date").reset_index(drop=True)
        p["Close"] = p["Close"].astype(float)
        if len(p) >= 2:
            last = float(p["Close"].iloc[-1])
            prev = float(p["Close"].iloc[-2])
            if prev != 0:
                logret_1 = np.log(last / prev)
        if len(p) >= 5:
            sma_5 = float(p["Close"].tail(5).mean())
        if len(p) >= 10:
            sma_10 = float(p["Close"].tail(10).mean())
        if len(p) >= 20:
            rets = p["Close"].pct_change().dropna().tail(20)
            vol_20 = float(rets.std())

    # Clip / sanity-check recent_sent: the raw_score is weighted sum of aspects with weights summing to 1,
    # and post-specific recency multipliers may inflate it but usually it stays small; clamp extreme values:
    if np.isnan(recent_sent) or np.isinf(recent_sent):
        recent_sent = 0.0
    # clamp to reasonable range
    recent_sent = float(np.clip(recent_sent, -5.0, 5.0))

    row = {
        "recent_sent": float(recent_sent),
        "post_count_30d": int(post_count_30d),
        "pos_ratio_30d": float(pos_ratio_30d),
        "avg_raw_score_30d": float(avg_raw_score_30d),
        "logret_1": float(logret_1),
        "sma_5": float(sma_5),
        "sma_10": float(sma_10),
        "vol_20": float(vol_20),
    }
    # stable column order
    cols = ["recent_sent", "post_count_30d", "pos_ratio_30d", "avg_raw_score_30d", "logret_1", "sma_5", "sma_10", "vol_20"]
    return pd.DataFrame([row], columns=cols)
