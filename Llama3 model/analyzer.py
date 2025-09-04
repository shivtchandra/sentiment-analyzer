import praw
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from utils import normalize_sentiment, llama_sentiment

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# Stock Sentiment Analyzer
# -----------------------------
class StockSentimentAnalyzer:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT')
        )

    def get_reddit_posts(self, stock_symbol, limit=20):
        search_term = stock_symbol
        if stock_symbol in ["^GSPC", "^IXIC", "^DJI", "^RUT"]:
            # Map indices to human-readable search terms
            mapping = {"^GSPC": "S&P 500", "^IXIC": "NASDAQ", "^DJI": "Dow Jones", "^RUT": "Russell 2000"}
            search_term = mapping.get(stock_symbol, stock_symbol)

        posts = []
        for post in self.reddit.subreddit('stocks+investing+wallstreetbets').search(
            f'{search_term} stock', limit=limit
        ):
            posts.append({
                'title': post.title,
                'text': post.selftext if post.selftext else post.title,
                'score': post.score,
                'created_utc': datetime.fromtimestamp(post.created_utc)
            })
        return pd.DataFrame(posts)

    def predict_trend(self, stock_symbol,debug=False):
        posts_df = self.get_reddit_posts(stock_symbol)
        if posts_df.empty:
            return "No Reddit posts found.", None

        posts_df['llama_sentiment'] = posts_df['text'].apply(llama_sentiment)

        def score_fn(s):
            if not isinstance(s, dict):
                return 0
            score, weights = 0, {"growth_sentiment": 0.5, "employment_sentiment": 0.3, "inflation_sentiment": 0.2}
            for k, w in weights.items():
                if k in s:
                    norm = normalize_sentiment(s[k])
                    if debug:
                        print(f"[DEBUG] {k}: raw={s[k]} -> normalized={norm}, weight={w}")
                    
                    score += w if norm == "positive" else -w if norm == "negative" else 0
            return score

        posts_df["score"] = posts_df["llama_sentiment"].apply(score_fn)
        avg_score = posts_df["score"].mean()
        if debug:
            print("[DEBUG] Raw Sentiments:", list(posts_df['llama_sentiment']))
            print("[DEBUG] Scores:", list(posts_df['score']))
            print("[DEBUG] Average Score:", avg_score)


        if pd.isna(avg_score):
            return "Unable to calculate sentiment.", posts_df
        elif avg_score > 0.2:
            return "Bullish (Positive sentiment detected)", posts_df
        elif avg_score < -0.2:
            return "Bearish (Negative sentiment detected)", posts_df
        else:
            return "Neutral (Mixed sentiment)", posts_df

    def get_stock_data(self, stock_symbol, days=30):
        end = datetime.now()
        start = end - timedelta(days=days)
        stock = yf.Ticker(stock_symbol)
        return stock.history(start=start, end=end)
