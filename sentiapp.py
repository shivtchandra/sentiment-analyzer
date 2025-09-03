import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from analyzer import StockSentimentAnalyzer
from utils import normalize_sentiment

st.set_page_config(page_title="Stock Sentiment Analyzer", layout="wide")

st.title("📈 Stock Sentiment Analyzer (Reddit + Llama3 via Ollama)")

stock_symbol = st.text_input("Enter stock symbol (e.g., AAPL)", value="AAPL")

if st.button("Analyze"):
    analyzer = StockSentimentAnalyzer()
    with st.spinner(f"Fetching Reddit posts and analyzing {stock_symbol}..."):
        prediction, posts = analyzer.predict_trend(stock_symbol)
        stock_data = analyzer.get_stock_data(stock_symbol)

    st.subheader(f"Final Trend Prediction for {stock_symbol}:")
    st.success(prediction)

    if posts is not None and not posts.empty:
        st.subheader("Recent Reddit Posts & Sentiment")

        sentiment_expanded = []
        for _, row in posts.iterrows():
            if isinstance(row["llama_sentiment"], dict):
                sentiment_expanded.append({
                    "Title": row["title"],
                    "Growth": normalize_sentiment(row["llama_sentiment"].get("growth_sentiment")),
                    "Employment": normalize_sentiment(row["llama_sentiment"].get("employment_sentiment")),
                    "Inflation": normalize_sentiment(row["llama_sentiment"].get("inflation_sentiment")),
                    "Score": row["score"]
                })

        if sentiment_expanded:
            df_sent = pd.DataFrame(sentiment_expanded)
            st.dataframe(df_sent.head(10))

            st.subheader("📊 Easy-to-Understand Sentiment Summary")

            growth_counts = df_sent["Growth"].value_counts(normalize=True) * 100
            emp_counts = df_sent["Employment"].value_counts(normalize=True) * 100
            infl_counts = df_sent["Inflation"].value_counts(normalize=True) * 100

            def fmt_summary(label, counts):
                pos = counts.get("positive", 0)
                neg = counts.get("negative", 0)
                neu = counts.get("neutral", 0)
                return f"**{label} Sentiment** → 🟢 {pos:.0f}% Positive | 🔴 {neg:.0f}% Negative | ⚪ {neu:.0f}% Neutral"

            st.write(fmt_summary("Growth", growth_counts))
            st.write(fmt_summary("Employment", emp_counts))
            st.write(fmt_summary("Inflation", infl_counts))

            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**Growth Sentiment**")
                st.bar_chart(df_sent["Growth"].value_counts())
            with col2:
                st.write("**Employment Sentiment**")
                st.bar_chart(df_sent["Employment"].value_counts())
            with col3:
                st.write("**Inflation Sentiment**")
                st.bar_chart(df_sent["Inflation"].value_counts())

            avg_growth = growth_counts.get("positive", 0) - growth_counts.get("negative", 0)
            if avg_growth > 20:
                verdict = "💡 **Overall Market Mood: Bullish 📈**"
            elif avg_growth < -20:
                verdict = "💡 **Overall Market Mood: Bearish 📉**"
            else:
                verdict = "💡 **Overall Market Mood: Neutral ⚪**"
            st.success(verdict)

    if stock_data is not None and not stock_data.empty:
        st.subheader(f"{stock_symbol} Stock Price vs Sentiment (Last 30 days)")

        if posts is not None and not posts.empty:
            posts["date"] = posts["created_utc"].dt.date
            daily_sentiment = posts.groupby("date")["score"].mean().reset_index()
            daily_sentiment["date"] = pd.to_datetime(daily_sentiment["date"])

            stock_data_reset = stock_data.reset_index()[["Date", "Close"]]
            stock_data_reset["Date"] = pd.to_datetime(stock_data_reset["Date"]).dt.tz_localize(None)
            daily_sentiment["date"] = pd.to_datetime(daily_sentiment["date"]).dt.tz_localize(None)

            merged = pd.merge(stock_data_reset, daily_sentiment, left_on="Date", right_on="date", how="left")

            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(merged["Date"], merged["Close"], color="blue", label="Stock Price (Close)")
            ax1.set_ylabel("Stock Price (USD)", color="blue")
            ax1.tick_params(axis="y", labelcolor="blue")

            ax2 = ax1.twinx()
            ax2.plot(merged["Date"], merged["score"], color="red", linestyle="--", label="Sentiment Score")
            ax2.set_ylabel("Sentiment Score", color="red")
            ax2.tick_params(axis="y", labelcolor="red")

            fig.suptitle(f"{stock_symbol} Stock Price vs Reddit Sentiment", fontsize=14)
            fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
            st.pyplot(fig)
        else:
            st.line_chart(stock_data["Close"])
