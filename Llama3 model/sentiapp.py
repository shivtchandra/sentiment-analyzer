import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from analyzer import StockSentimentAnalyzer
from utils import normalize_sentiment, resolve_symbol

# Predefined indices (Yahoo Finance symbols)
indices = {
    "S&P 500": "^GSPC",
    "NASDAQ Composite": "^IXIC",
    "Dow Jones": "^DJI",
    "Russell 2000": "^RUT"
}

# --- Input Section ---
option = st.selectbox("Choose an index (or select None to type a stock):", ["None"] + list(indices.keys()))

if option != "None":
    # User picked an index
    stock_symbol = indices[option]
    user_input = option
    st.info(f"Selected index: **{option}** → Symbol: **{stock_symbol}**")
else:
    # Fallback: user types stock name/symbol
    user_input = st.text_input("Enter stock name or symbol", value="Apple")
    stock_symbol = resolve_symbol(user_input)

    if stock_symbol and stock_symbol != user_input:
        st.info(f"Resolved **{user_input}** → **{stock_symbol}**")

debug_mode = st.checkbox("Enable Debug Mode (print raw outputs)", value=False)

# --- Run Analysis Button ---
if st.button("Analyze"):
    if not stock_symbol or stock_symbol.strip() == "":
        st.error(f"❌ Could not resolve **{user_input}** into a valid stock symbol. Please try again.")
    else:
        analyzer = StockSentimentAnalyzer()
        with st.spinner(f"Fetching Reddit posts and analyzing {stock_symbol}..."):
            prediction, posts = analyzer.predict_trend(stock_symbol, debug=debug_mode)
            stock_data = analyzer.get_stock_data(stock_symbol)

        # ✅ Save results into session_state
        st.session_state["prediction"] = prediction
        st.session_state["posts"] = posts
        st.session_state["stock_data"] = stock_data
        st.session_state["stock_symbol"] = stock_symbol

# --- Use Saved Results (so toggles like Heatmap/Line Chart still work) ---
if "posts" in st.session_state and "stock_data" in st.session_state:
    prediction = st.session_state["prediction"]
    posts = st.session_state["posts"]
    stock_data = st.session_state["stock_data"]
    stock_symbol = st.session_state["stock_symbol"]

    # --- Prediction Output ---
    st.subheader(f"Final Trend Prediction for {stock_symbol}:")
    st.success(prediction)

    # --- Sentiment Details ---
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

    # --- Charts: Stock Price vs Sentiment ---
    if stock_data is not None and not stock_data.empty:
        st.subheader(f"{stock_symbol} Stock Price vs Sentiment (Last 30 days)")

        if posts is not None and not posts.empty:
            posts["date"] = posts["created_utc"].dt.date
            daily_sentiment = posts.groupby("date")["score"].mean().reset_index()
            daily_sentiment["date"] = pd.to_datetime(daily_sentiment["date"])
            daily_sentiment["score_scaled"] = daily_sentiment["score"] * 100

            stock_data_reset = stock_data.reset_index()[["Date", "Close"]]
            stock_data_reset["Date"] = pd.to_datetime(stock_data_reset["Date"]).dt.tz_localize(None)
            daily_sentiment["date"] = pd.to_datetime(daily_sentiment["date"]).dt.tz_localize(None)

            merged = pd.merge(stock_data_reset, daily_sentiment, left_on="Date", right_on="date", how="left")

            # ✅ Toggle between Line Chart and Heatmap
           

            if chart_type == "Line Chart":
                fig, ax1 = plt.subplots(figsize=(10, 5))
                ax1.plot(merged["Date"], merged["Close"], color="blue", label="Stock Price (Close)")
                ax1.set_ylabel("Stock Price (USD)", color="blue")
                ax1.tick_params(axis="y", labelcolor="blue")

                ax2 = ax1.twinx()
                ax2.plot(merged["Date"], merged["score_scaled"], color="red", linestyle="--", label="Sentiment Score (scaled x100)")
                ax2.set_ylabel("Sentiment Score (scaled)", color="red")
                ax2.tick_params(axis="y", labelcolor="red")

                fig.suptitle(f"{stock_symbol} Stock Price vs Reddit Sentiment", fontsize=14)
                fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
                st.pyplot(fig)

           
            
        else:
            st.line_chart(stock_data["Close"])
