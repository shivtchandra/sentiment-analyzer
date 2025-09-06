# sentiapp.py (full, integrated)
import os
import traceback
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from analyzer import StockSentimentAnalyzer

# helpers that must exist in your utils.py
# make_feature_row_from_recent(prices_df, posts_df) -> pd.DataFrame (1 row)
# apply_scaler_safe(X, scaler_np_path) -> (X_scaled, info_dict)
from utils import make_feature_row_from_recent, apply_scaler_safe

st.set_page_config(page_title="Stock Sentiment Analyzer", layout="wide")
st.title("📈 Stock Sentiment Analyzer — sentiment ↔ prices ↔ 1-month forecast (model optional)")

an = StockSentimentAnalyzer()

# --- Controls ---
col1, col2, col3 = st.columns(3)
with col1:
    stock_symbol = st.text_input("Stock symbol", value="AAPL").upper()
with col2:
    posts_limit = st.slider("Posts to fetch (last month)", 20, 1000, 300, 10)
with col3:
    use_titles_only = st.checkbox("Titles only", value=True)
debug = st.checkbox("Debug", value=False)

st.write("---")

# --- Analyze Button: sentiment + optional numeric forecast ---
if st.button("Analyze sentiment & map to prices"):
    with st.spinner("Fetching posts and computing sentiment..."):
        try:
            # predict_trend returns (label, posts_df, avg_score) OR (label, posts_df)
            res = an.predict_trend(stock_symbol, limit=posts_limit, use_titles_only=use_titles_only, debug=debug)
            if isinstance(res, (tuple, list)):
                if len(res) == 3:
                    pred_label, posts_df, avg_score = res
                elif len(res) == 2:
                    pred_label, posts_df = res
                    avg_score = None
                else:
                    raise RuntimeError("predict_trend returned unexpected shape")
            else:
                raise RuntimeError("predict_trend returned unexpected type")
        except Exception as e:
            st.error(f"Failed to fetch/analyze: {e}")
            if debug:
                st.text(traceback.format_exc())
            raise

    # --- Show sentiment label ---
    st.subheader("Sentiment forecast (last-month aggregate)")
    if isinstance(pred_label, str) and "Bullish" in pred_label:
        st.success(pred_label)
    elif isinstance(pred_label, str) and "Bearish" in pred_label:
        st.error(pred_label)
    else:
        st.info(pred_label)
    if avg_score is not None:
        st.write(f"Avg weighted sentiment score (recent month): **{avg_score:.4f}**")

    # --- Show recent posts & sentiment ---
    st.subheader("Recent posts & sentiment")
    if posts_df is None or posts_df.empty:
        st.warning("No posts found.")
    else:
        def _lab(x):
            if isinstance(x, dict):
                return x.get("label") if "label" in x else (x if isinstance(x, str) else "")
            return x if isinstance(x, str) else ""
        posts_view = posts_df.copy()
        # try to display sensible columns; fallbacks if absent
        if "aspect_sentiment" in posts_view.columns:
            posts_view["growth"] = posts_view["aspect_sentiment"].apply(lambda a: _lab(a.get("growth_sentiment") if isinstance(a, dict) else a))
        if "raw_score" not in posts_view.columns and "aspect_sentiment" in posts_view.columns:
            # compute a fallback raw_score for display (simple)
            def _compute_raw(a):
                if not isinstance(a, dict):
                    return 0.0
                val = 0.0
                w = {"growth_sentiment": 0.5, "employment_sentiment": 0.3, "inflation_sentiment": 0.2}
                for k, wt in w.items():
                    lab = a.get(k)
                    if isinstance(lab, str):
                        lab = lab.lower()
                        if lab == "positive": val += wt
                        elif lab == "negative": val -= wt
                return val
            posts_view["raw_score"] = posts_view["aspect_sentiment"].apply(_compute_raw)
        # safe display columns
        cols_to_show = [c for c in ["created_utc", "title", "growth", "raw_score", "score"] if c in posts_view.columns]
        st.dataframe(posts_view[cols_to_show].sort_values("score", ascending=False).head(200))

    # --- Price vs sentiment plot (last 90 days) ---
    try:
        stock_data = an.get_stock_data(stock_symbol, days=90)
    except Exception:
        stock_data = None

    if stock_data is not None and not stock_data.empty and posts_df is not None and not posts_df.empty:
        try:
            posts_df["created_utc"] = pd.to_datetime(posts_df["created_utc"], utc=True)
            posts_df["date"] = posts_df["created_utc"].dt.date
            daily_sent = posts_df.groupby("date")["score"].mean().reset_index().rename(columns={"score":"daily_sentiment"})
            daily_sent["date"] = pd.to_datetime(daily_sent["date"])
            stock_reset = stock_data.reset_index()[["Date", "Close"]]
            stock_reset["Date"] = pd.to_datetime(stock_reset["Date"]).dt.tz_localize(None)
            merged = pd.merge(stock_reset, daily_sent, left_on="Date", right_on="date", how="left")
            fig, ax1 = plt.subplots(figsize=(11, 5))
            ax1.plot(merged["Date"], merged["Close"], label="Close")
            ax1.set_ylabel("Price (Close)")
            ax2 = ax1.twinx()
            ax2.plot(merged["Date"], merged["daily_sentiment"], linestyle="--", label="Daily sentiment")
            ax2.axhline(0, color="gray", linewidth=1)
            ax2.set_ylabel("Daily sentiment")
            fig.legend(loc="upper left")
            st.pyplot(fig)
        except Exception as e:
            if debug:
                st.text("Plotting error: " + str(e))
                st.text(traceback.format_exc())
            st.info("Could not draw combined chart (data alignment issue).")
    elif stock_data is not None and not stock_data.empty:
        st.line_chart(stock_data["Close"])
    else:
        st.info("Price data not available.")

    # --- Numeric forecast (optional): build feature row & run model inference ---
    # st.subheader("📊 Numeric forecast (optional): model-based 30d return estimate")
    # feat_row = None
    # try:
    #     # Fetch longer price window for features (1 year)
    #     prices_for_feat = an.get_stock_data(stock_symbol, days=365)
    #     feat_row = make_feature_row_from_recent(prices_for_feat, posts_df)
    #     st.write("Feature row (used for inference):")
    #     st.json(feat_row.to_dict(orient="records")[0])
    # except Exception as e:
    #     if debug:
    #         st.text("Feature builder failed:")
    #         st.text(traceback.format_exc())
    #     st.info("Feature row not available (skipping model inference).")

    # # Try XGBoost first, then RF; use apply_scaler_safe to avoid immediate shape crashes
    # if feat_row is not None:
    #     did_predict = False

    #     # XGBoost path
    #     try:
    #         import xgboost as xgb
    #         scaler_path = "models/scaler.npz"
    #         xgb_model_path = "models/xgb_reg.model"  # keep same name you saved during training

    #         if os.path.exists(xgb_model_path):
    #             st.write("Attempting XGBoost inference using", xgb_model_path)
    #             booster = xgb.Booster()
    #             booster.load_model(xgb_model_path)

    #             x = feat_row.values.astype(float)
    #             if os.path.exists(scaler_path):
    #                 x_scaled, s_info = apply_scaler_safe(x, scaler_path)
    #                 st.write("Scaler info:", s_info)
    #             else:
    #                 x_scaled = x
    #                 st.write("No scaler found, using raw features (not recommended).")

    #             st.write("Input sample to model (first row):", x_scaled.tolist())
    #             dmat = xgb.DMatrix(x_scaled)
    #             preds = booster.predict(dmat)
    #             st.write("Raw xgb preds array:", preds.tolist())
    #             pred = float(preds[0])
    #             st.success(f"XGBoost predicted 30d return: {pred:.4%}")
    #             did_predict = True
    #     except Exception as e:
    #         if debug:
    #             st.text("XGBoost inference exception:")
    #             st.text(traceback.format_exc())
    #         else:
    #             st.write("XGBoost inference failed (enable Debug to see stack).")

    #     # RandomForest fall-back
    #     if not did_predict:
    #         try:
    #             import joblib
    #             rf_model_path = "models_rf/rf_reg.joblib"
    #             rf_scaler_path = "models_rf/scaler.joblib"
    #             if os.path.exists(rf_model_path) and os.path.exists(rf_scaler_path):
    #                 st.write("Attempting RandomForest inference using", rf_model_path)
    #                 rf = joblib.load(rf_model_path)
    #                 rf_scaler = joblib.load(rf_scaler_path)
    #                 x = feat_row.values.astype(float)
    #                 x_scaled = rf_scaler.transform(x)
    #                 st.write("Input sample to RF (after scaler):", x_scaled.tolist())
    #                 pred = float(rf.predict(x_scaled)[0])
    #                 st.success(f"RandomForest predicted 30d return: {pred:.4%}")
    #                 did_predict = True
    #         except Exception as e:
    #             if debug:
    #                 st.text("RF inference exception:")
    #                 st.text(traceback.format_exc())
    #             else:
    #                 st.write("RF inference failed (enable Debug to see stack).")

    #     if not did_predict:
    #         st.info("No usable saved model found. Place XGBoost model in models/xgb_reg.model and scaler in models/scaler.npz, or RF in models_rf/. You can also use the baseline while debugging.")

    # # baseline fallback for interactive exploration
    # st.markdown("**Baseline (debug)**: quick interpretable baseline (not a trained model).")
    # try:
    #     if feat_row is not None:
    #         r = feat_row.iloc[0]
    #         baseline = r["recent_sent"] * 0.03 + 0.01 * (r["pos_ratio_30d"] - 0.5) + 0.002 * r["logret_1"]
    #         st.info(f"Baseline predicted 30d return (debug): {baseline:.4%}")
    # except Exception:
    #     pass

st.write("---")
st.markdown(
    """
Notes:
- This app currently uses `predict_trend()` from `analyzer.py` (calls your FinBERT or fallback aspect sentiment).
- For numeric inference, the saved model and scaler must match the feature columns & order returned by `make_feature_row_from_recent`.
- If predictions look constant across tickers, run the diagnostic script to verify feature variability and scaler shapes.
"""
)
