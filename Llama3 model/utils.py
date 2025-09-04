import ollama
import json
import re
import requests

def resolve_symbol(query: str):
    """
    Resolve a company name or ticker symbol into a proper Yahoo Finance ticker.
    Hybrid approach:
      - If input is already a symbol (like AAPL), it usually works directly.
      - If input is a name (like Apple), try Yahoo Finance search API.
    """
    query = query.strip()

    # If user already enters uppercase letters (likely a ticker), return directly
    if query.isupper() and len(query) <= 5:  
        return query

    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        r = requests.get(url, timeout=5).json()
        if "quotes" in r and r["quotes"]:
            return r["quotes"][0]["symbol"]  # pick the top result
    except Exception as e:
        print(f"[DEBUG] resolve_symbol failed for {query}: {e}")

    # Fallback: return input as-is (let yfinance handle it)
    return query

# -----------------------------
# Sentiment Normalizer
# -----------------------------
def normalize_sentiment(value):
    if value is None:
        return "neutral"
    if isinstance(value, (int, float)):
        return "positive" if value > 0 else "negative" if value < 0 else "neutral"
    if isinstance(value, dict) and "polarity" in value:
        return "positive" if value["polarity"] > 0 else "negative" if value["polarity"] < 0 else "neutral"
    if isinstance(value, str):
        v = value.strip().lower()
        if "pos" in v: return "positive"
        if "neg" in v: return "negative"
        if "neu" in v: return "neutral"
    return "neutral"

# -----------------------------
# Llama sentiment via Ollama
# -----------------------------
def llama_sentiment(text: str):
    prompt = f"""
Analyze the following economic/financial sentence and return structured sentiment JSON
with exactly these fields: growth_sentiment, employment_sentiment, inflation_sentiment.

Sentence: {text}

Output only valid JSON and no explanation.
"""
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    content = response['message']['content']
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            return {"raw_output": content}
    return {"raw_output": content}
