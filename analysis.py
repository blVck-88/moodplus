# analysis.py

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import google.generativeai as genai


# ---------------------------------------------------------------------------
# 1. VADER SENTIMENT INITIALIZATION
# ---------------------------------------------------------------------------

# Ensure VADER lexicon is available
try:
    nltk.data.find("sentiment/vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon")

analyzer = SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# 2. SENTIMENT ANALYSIS
# ---------------------------------------------------------------------------

def analyze_sentiment(text: str) -> float:
    """Returns the compound score (-1 to +1)."""
    if not text:
        return 0.0
    score = analyzer.polarity_scores(text)
    return score.get("compound", 0.0)


# ---------------------------------------------------------------------------
# 3. STRESS CLASSIFICATION
# ---------------------------------------------------------------------------

def classify_stress_level(compound_score: float, mood_rating: float) -> str:
    """Simple rule-based stress classifier."""
    if compound_score <= -0.5 or mood_rating <= 3:
        return "High"
    if compound_score < 0 or mood_rating <= 5:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# 4. RULE-BASED SUGGESTIONS
# ---------------------------------------------------------------------------

def generate_suggestions(compound_score: float, stress_level: str, history_df: pd.DataFrame) -> list:
    """Deterministic rule-based wellness suggestions."""
    suggestions = []

    # Current day feedback
    if stress_level == "High":
        suggestions.append("🛑 **Immediate Focus:** Step away briefly and slow your breathing.")
    elif compound_score < 0 and stress_level == "Medium":
        suggestions.append("⚖️ **Mood Check-in:** Reach out to someone or try a relaxing activity.")
    elif compound_score > 0.5:
        suggestions.append("🎉 **Positive Momentum:** Note what made today good.")
    else:
        suggestions.append("🧘 **Maintain Balance:** Keep your routine steady today.")

    # Historical trends
    if not history_df.empty and len(history_df) >= 7:
        history_df["Mood Rating (1-10)"] = pd.to_numeric(history_df["Mood Rating (1-10)"], errors="ignore")
        recent = history_df.tail(7)

        recent_mood = recent["Mood Rating (1-10)"].mean()
        overall_mood = history_df["Mood Rating (1-10)"].mean()

        # Trend: dropping mood
        if recent_mood < (overall_mood * 0.9):
            suggestions.append("📉 **Recent Drop:** Your last week’s mood is lower than usual. Prioritize rest and daylight.")

        # Trend: many high-stress days
        high_days = (history_df["Stress Level"] == "High").sum()
        if high_days > len(history_df) * 0.3:
            suggestions.append("🚨 **Frequent Stress:** Your entries show ongoing stress. Consider light schedule adjustments.")

    return list(dict.fromkeys(suggestions))  # Remove duplicates, keep order


# ---------------------------------------------------------------------------
# 5. GEMINI AI INITIALIZATION (unchanged as requested)
# ---------------------------------------------------------------------------

GOOGLE_API_KEY = "AIzaSyAS8c21XgXBXw_65ouRIXgRRbrKZsROldU"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-pro")


# ---------------------------------------------------------------------------
# 6. GEMINI-POWERED SUGGESTIONS
# ---------------------------------------------------------------------------

def generate_suggestions_with_model(
    compound_score: float,
    stress_level: str,
    raw_text: str,
    history_df: pd.DataFrame,
) -> list:
    """
    Returns Gemini-generated suggestions, guaranteed to always produce output.
    """

    # --- Historical summary ---
    if not history_df.empty and len(history_df) >= 7:
        recent = history_df.tail(7)
        recent_mood = pd.to_numeric(
            recent["Mood Rating (1-10)"], errors="coerce"
        ).mean()
        high_stress_count = (history_df["Stress Level"] == "High").sum()

        history_summary = (
            f"Last 7-day average mood: {recent_mood:.1f}. "
            f"High-stress days recorded: {high_stress_count}."
        )
    else:
        history_summary = "No long-term trend data available."

    # --- Prompt ---
    prompt = f"""
You are a supportive wellness assistant. Provide only 2–3 concise, actionable suggestions.

[DATA]
User Entry: "{raw_text}"
Sentiment Score: {compound_score:.2f}
Stress Level: {stress_level}
Historical Summary: {history_summary}

[INSTRUCTIONS]
- Return 2–3 bullet points.
- Each line must contain: emoji + bold title + colon + short action.
- No extra explanations or numbering.
- No repeated ideas.

Example:
🛑 **Stress Relief:** Take 10 slow breaths.
🤝 **Connection:** Reach out to someone close.
🧘 **Recovery:** Prioritize sleep tonight.

Now generate the suggestions:
"""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        raw_lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

        suggestions = []

        # AI Output Extraction:
        # Accept any line beginning with an emoji (or non-alphanumeric symbol)
        for line in raw_lines:
            stripped = line.lstrip()
            if stripped and not stripped[0].isalnum():  # Accept emojis & symbols
                suggestions.append(f"• {stripped}")

        # If suggestions list is empty, fall back to ANY meaningful line
        if not suggestions:
            for ln in raw_lines:
                if ln.strip():
                    suggestions.append(f"• {ln.strip()}")

        # If still empty, use rule-based fallback
        if not suggestions:
            fallback = generate_suggestions(compound_score, stress_level, history_df)
            return [f"• {s}" for s in fallback]

        return suggestions

    except Exception:
        # Hard fallback: rule-based
        fallback = generate_suggestions(compound_score, stress_level, history_df)
        return [f"• {s}" for s in fallback]
