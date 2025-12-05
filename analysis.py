# analysis.py

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from transformers import pipeline

# Download VADER lexicon if not already present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon')

# Initialize the Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """
    Performs VADER sentiment analysis on the input text.
    VADER returns scores for negative (neg), neutral (neu), positive (pos),
    and a normalized, weighted composite score (compound).
    
    The compound score is the most useful for overall mood tracking.
    """
    
    if not text:
        return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
        
    vs = analyzer.polarity_scores(text)
    
    # We will primarily use the compound score for the main mood metric
    compound_score = vs['compound']
    
    return compound_score
    
# Placeholder for the future ML Stress Classifier
def classify_stress_level(compound_score, mood_rating):
    """
    A rule-based classifier that considers both sentiment and mood rating.
    Returns stress level as 'High', 'Medium', or 'Low'.
    
    Logic:
    - High: Very negative sentiment OR very low mood rating
    - Medium: Moderately negative sentiment OR low-to-moderate mood
    - Low: Positive sentiment AND good mood rating
    """
    # High Stress: Strong negative indicators
    if compound_score < -0.5 or mood_rating <= 3:
        return "High"
    
    # Medium Stress: Moderate negative indicators
    elif compound_score < 0 or mood_rating <= 5:
        return "Medium"
    
    # Low Stress: Positive indicators
    else:
        return "Low"


def generate_suggestions(compound_score: float, stress_level: str, history_df: pd.DataFrame) -> list:
    """
    Generates tailored wellness suggestions based on current analysis and historical trends.
    """
    suggestions = []

    # --- 1. Current Entry Feedback ---
    
    if stress_level == "High":
        suggestions.append("🛑 **Immediate Focus: Stress Reduction.** Try taking 10 deep breaths, stepping away from your task for 5 minutes, or listening to a calming playlist.")
    elif compound_score < 0 and stress_level == "Medium":
        suggestions.append("⚖️ **Mood Check-in:** Your entry shows a negative tone. Consider reaching out to a friend or engaging in a hobby you enjoy to shift your focus.")
    elif compound_score > 0.5:
        suggestions.append("🎉 **Positive Reinforcement:** Keep doing what you're doing! Note down specifically why today felt good so you can replicate it.")
    else:
        # Default suggestion for neutral/low-stress days
        suggestions.append("🧘 **Maintain Balance:** Focus on consistency today. Ensure you are staying hydrated and getting sufficient rest.")

    # --- 2. Historical Trend Feedback ---
    
    if not history_df.empty and len(history_df) >= 7:
        # Calculate recent average (last 7 days)
        recent_df = history_df.tail(7)
        recent_mood_avg = recent_df['Mood Rating (1-10)'].mean()
        recent_vader_avg = recent_df['VADER Sentiment Score'].mean()
        
        overall_mood_avg = history_df['Mood Rating (1-10)'].mean()
        
        # Trend 1: Declining Mood
        if recent_mood_avg < overall_mood_avg * 0.9: # If mood is 10% lower than overall average
            suggestions.append("📉 **Trend Alert:** Your mood rating over the last week has been lower than your average. Prioritize sleep and outdoor time this week.")
        
        # Trend 2: Chronic Stress
        high_stress_count = history_df[history_df['Stress Level'] == 'High'].shape[0]
        if high_stress_count > len(history_df) * 0.3: # If more than 30% of days are classified as High stress
            suggestions.append("🚨 **Long-Term Pattern:** A high percentage of your entries indicate high stress. Consider reviewing your schedule or seeking professional help if the pattern continues.")
            
    # Remove duplicates and return
    return list(set(suggestions))
generator = pipeline(
    "text-generation", 
    model="gpt2", # Example model, use something like 'distilgpt2' or a 7B LLM if possible
    device=-1 # Use CPU; change to 0 for GPU
)

def generate_suggestions_with_model(compound_score: float, stress_level: str, raw_text: str, history_df: pd.DataFrame) -> list:
    """
    Generates tailored wellness suggestions using a large language model prompt.
    """
    
    # 1. Prepare Historical Context
    historical_summary = "No long-term trend data available."
    if not history_df.empty and len(history_df) >= 7:
        recent_df = history_df.tail(7)
        recent_mood_avg = recent_df['Mood Rating (1-10)'].mean()
        high_stress_count = history_df[history_df['Stress Level'] == 'High'].shape[0]
        
        historical_summary = (
            f"Over the last 7 days, the average mood rating was {recent_mood_avg:.1f}. "
            f"Overall, {high_stress_count} days were classified as HIGH stress. "
        )

    # 2. Construct the Prompt (The key to personalized output)
    prompt = f"""
    You are an empathetic, non-judgmental wellness assistant. Your task is to provide 
    1-3 concise, actionable suggestions based on the user's input and history.
    
    [ANALYSIS]
    Current Entry Text: "{raw_text}"
    Sentiment Score (VADER): {compound_score:.2f}
    Calculated Stress Level: {stress_level}
    Historical Summary: {historical_summary}
    
    
    
    Generate the advice in a list format, starting with an emoji for each point.
    """

    # 3. Generate Advice from the Model
    try:
        response = generator(prompt, max_length=512, num_return_sequences=1, truncation=True)
        # The model's output needs cleaning to extract the list.
        # This is a simplification; a dedicated text model requires careful parsing.
        advice_text = response[0]['generated_text'].replace(prompt, '').strip()
        
        # Simple split on newlines for suggestions (requires the model to follow instructions!)
        suggestions = [line for line in advice_text.split('\n') if line.strip() and len(line) > 5]
        
    except Exception as e:
        # Fallback to the old rule-based system if the model fails
        print(f"Model generation failed: {e}. Falling back to rule-based system.")
        suggestions = generate_suggestions_rule_based(compound_score, stress_level, history_df) # You'd rename your original function

    return suggestions