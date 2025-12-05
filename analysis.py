# analysis.py

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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
    A simple rule-based classifier until we train a Scikit-learn model.
    """
    if compound_score < -0.2 and mood_rating <= 4:
        return "High"
    elif compound_score < 0.2 and mood_rating <= 6:
        return "Medium"
    else:
        return "Low"