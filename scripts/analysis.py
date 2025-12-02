import sys
import os
import pandas as pd
from tqdm import tqdm
# --- Sentiment Analysis Imports ---
# NOTE: The user MUST install the nltk library for VADER analysis:
# pip install nltk

# --- Sentiment Analysis Imports and NLTK Setup ---
# NOTE: The user MUST install the nltk library for VADER analysis:
# pip install nltk
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- NLTK PATH SETUP AND RESOURCE DOWNLOADS (CRITICAL FIX) ---
# NOTE: This block ensures NLTK resources are found and downloaded reliably.
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except ImportError:
    print("Error: The 'nltk' library is not installed. Please run: pip install nltk")
    sys.exit(1)


# 1. Define a reliable, accessible directory for NLTK data (e.g., in the user's home folder)
NLTK_DATA_DIR = os.path.join(os.path.expanduser('~'), '.nltk_data')

# 2. Add this path to NLTK's data search path
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)
    print(f"Set NLTK data path to: {NLTK_DATA_DIR}")

# 3. Define all resources needed for VADER and Text Normalization
REQUIRED_NLTK_DATA = {
    'vader_lexicon': 'sentiment/vader_lexicon.zip', # For VADER
    'punkt': 'tokenizers/punkt',                   # For word_tokenize
    'punkt_tab': 'tokenizers/punkt_tab',
    'wordnet': 'corpora/wordnet',                  # For Lemmatization
    'stopwords': 'corpora/stopwords'               # For Stopword Removal
}

# 4. Check and download each resource individually using the configured path
for name, resource_path in REQUIRED_NLTK_DATA.items():
    try:
        # Check if the resource is already downloaded in the search paths
        nltk.data.find(resource_path)
    except LookupError:
        # Download the resource to the primary search location if it's not found
        print(f"Downloading missing NLTK resource: {name}...")
        # We assume the download will now save correctly to one of the valid paths
        nltk.download(name, quiet=True)
# --- END NLTK SETUP ---


# --- Thematic Analysis Imports ---
# NOTE: The user MUST install the scikit-learn library for TF-IDF:
# pip install scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
# --- End Thematic Analysis Imports ---


# ---Text Normalization Function ---

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs tokenization, stop-word removal, and lemmatization on the review text.
    Creates a new 'normalized_review' column for thematic analysis.
    """
    print("\n--- Starting Text Normalization (Lemmatization) ---")
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def apply_normalization(text):
        if not isinstance(text, str):
            return ""
        # 1. Lowercase and Tokenize
        words = nltk.word_tokenize(text.lower())
        # 2. Stop-word removal, filtering non-alphabetic, and Lemmatization
        normalized_words = [
            lemmatizer.lemmatize(word)
            for word in words
            # Keep only words (not punctuation) and filter out stop words
            if word.isalpha() and word not in stop_words
        ]
        return " ".join(normalized_words)

    # Apply normalization to the 'review' column
    df['normalized_review'] = df['review'].apply(apply_normalization)
    print("Text normalization complete. New column 'normalized_review' created.")
    return df

# --- Sentiment Analysis Function ---

def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes sentiment scores using VADER and classifies reviews into 
    Positive, Negative, or Neutral.
    
    NOTE: This uses VADER from the NLTK library (the simpler alternative).

    Args:
        df (pd.DataFrame): The DataFrame with a 'review' column.

    Returns:
        pd.DataFrame: The DataFrame with 'compound_score' and 'sentiment_label' columns.
    """
    # Sentiment analysis will now fail with ImportError if nltk or VADER is missing,
    # ensuring the packages are installed before proceeding.
    
    print("\n--- Starting Sentiment Analysis (using VADER) ---")
    sia = SentimentIntensityAnalyzer()

    # Calculate VADER sentiment scores for each review
    df['vader_scores'] = df['review'].apply(lambda review: sia.polarity_scores(str(review)))
    df['compound_score'] = df['vader_scores'].apply(lambda score: score['compound'])
    
    # Classify sentiment based on the compound score
    # These thresholds are standard for VADER:
    # Score >= 0.05 is positive, Score <= -0.05 is negative, otherwise neutral.
    def classify_sentiment(score):
        if score >= 0.05:
            return 'POSITIVE'
        elif score <= -0.05:
            return 'NEGATIVE'
        else:
            return 'NEUTRAL'

    df['sentiment_label'] = df['compound_score'].apply(classify_sentiment)
    
    print(f"Sentiment analysis complete. Added 'compound_score' and 'sentiment_label'.")
    return df.drop(columns=['vader_scores'], errors='ignore')


# --- Aggregation Function ---

def aggregate_sentiment(df: pd.DataFrame):
    """
    Aggregates sentiment results by bank and rating.

    Args:
        df (pd.DataFrame): The DataFrame with 'bank', 'rating', and sentiment columns.
    
    Returns:
        pd.DataFrame: Aggregated results showing mean compound score and sentiment counts.
    """
    if 'compound_score' not in df.columns:
        print("Skipping aggregation: Sentiment scores not available.")
        return None

    print("\n--- Starting Sentiment Aggregation ---")

    # 1. Mean Compound Score by Bank and Rating
    mean_sentiment = df.groupby(['bank', 'rating'])['compound_score'].mean().reset_index()
    mean_sentiment = mean_sentiment.rename(columns={'compound_score': 'mean_compound_score'})
    
    print("\nMean Compound Score per Bank and Rating:")
    print(mean_sentiment)

    # 2. Count of Sentiment Labels by Bank and Rating
    sentiment_counts = df.groupby(['bank', 'rating', 'sentiment_label']).size().reset_index(name='count')
    
    # Pivot the table to have sentiment labels as columns
    sentiment_pivot = sentiment_counts.pivot_table(
        index=['bank', 'rating'], 
        columns='sentiment_label', 
        values='count', 
        fill_value=0
    ).reset_index()
    
    # Ensure all labels are present, even if zero
    for label in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
        if label not in sentiment_pivot.columns:
            sentiment_pivot[label] = 0
            
    # Calculate the total reviews for percentage calculation
    sentiment_pivot['total_reviews'] = sentiment_pivot[['POSITIVE', 'NEGATIVE', 'NEUTRAL']].sum(axis=1)
    
    print("\nSentiment Label Counts per Bank and Rating:")
    print(sentiment_pivot)
    
    # Merge the mean score and counts into a final aggregate table
    final_aggregate = pd.merge(mean_sentiment, sentiment_pivot, on=['bank', 'rating'])
    
    print("\n--- Final Aggregated Results ---")
    print(final_aggregate)
    
    return final_aggregate

    
# ---Thematic Analysis Function

def perform_thematic_analysis(df: pd.DataFrame, top_n_features: int = 50) -> tuple[pd.DataFrame, dict]:
    """
    Performs Keyword Extraction using TF-IDF and applies a Rule-Based
    Clustering to group keywords into actionable themes. (V2)
    
    A theme refers to a recurring concept or topic within user reviews. For this
    challenge, themes will help summarize user feedback into actionable
    categories for the banks.

    Args:
        df (pd.DataFrame): The DataFrame containing the 'review' column.
        top_n_features (int): The number of top keywords/n-grams to extract.

    Returns:
        tuple[pd.DataFrame, dict]: 
            - A DataFrame of the extracted keywords and their assigned themes.
            - The keyword-to-theme mapping dictionary for review tagging.
    """
    if df.empty or 'review' not in df.columns:
        print("Skipping thematic analysis: DataFrame is empty or missing 'review' column.")
        return pd.DataFrame(), {}

    print("\n--- Starting Thematic Analysis (Keyword Extraction using TF-IDF) ---")
    
    # 1. Keyword/N-gram Extraction using TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2), # Extract single words and two-word phrases
        max_features=top_n_features
    )
    
    tfidf_matrix = vectorizer.fit_transform(df['review'].astype(str))
    feature_names = vectorizer.get_feature_names_out()
    
    total_scores = tfidf_matrix.sum(axis=0).A1
    keyword_scores = pd.Series(total_scores, index=feature_names).sort_values(ascending=False)
    
    keywords_df = pd.DataFrame(keyword_scores, columns=['tfidf_score']).reset_index()
    keywords_df.rename(columns={'index': 'keyword_phrase'}, inplace=True)
    
    print(f"Extracted {len(keywords_df)} top keywords/n-grams.")

    # 2. Rule-Based Clustering/Grouping (Manual Logic)
    theme_mapping = {
        'Account Access & Stability': ['login', 'fingerprint', 'cant', 'app crashed', 'keeps crashing', 'open app', 'password', 'face id'],
        'Transaction Performance': ['transfer', 'slow', 'pending', 'takes long', 'money', 'transaction', 'instant'],
        'User Interface & Experience': ['interface', 'design', 'ui', 'easy use', 'user friendly', 'update', 'bad update', 'navigation'],
        'Customer Support & Service': ['customer service', 'branch', 'help', 'contact', 'support', 'call center', 'speak to'],
        'Feature Requests & Missing Functionality': ['feature', 'need', 'add', 'option', 'dark mode', 'missing']
    }
    
    # Flatten the mapping for fast lookups (Used for assigning themes to reviews)
    keyword_to_theme = {}
    for theme, keywords in theme_mapping.items():
        for keyword in keywords:
            # Storing keyword in lowercase for case-insensitive matching later
            keyword_to_theme[keyword.lower()] = theme 

    def assign_theme(keyword_phrase):
        # Checks against both the exact phrase and its components for n-grams
        for keyword, theme in keyword_to_theme.items():
            if keyword in keyword_phrase:
                return theme
        return 'Other/General'

    keywords_df['assigned_theme'] = keywords_df['keyword_phrase'].apply(assign_theme)
    
    # 3. Document Grouping Logic and Display
    print("\n--- Thematic Grouping Logic (Documentation) ---")
    print("Keywords were grouped into 5 themes based on keyword matching:")
    for theme, keywords in theme_mapping.items():
        print(f" - {theme}: Matches keywords like '{', '.join(keywords[:4])}...'")
        
    print("\nTop Keywords and Their Assigned Themes:")
    print(keywords_df[['keyword_phrase', 'assigned_theme']].head(15))
    
    # Filter for keywords that were successfully mapped to a defined theme for next steps
    themed_keywords_df = keywords_df[keywords_df['assigned_theme'] != 'Other/General']
    
    return themed_keywords_df, keyword_to_theme

# --- Theme Assignment Function ---

def assign_themes_to_reviews(df: pd.DataFrame, keyword_to_theme: dict) -> pd.DataFrame:
    """
    Tags each review with the relevant themes by checking for the presence of
    the identified significant keywords from the theme mapping.

    Args:
        df (pd.DataFrame): The DataFrame containing the 'review' column and sentiment data.
        keyword_to_theme (dict): Mapping of significant keyword/phrase (lowercase) to its theme.

    Returns:
        pd.DataFrame: The input DataFrame with a new 'identified_themes' column (list of strings).
    """
    print("\n--- Starting Theme Assignment to Individual Reviews ---")

    def tag_themes(review_text):
        if not isinstance(review_text, str):
            return []
        
        # Lowercase the review for case-insensitive matching against keywords
        review_text = review_text.lower()
        themes = set()
        
        # Check for presence of each keyword/phrase in the review
        for keyword, theme in keyword_to_theme.items():
            if keyword in review_text:
                themes.add(theme)
                
        # Return as a list
        return list(themes)

    df['identified_themes'] = df['review'].apply(tag_themes)
    
    print(f"Theme assignment complete. Added 'identified_themes' column.")
    return df
