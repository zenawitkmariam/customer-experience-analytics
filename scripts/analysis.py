import sys
import os
import pandas as pd
from tqdm import tqdm
import collections
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


# --- Visualization Imports ---
import matplotlib.pyplot as plt
import numpy as np
# --- End Visualization Imports ---

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
# --- 8. NEW: Drivers and Pain Points Identification ---

def identify_drivers_and_pain_points(df: pd.DataFrame, top_n: int = 3) -> dict:
    """
    Identifies the top N thematic drivers (themes in POSITIVE reviews) and 
    pain points (themes in NEGATIVE reviews) for each bank.

    Args:
        df (pd.DataFrame): The DataFrame containing 'bank', 'sentiment_label', 
                           and 'identified_themes'.
        top_n (int): The number of top drivers/pain points to identify.

    Returns:
        dict: A dictionary structured as {bank_name: {drivers: [], pain_points: []}}.
    """
    if 'identified_themes' not in df.columns:
        print("Skipping drivers/pain points identification: Missing 'identified_themes' column.")
        return {}
        
    print("\n--- Identifying Top Drivers and Pain Points Per Bank ---")
    
    banks = df['bank'].unique()
    analysis_results = {}
    
    for bank in tqdm(banks, desc="Analyzing Drivers & Pain Points"):
        bank_data = df[df['bank'] == bank]
        
        # --- 1. Identify Drivers (Themes in POSITIVE Reviews) ---
        positive_reviews = bank_data[bank_data['sentiment_label'] == 'POSITIVE']
        
        # Flatten the list of lists in 'identified_themes' column
        all_positive_themes = [theme for sublist in positive_reviews['identified_themes'] for theme in sublist]
        
        # Count the frequency of each theme
        driver_counts = collections.Counter(all_positive_themes)
        
        # Get the top N drivers as a list of tuples (Theme, Count)
        top_drivers = driver_counts.most_common(top_n)
        
        
        # --- 2. Identify Pain Points (Themes in NEGATIVE Reviews) ---
        negative_reviews = bank_data[bank_data['sentiment_label'] == 'NEGATIVE']
        
        # Flatten the list of lists
        all_negative_themes = [theme for sublist in negative_reviews['identified_themes'] for theme in sublist]
        
        # Count the frequency of each theme
        pain_point_counts = collections.Counter(all_negative_themes)
        
        # Get the top N pain points as a list of tuples (Theme, Count)
        top_pain_points = pain_point_counts.most_common(top_n)
        
        # Store results
        analysis_results[bank] = {
            'drivers': [f"{theme} ({count} mentions)" for theme, count in top_drivers],
            'pain_points': [f"{theme} ({count} mentions)" for theme, count in top_pain_points]
        }
        
    print("Driver and Pain Point analysis complete.")
    return analysis_results

    
# --- Theme Sentiment Distribution for Comparison ---
def get_theme_sentiment_distribution(df: pd.DataFrame, min_theme_mentions: int = 50) -> pd.DataFrame:
    """
    Calculates the distribution of POSITIVE/NEGATIVE/NEUTRAL sentiment 
    for major themes, broken down by bank.

    Args:
        df: DataFrame with 'bank', 'sentiment_label', and 'identified_themes'.
        min_theme_mentions: Minimum total mentions required for a theme to be included.

    Returns:
        pd.DataFrame: A comparative table of theme sentiment percentages.
    """
    if 'identified_themes' not in df.columns:
        print("Error: 'identified_themes' column is missing for comparative analysis.")
        return pd.DataFrame()
        
    print("\n--- Calculating Comparative Theme Sentiment Distribution ---")

    # 1. Expand the DataFrame: one row per theme mention
    exploded_df = df.explode('identified_themes')
    exploded_df = exploded_df.rename(columns={'identified_themes': 'theme'})
    
    # Remove rows where theme is None (i.e., reviews without a key theme)
    exploded_df.dropna(subset=['theme'], inplace=True)
    
    # 2. Filter out low-frequency themes globally
    theme_counts = exploded_df['theme'].value_counts()
    frequent_themes = theme_counts[theme_counts >= min_theme_mentions].index
    
    comparative_df = exploded_df[exploded_df['theme'].isin(frequent_themes)]
    
    # 3. Calculate sentiment count per bank and theme
    theme_sentiment_counts = comparative_df.groupby(['bank', 'theme', 'sentiment_label']).size().reset_index(name='count')
    
    # 4. Pivot and calculate total
    pivot_df = theme_sentiment_counts.pivot_table(
        index=['bank', 'theme'], 
        columns='sentiment_label', 
        values='count', 
        fill_value=0
    ).reset_index()

    # Ensure all labels exist
    for label in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
        if label not in pivot_df.columns:
            pivot_df[label] = 0

    pivot_df['Total Mentions'] = pivot_df[['POSITIVE', 'NEGATIVE', 'NEUTRAL']].sum(axis=1)

    # 5. Calculate percentages
    for label in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
        pivot_df[f'{label} %'] = (pivot_df[label] / pivot_df['Total Mentions']) * 100

    print("Comparative Theme Sentiment Distribution Table:")
    print(pivot_df[['bank', 'theme', 'POSITIVE %', 'NEGATIVE %', 'Total Mentions']].sort_values(by=['theme', 'bank']))
    
    return pivot_df

# --- Visualization ---

def plot_theme_sentiment_distribution(comparative_df: pd.DataFrame):
    """
    Creates a comparative stacked bar chart of theme sentiment across banks.
    """
    if comparative_df.empty:
        print("Cannot plot comparison: comparative_df is empty.")
        return

    # Select the top 4 themes overall for plotting clarity
    themes_to_plot = comparative_df['theme'].value_counts().nlargest(4).index.tolist()
    plot_data = comparative_df[comparative_df['theme'].isin(themes_to_plot)].copy()

    # Prep for plotting
    plot_data['theme_bank'] = plot_data['theme'] + ' - ' + plot_data['bank']
    
    labels = ['POSITIVE %', 'NEUTRAL %', 'NEGATIVE %']
    colors = ['#10B981', '#FCD34D', '#EF4444'] # Tailwind green, amber, red

    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plotting the stacked bars
    current_bottom = [0] * len(plot_data)
    
    for label, color in zip(labels, colors):
        ax.bar(plot_data['theme_bank'], plot_data[label], bottom=current_bottom, label=label, color=color)
        current_bottom = current_bottom + plot_data[label]

    # Clean up the chart
    ax.set_title('Comparative Bank Performance by Key Thematic Sentiment', fontsize=16)
    ax.set_ylabel('Percentage of Reviews Mentioning Theme', fontsize=12)
    ax.set_xticks(np.arange(len(plot_data['theme_bank'])))
    ax.set_xticklabels(plot_data['theme_bank'], rotation=45, ha='right')
    
    # Add separating lines between themes for clarity
    theme_boundaries = []
    current_theme = None
    for i, (theme, bank) in enumerate(zip(plot_data['theme'], plot_data['bank'])):
        if theme != current_theme:
            if i > 0:
                theme_boundaries.append(i - 0.5)
            current_theme = theme
            
    for boundary in theme_boundaries:
        ax.axvline(x=boundary, color='gray', linestyle='--', linewidth=1)

    ax.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.close(fig)

    print("Comparative Theme Sentiment Visualization displayed.")


# ---Generate Improvement Suggestions Function ---

def generate_improvement_suggestions(comparative_df: pd.DataFrame) -> dict:
    """
    Generates actionable improvement suggestions based on the comparative sentiment data.

    Suggestions focus on:
    1. Addressing the largest internal pain point (highest Negative %).
    2. Addressing the largest competitive gap (biggest lag in Positive % vs. rival).
    
    Returns:
        dict: Suggestions structured by bank.
    """
    suggestions = {}
    banks = comparative_df['bank'].unique()
    
    if len(banks) < 2:
        return {} # Need at least two banks for comparative analysis

    bank1, bank2 = banks[0], banks[1]

    for current_bank in banks:
        current_suggestions = []
        rival_bank = bank2 if current_bank == bank1 else bank1
        
        # Filter data for the current bank
        current_data = comparative_df[comparative_df['bank'] == current_bank].copy()
        rival_data = comparative_df[comparative_df['bank'] == rival_bank].copy()

        # --- 1. Identify Largest Internal Pain Point (Highest Negative %) ---
        if not current_data.empty:
            pain_point_theme = current_data.loc[current_data['NEGATIVE %'].idxmax()]
            theme_name = pain_point_theme['theme']
            neg_pct = pain_point_theme['NEGATIVE %']
            
            suggestion = (
                f"**Critical Priority Fix:** Focus on the '{theme_name}' theme, which has the bank's highest negative sentiment ({neg_pct:.1f}%). "
                f"This suggests fundamental defects. Launch an immediate bug triage and user flow audit for all components related to this theme (e.g., login, transfer flows)."
            )
            current_suggestions.append(suggestion)

        # --- 2. Identify Largest Competitive Gap (Lagging Positive %) ---
        if not current_data.empty and not rival_data.empty:
            # Merge data for direct comparison
            merged_df = pd.merge(
                current_data[['theme', 'POSITIVE %', 'NEGATIVE %']],
                rival_data[['theme', 'POSITIVE %', 'NEGATIVE %']],
                on='theme',
                suffixes=('_current', '_rival')
            )
            
            # Calculate competitive gap (Current Positive % - Rival Positive %)
            merged_df['Positive Gap'] = merged_df['POSITIVE %_current'] - merged_df['POSITIVE %_rival']
            
            # Find the largest negative gap (where the current bank performs worst relative to rival)
            if not merged_df.empty:
                largest_lag = merged_df.loc[merged_df['Positive Gap'].idxmin()]
                lag_theme = largest_lag['theme']
                gap = largest_lag['Positive Gap']
                rival_pos = largest_lag['POSITIVE %_rival']
                
                # Only suggest if the gap is significant (e.g., more than 5%)
                if gap < -5:
                    suggestion = (
                        f"**Competitive Focus:** The bank significantly lags in '{lag_theme}' (Positive Gap: {gap:.1f}%). "
                        f"The rival {rival_bank} achieves {rival_pos:.1f}% positive sentiment here. "
                        f"Conduct a competitive teardown of {rival_bank}'s specific implementation for this feature/service to rapidly close the performance gap."
                    )
                    current_suggestions.append(suggestion)
                elif len(current_suggestions) < 2:
                     # Fallback suggestion if the competitive gap isn't big enough for the second point
                    suggestion = (
                        f"**Feature Enhancement:** Despite a high internal rating, the bank is losing ground in 'Feature Requests & Missing Functionality'. "
                        f"Prioritize implementing the top-requested features to maintain a competitive edge and reduce the long-term negative drift in this area."
                    )
                    current_suggestions.append(suggestion)


        suggestions[current_bank] = current_suggestions
        
    return suggestions
# ---Visualization Functions

def plot_rating_distribution(df: pd.DataFrame):
    """
    Plots the count of each star rating (1-5) for all banks.
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate counts of each rating for each bank
    rating_counts = df.groupby(['bank', 'rating']).size().reset_index(name='count')
    
    # Define bar width and positions
    banks = df['bank'].unique()
    n_banks = len(banks)
    ratings = sorted(df['rating'].unique())
    n_ratings = len(ratings)
    
    bar_width = 0.8 / n_banks
    r = np.arange(n_ratings)
    
    colors = plt.cm.get_cmap('viridis', n_banks) 

    for i, bank in enumerate(banks):
        bank_data = rating_counts[rating_counts['bank'] == bank]
        
        # Ensure all ratings are present for consistent plotting
        plot_data = bank_data.set_index('rating').reindex(ratings, fill_value=0)['count']
        
        # Position for the bar group
        x_pos = r + i * bar_width - (n_banks - 1) * bar_width / 2
        
        plt.bar(x_pos, plot_data, color=colors(i), width=bar_width, edgecolor='grey', label=bank)

    plt.title('Comparative Star Rating Distribution Across Banks', fontsize=16)
    plt.xlabel('Star Rating (1 = Worst, 5 = Best)', fontsize=12)
    plt.ylabel('Number of Reviews', fontsize=12)
    plt.xticks(r, ratings)
    plt.legend(title='Bank', loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    print("Plot 1: Rating Distribution displayed.")


def plot_sentiment_trends(df: pd.DataFrame):
    """
    Plots the rolling mean of the compound sentiment score over time (monthly).
    """
    plt.figure(figsize=(12, 7))
    
    # Ensure 'date' is the index for time-series operations
    df_ts = df[['date', 'bank', 'compound_score']].set_index('date').sort_index()

    for bank in df_ts['bank'].unique():
        bank_data = df_ts[df_ts['bank'] == bank]
        
        # Resample data monthly and calculate the mean compound score
        monthly_sentiment = bank_data['compound_score'].resample('M').mean().dropna()
        
        if not monthly_sentiment.empty:
            plt.plot(
                monthly_sentiment.index, 
                monthly_sentiment.values, 
                marker='o', 
                linestyle='-', 
                label=bank
            )

    plt.title('Monthly Average Sentiment Trend (VADER Compound Score)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Average Compound Sentiment Score (Higher is Better)', fontsize=12)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Neutral Threshold (0.0)')
    plt.legend(title='Bank')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()
    print("Plot 2: Sentiment Trend over time displayed.")
    

def plot_top_keywords(keywords_df: pd.DataFrame):
    """
    Plots the top N high-scoring keywords/themes from the TF-IDF analysis.
    """
    if keywords_df.empty:
        print("Cannot plot keywords: keywords_df is empty.")
        return
        
    # Filter out 'Other/General' and take the top 15 highest scoring keywords
    plot_data = keywords_df[keywords_df['assigned_theme'] != 'Other/General'].sort_values(by='tfidf_score', ascending=False).head(15)
    
    plt.figure(figsize=(12, 6))
    
    # Use different colors based on the assigned theme
    themes = plot_data['assigned_theme'].unique()
    theme_colors = plt.cm.get_cmap('Set3', len(themes))
    color_map = {theme: theme_colors(i) for i, theme in enumerate(themes)}
    
    colors = [color_map[theme] for theme in plot_data['assigned_theme']]
    
    plt.barh(plot_data['keyword_phrase'], plot_data['tfidf_score'], color=colors)
    
    # Create custom legend for themes
    handles = [plt.Rectangle((0,0),1,1, color=color_map[theme]) for theme in themes]
    plt.legend(handles, themes, title="Assigned Theme", loc='lower right')

    plt.title('Top 15 Most Dominant Keywords and Themes (by TF-IDF Score)', fontsize=16)
    plt.xlabel('TF-IDF Score (Importance/Frequency)', fontsize=12)
    plt.ylabel('Keyword / N-gram', fontsize=12)
    plt.gca().invert_yaxis() # Highest score at the top
    plt.tight_layout()
    plt.show()
    print("Plot 3: Top Keywords/Themes displayed.")
   