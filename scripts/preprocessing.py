
import pandas as pd
from config import DATA_PATHS
# -- Data Preprocessing Function ---

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and structures the raw DataFrame for analysis.
    
    Operations:
    1. Removes duplicates based on 'review_id'.
    2. Handles missing data (removes rows where 'content' or 'rating' is missing).
    3. Normalizes 'date' to YYYY-MM-DD format.
    4. Selects and renames columns to the desired output structure.

    Args:
        df (pd.DataFrame): The raw DataFrame from scrape_bank_reviews.

    Returns:
        pd.DataFrame: The cleaned and structured DataFrame.
    """
    print("\n--- Starting Data Preprocessing ---")
    initial_rows = len(df)
    
    # 1. Remove duplicates based on the unique review ID
    df = df.drop_duplicates(subset=['review_id'], keep='first')
    print(f"Removed duplicates. Rows remaining: {len(df)}")
    
    # 2. Handle missing data (Reviews must have content and a score)
    # Filling NaN 'content' with empty string for robustness before dropping based on rating
    df['content'] = df['content'].fillna('')
    # Drop rows where rating is missing (although unlikely from Google Play Scraper)
    df = df.dropna(subset=['rating'])
    print(f"Removed rows with missing rating. Rows remaining: {len(df)}")
    
    # 3. Normalize dates to YYYY-MM-DD format
    # The 'date' column is already a string in 'YYYY-MM-DD HH:MM:SS' format from scraping.
    # Convert it to datetime and then format to YYYY-MM-DD.
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    print("Dates normalized to YYYY-MM-DD.")
    
    # 4. Select and rename columns to the required output format
    final_df = df.rename(columns={
        'content': 'review',
        'bank_name': 'bank'
    })
    
    # Select the final, required columns in order
    final_df = final_df[['review', 'rating', 'date', 'bank', 'app_id']]
    
    # Rename 'app_id' to 'source' as requested (using app_id as the source identifier)
    final_df = final_df.rename(columns={'app_id': 'source'})
    final_rows = len(final_df)
    print(f"Preprocessing complete. Total rows cleaned: {initial_rows - final_rows}")
    
    return final_df
