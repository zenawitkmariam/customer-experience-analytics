import time
import pandas as pd
from tqdm import tqdm
from google_play_scraper import reviews_all,reviews, Sort

# --- The Scraping Function ---

def scrape_bank_reviews(app_ids: dict, max_reviews_per_bank: int = 400):
    """
    Scrapes reviews, ratings, dates, and app names for banking apps from Google Play Store.

    Args:
        app_ids (dict): A dictionary where keys are bank names (e.g., 'CBE') and 
                        values are their corresponding Google Play package IDs.
        max_reviews_per_bank (int): The target minimum number of reviews to collect 
                                    for each bank (default 400).

    Returns:
        pd.DataFrame: A DataFrame containing all collected review data.
    """
    
    all_reviews_list = []
    
    # Use tqdm to show overall progress through the list of banks
    for bank_name, app_id in tqdm(app_ids.items(), desc="Total Scraping Progress"):
        print(f"\n--- Starting scrape for {bank_name} (ID: {app_id}) ---")
        
        try:
           # FIX: Using 'reviews' function instead of 'reviews_all' to strictly respect the 'count' limit.
            # reviews returns a tuple: (reviews_list, continuation_token)
            result, _ = reviews(
                app_id,
                lang='en', 
                country='us',
                sort=Sort.NEWEST, # Use NEWEST to get the most recent data
                count=max_reviews_per_bank, # Specify the target review count (limit)
                filter_score_with=None # Get reviews regardless of score
            )
            
            scraped_reviews = result # The actual list of reviews is the first element
            
            print(f"Successfully scraped {len(scraped_reviews)} reviews for {bank_name}.")

            # Process the results into a standardized format
            for review in scraped_reviews:
                all_reviews_list.append({
                    'bank_name': bank_name,
                    'app_id': app_id,
                    'review_id': review.get('reviewId'),
                    'user_name': review.get('userName'),
                    # RATING: The 'score' field provides the rating (1 to 5)
                    'rating': review.get('score'), 
                    'content': review.get('content'),
                    # DATE: The 'at' field provides the date
                    'date': review.get('at').strftime('%Y-%m-%d %H:%M:%S'), 
                    'reply_content': review.get('replyContent'),
                    'replied_at': review.get('repliedAt').strftime('%Y-%m-%d %H:%M:%S') if review.get('repliedAt') else None,
                    'thumbs_up': review.get('thumbsUpCount')
                })
                
        except Exception as e:
            print(f"An error occurred while scraping {bank_name}: {e}")
            
        # Introduce a sleep time (5 seconds) to be polite to the Google Play Store servers
        time.sleep(5) 
        
    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(all_reviews_list)
    return df