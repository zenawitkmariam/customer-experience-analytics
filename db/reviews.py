import pandas as pd
from sqlalchemy import text
from db.connection import engine

def insert_reviews_from_df(df: pd.DataFrame):
    """
    Insert reviews from a DataFrame into the 'reviews' table.
    
    Columns expected in df:
        review, rating, date, bank, source, normalized_review, compound_score, sentiment_label
    """
    with engine.begin() as conn:  # begin transaction
        for _, row in df.iterrows():
            # Get bank_id from banks table
            bank_query = text("SELECT bank_id FROM banks WHERE bank_name = :bank_name LIMIT 1")
            result = conn.execute(bank_query, {"bank_name": row['bank']}).fetchone()
            
            if result is None:
                print(f"Bank not found: {row['bank']}, skipping this review.")
                continue
            
            bank_id = result[0]
            
            # Insert review
            insert_query = text("""
                INSERT INTO reviews
                (bank_id, review_text, rating, review_date, sentiment_label, sentiment_score, source)
                VALUES
                (:bank_id, :review_text, :rating, :review_date, :sentiment_label, :sentiment_score, :source)
            """)
            
            conn.execute(insert_query, {
                "bank_id": bank_id,
                "review_text": row['review'],
                "rating": row['rating'],
                "review_date": row['date'],
                "sentiment_label": row['sentiment_label'],
                "sentiment_score": row['compound_score'],
                "source": row['source']
            })
        
        print("All reviews inserted successfully.")