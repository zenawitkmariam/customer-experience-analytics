import pandas as pd
import os

def save_dataframe(df: pd.DataFrame, path: str):
    """
    Save any pandas DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save.
    path : str
        Full file path (including filename.csv).
    """
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df.to_csv(path, index=False, encoding='utf-8')
    print(f"DataFrame saved successfully to: {path}")