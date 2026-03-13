import pandas as pd
import numpy as np

def load_and_clean_data(data_path: str) -> pd.DataFrame:
    """
    Load raw CSV data, set a continuous hourly index, and clean missing/duplicate values.
    """
    df = pd.read_csv(data_path)
    
    # Create proper datetime index
    df['datetime'] = pd.to_datetime(df['Date']) + pd.to_timedelta(df['Hour'], unit='h')
    
    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset='datetime', keep='first')
    
    # Set index and drop redundant columns
    df = df.set_index('datetime').sort_index()
    df = df.drop(columns=['Date', 'Weekday', 'Hour'])
    
    # Reindex to fill 4,440 missing hours
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq='h')
    df = df.reindex(full_idx)
    df.index.name = 'datetime'
    
    # Interpolate missing values (linear up to 48h, fallback to ffill)
    df = df.interpolate(method='linear', limit=48)
    df = df.ffill(limit=24).bfill(limit=24)
    
    # Drop any remaining nulls
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
        
    return df
