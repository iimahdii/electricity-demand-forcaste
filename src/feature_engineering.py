import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate time-based, weather, lag, rolling, and interaction features.
    Returns:
        pd.DataFrame with engineered features, dropping NaN rows introduced by lags.
    """
    df = df.copy()
    
    # --- 1. Time-based features ---
    df['hour'] = df.index.hour
    df['dow'] = df.index.dayofweek
    df['month'] = df.index.month
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    
    holiday_dates = set()
    for year in range(df.index.year.min(), df.index.year.max() + 1):
        holiday_dates.update([f'{year}-01-01', f'{year}-07-01', f'{year}-12-25', f'{year}-12-26'])
    df['is_holiday'] = df.index.strftime('%Y-%m-%d').isin(holiday_dates).astype(int)
    
    # --- 2. Weather-derived features ---
    COMFORT_TEMP = 18.0
    df['HDD'] = np.maximum(0, COMFORT_TEMP - df['Temperature'])
    df['CDD'] = np.maximum(0, df['Temperature'] - COMFORT_TEMP)
    df['temp_squared'] = df['Temperature'] ** 2
    df['wind_chill_effect'] = df['Wind_Speed'] * df['HDD']
    
    # --- 3. Lag features (Must be >= 24h to avoid leakage) ---
    for lag in [24, 48, 72, 168]:
        df[f'demand_lag_{lag}'] = df['Ontario_Demand'].shift(lag)
    df['price_lag_24'] = df['HOEP'].shift(24)
    df['price_lag_48'] = df['HOEP'].shift(48)
    
    # --- 4. Rolling features ---
    df['demand_rolling_mean_24'] = df['Ontario_Demand'].shift(24).rolling(24, min_periods=24).mean()
    df['demand_rolling_std_24'] = df['Ontario_Demand'].shift(24).rolling(24, min_periods=24).std()
    df['demand_rolling_mean_168'] = df['Ontario_Demand'].shift(24).rolling(168, min_periods=168).mean()
    df['demand_rolling_min_24'] = df['Ontario_Demand'].shift(24).rolling(24, min_periods=24).min()
    df['demand_rolling_max_24'] = df['Ontario_Demand'].shift(24).rolling(24, min_periods=24).max()
    df['demand_diff_24'] = df['Ontario_Demand'].shift(24) - df['Ontario_Demand'].shift(48)
    
    # --- 5. Interactions ---
    df['hour_x_weekend'] = df['hour_sin'] * df['is_weekend']
    df['cdd_x_hour'] = df['CDD'] * df['hour_sin']
    
    # Cleanup
    df = df.drop(columns=['hour', 'dow', 'month'])
    return df.dropna()
