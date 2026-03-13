import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from src.config import RANDOM_STATE

def train_baselines(X_train, y_train, X_test):
    """Train Naive, Ridge, and Random Forest models."""
    naive_pred = X_test['demand_lag_24'].values
    
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    
    return {
        "Naive (Yesterday)": naive_pred,
        "Ridge Regression": ridge_pred,
        "Random Forest": rf_pred
    }

def train_lightgbm(X_train, y_train, X_test, y_test, params=None):
    """Train LightGBM model."""
    if params is None:
        params = {
            'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 8,
            'num_leaves': 63, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'random_state': RANDOM_STATE, 'verbose': -1, 'n_jobs': -1
        }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.log_evaluation(0)])
    return model, model.predict(X_test)

def train_xgboost(X_train, y_train, X_test, y_test, params=None):
    """Train XGBoost model."""
    if params is None:
        params = {
            'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 8,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'random_state': RANDOM_STATE, 'verbosity': 0, 'n_jobs': -1
        }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return model, model.predict(X_test)

def train_lstm(df_clean, target_col, feature_cols, train_end_date, lookback=168):
    """Train an LSTM model using Keras/JAX."""
    import os
    os.environ['KERAS_BACKEND'] = 'jax'
    try:
        from keras import layers, models, callbacks
        from sklearn.preprocessing import StandardScaler
        
        scaler_X, scaler_y = StandardScaler(), StandardScaler()
        X_all = scaler_X.fit_transform(df_clean[feature_cols].values)
        y_all = scaler_y.fit_transform(df_clean[target_col].values.reshape(-1, 1)).flatten()
        
        # Create sequences
        Xs, ys = [], []
        for i in range(lookback, len(X_all)):
            Xs.append(X_all[i - lookback:i])
            ys.append(y_all[i])
        X_seq, y_seq = np.array(Xs), np.array(ys)
        
        # Split index
        train_end_idx = df_clean.index.get_loc(df_clean.loc[:train_end_date].index[-1]) + 1
        split_idx = train_end_idx - lookback
        
        X_train_seq, y_train_seq = X_seq[:split_idx], y_seq[:split_idx]
        X_test_seq, y_test_seq = X_seq[split_idx:], y_seq[split_idx:]
        
        model = models.Sequential([
            layers.Input(shape=(lookback, X_train_seq.shape[2])),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        model.fit(X_train_seq, y_train_seq, validation_split=0.1, epochs=30, batch_size=64, 
                  callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5)], verbose=0)
        
        lstm_pred_scaled = model.predict(X_test_seq, verbose=0).flatten()
        lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).flatten()
        return model, lstm_pred
    except Exception as e:
        print(f"LSTM failed: {e}")
        return None, None
