import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
import optuna
from src.config import RANDOM_STATE

optuna.logging.set_verbosity(optuna.logging.WARNING)

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
    
    # Fix Leakage: Create an internal validation set from train data for early stopping
    val_size = int(len(X_train) * 0.1)
    X_tr, X_val = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
    y_tr, y_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]
    
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.log_evaluation(0)])
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
    
    # Fix Leakage: Create an internal validation set from train data for early stopping
    val_size = int(len(X_train) * 0.1)
    X_tr, X_val = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
    y_tr, y_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]
    
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model, model.predict(X_test)

def tune_lightgbm_optuna(X_train, y_train, n_trials=30):
    """Use Optuna with TimeSeriesSplit CV to find optimal LightGBM hyperparameters."""
    tscv = TimeSeriesSplit(n_splits=3)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'random_state': RANDOM_STATE, 'verbose': -1, 'n_jobs': -1
        }
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.log_evaluation(0)])
            scores.append(mean_absolute_error(y_val, model.predict(X_val)))
        return np.mean(scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    best_params.update({'random_state': RANDOM_STATE, 'verbose': -1, 'n_jobs': -1})
    print(f"  Optuna best MAE (CV): {study.best_value:.1f} MW  |  Trials: {n_trials}")
    return best_params

def tune_xgboost_optuna(X_train, y_train, n_trials=30):
    """Use Optuna with TimeSeriesSplit CV to find optimal XGBoost hyperparameters."""
    tscv = TimeSeriesSplit(n_splits=3)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'random_state': RANDOM_STATE, 'verbosity': 0, 'n_jobs': -1
        }
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            scores.append(mean_absolute_error(y_val, model.predict(X_val)))
        return np.mean(scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    best_params.update({'random_state': RANDOM_STATE, 'verbosity': 0, 'n_jobs': -1})
    print(f"  Optuna best MAE (CV): {study.best_value:.1f} MW  |  Trials: {n_trials}")
    return best_params

def build_stacking_ensemble(X_train, y_train, X_test, best_lgb_params, best_xgb_params):
    """
    Build a stacking ensemble using Ridge regression as meta-learner.
    Uses proper TimeSeriesSplit to generate unbiased out-of-fold predictions
    for the meta-learner, completely avoiding in-sample overfitting.
    """
    from sklearn.ensemble import StackingRegressor
    
    estimators = [
        ('lgb', lgb.LGBMRegressor(**best_lgb_params)),
        ('xgb', xgb.XGBRegressor(**best_xgb_params))
    ]
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=tscv,
        n_jobs=-1
    )
    
    stacking_model.fit(X_train, y_train)
    
    print("\n  Stacking Meta-Learner Weights:")
    for name, weight in zip([e[0] for e in estimators], stacking_model.final_estimator_.coef_):
        print(f"    {name}: {weight:.3f}")
        
    return stacking_model, stacking_model.predict(X_test)

def train_lstm(df_clean, target_col, feature_cols, train_end_date, lookback=168):
    """Train an LSTM model using Keras/JAX."""
    import os
    os.environ['KERAS_BACKEND'] = 'jax'
    try:
        from keras import layers, models, callbacks
        from sklearn.preprocessing import StandardScaler
        
        # Fix Leakage: Ensure scalers are strictly fitted on the Training subset
        train_df = df_clean.loc[:train_end_date]
        
        scaler_X, scaler_y = StandardScaler(), StandardScaler()
        # Fit ONLY on the isolated training subset
        scaler_X.fit(train_df[feature_cols].values)
        scaler_y.fit(train_df[target_col].values.reshape(-1, 1))
        
        # Transform the full dataset based on training stats
        X_all = scaler_X.transform(df_clean[feature_cols].values)
        y_all = scaler_y.transform(df_clean[target_col].values.reshape(-1, 1)).flatten()
        
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

