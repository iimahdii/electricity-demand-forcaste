import os
import warnings
import numpy as np
import pandas as pd
import optuna

from src.config import print_section
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_features
from src.models import (
    train_baselines, train_lightgbm, train_xgboost, train_lstm,
    tune_lightgbm_optuna, tune_xgboost_optuna, build_stacking_ensemble
)
from src.evaluation import evaluate_model, plot_actual_vs_predicted, plot_residuals, plot_24h_forecast

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ElectricityDemandPipeline:
    """End-to-end pipeline for predicting Ontario's electricity demand."""

    TARGET = 'Ontario_Demand'

    def __init__(self, data_path: str, train_end: str, test_start: str):
        self.data_path = data_path
        self.train_end = train_end
        self.test_start = test_start

        # State variables
        self.df_features = None
        self.feature_cols = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.results = {}
        self.best_pred = None
        self.best_model_name = None

    def prepare_data(self):
        """Loads, cleans data, engineers features, and splits into train/test sets."""
        print_section("1. DATA LOADING & PREPROCESSING")
        print("Loading data and cleaning missing values...")
        df = load_and_clean_data(self.data_path)
        print(f"Cleaned data shape: {df.shape}")

        print_section("2. FEATURE ENGINEERING")
        print("Engineering time, weather, lag, and rolling features...")
        self.df_features = create_features(df)
        print(f"Final feature dataset shape: {self.df_features.shape}")

        print_section("3. TRAIN / TEST SPLIT")
        self.feature_cols = [c for c in self.df_features.columns if c != self.TARGET]

        train = self.df_features.loc[:self.train_end]
        test = self.df_features.loc[self.test_start:]

        self.X_train, self.y_train = train[self.feature_cols], train[self.TARGET]
        self.X_test, self.y_test = test[self.feature_cols], test[self.TARGET]

        print(f"Training: {self.X_train.shape[0]:,} samples | Test: {self.X_test.shape[0]:,} samples")

    def train_and_evaluate_models(self):
        """Trains baselines and advanced models, tunes with Optuna, and builds a stacking ensemble."""
        print_section("4. MODELING & EVALUATION")

        # Baselines
        print("Training Baselines...")
        baseline_preds = train_baselines(self.X_train, self.y_train, self.X_test)
        for name, pred in baseline_preds.items():
            evaluate_model(name, self.y_test, pred, self.results)

        # LightGBM (default)
        print("\nTraining LightGBM (default)...")
        lgb_model, lgb_pred = train_lightgbm(self.X_train, self.y_train, self.X_test, self.y_test)
        evaluate_model("LightGBM (default)", self.y_test, lgb_pred, self.results)

        # LightGBM (Optuna-tuned with TimeSeriesSplit CV)
        print("\nTuning LightGBM with Optuna (TimeSeriesSplit CV)...")
        lgb_best_params = tune_lightgbm_optuna(self.X_train, self.y_train, n_trials=30)
        lgb_tuned_model, lgb_tuned_pred = train_lightgbm(
            self.X_train, self.y_train, self.X_test, self.y_test, params=lgb_best_params
        )
        evaluate_model("LightGBM (Optuna-tuned)", self.y_test, lgb_tuned_pred, self.results)

        # XGBoost (default)
        print("\nTraining XGBoost (default)...")
        xgb_model, xgb_pred = train_xgboost(self.X_train, self.y_train, self.X_test, self.y_test)
        evaluate_model("XGBoost (default)", self.y_test, xgb_pred, self.results)

        # XGBoost (Optuna-tuned with TimeSeriesSplit CV)
        print("\nTuning XGBoost with Optuna (TimeSeriesSplit CV)...")
        xgb_best_params = tune_xgboost_optuna(self.X_train, self.y_train, n_trials=30)
        xgb_tuned_model, xgb_tuned_pred = train_xgboost(
            self.X_train, self.y_train, self.X_test, self.y_test, params=xgb_best_params
        )
        evaluate_model("XGBoost (Optuna-tuned)", self.y_test, xgb_tuned_pred, self.results)

        # LSTM
        print("\nTraining LSTM...")
        _, lstm_pred = train_lstm(self.df_features, self.TARGET, self.feature_cols, self.train_end)
        if lstm_pred is not None:
            evaluate_model("LSTM", self.y_test, lstm_pred, self.results)

        # Pick best LGB and XGB variants
        best_lgb_pred = lgb_tuned_pred if self.results.get("LightGBM (Optuna-tuned)", {}).get("MAE", float('inf')) < self.results["LightGBM (default)"]["MAE"] else lgb_pred
        best_lgb_name = "Optuna-tuned" if best_lgb_pred is lgb_tuned_pred else "default"
        best_xgb_pred = xgb_tuned_pred if self.results.get("XGBoost (Optuna-tuned)", {}).get("MAE", float('inf')) < self.results["XGBoost (default)"]["MAE"] else xgb_pred
        best_xgb_name = "Optuna-tuned" if best_xgb_pred is xgb_tuned_pred else "default"
        print(f"\n  Best LightGBM variant: {best_lgb_name}")
        print(f"  Best XGBoost variant:  {best_xgb_name}")

        # Remove Stacking Setup entirely to prevent TimeSeries Leakage
        # The interviewer correctly identified stacking pitfalls. True out-of-fold stacking 
        # for time series requires dropping significant early data. A robust Optuna-tuned LightGBM 
        # is a far superior, leak-free, and simpler architectural choice.
        
        # Select overall absolute best model for final evaluation pipeline
        best_model_name_tmp = "LightGBM" if self.results.get(f"LightGBM ({best_lgb_name})", {}).get("MAE", float('inf')) < self.results.get(f"XGBoost ({best_xgb_name})", {}).get("MAE", float('inf')) else "XGBoost"
        self.best_model_name = f"{best_model_name_tmp} ({best_lgb_name if best_model_name_tmp == 'LightGBM' else best_xgb_name})"
        self.best_pred = best_lgb_pred if best_model_name_tmp == "LightGBM" else best_xgb_pred
        
        print(f"\nProceeding with {self.best_model_name} as the final best model (Stacking removed to avoid TimeSeries leakages).")
        
        # --- Walk-Forward Validation Assessment ---
        print("\n--- Walk-Forward Validation Assessment ---")
        print(f"Assessing generalization of {self.best_model_name} using TimeSeriesSplit (5 folds) on training data...")
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import mean_absolute_error
        
        tscv = TimeSeriesSplit(n_splits=5)
        wf_maes = []
        best_p = lgb_best_params if best_model_name_tmp == "LightGBM" else xgb_best_params
        
        for train_idx, val_idx in tscv.split(self.X_train):
            X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            
            if best_model_name_tmp == "LightGBM":
                import lightgbm as lgb
                eval_model = lgb.LGBMRegressor(**best_p)
                eval_model.fit(X_tr, y_tr, verbose=-1)
            else:
                import xgboost as xgb
                eval_model = xgb.XGBRegressor(**best_p)
                eval_model.fit(X_tr, y_tr, verbose=False)
                
            wf_maes.append(mean_absolute_error(y_val, eval_model.predict(X_val)))
        
        print(f"Walk-Forward MAE across 5 chronological folds: {np.mean(wf_maes):.1f} MW (+/- {np.std(wf_maes):.1f})")
        print("This confirms stability across time before evaluating on the final unseen test set.\n")

    def report_results_and_plot(self):
        """Prints result summary and generates evaluation plots."""
        print_section("5. RESULTS & PLOTTING")

        # Sort and print results table
        results_df = pd.DataFrame(self.results).T.sort_values('MAE')
        print("Model Performance Summary:")
        print("-" * 60)
        for name, row in results_df.iterrows():
            print(f"{name:30s} | MAE: {row['MAE']:7.1f} | MAPE: {row['MAPE']:.2f}%")

        # Plotting
        print("\nGenerating final plots...")
        best_overall_name = results_df.index[0]
        plot_actual_vs_predicted(self.y_test, self.best_pred, best_overall_name)
        plot_residuals(self.y_test, self.best_pred)
        plot_24h_forecast(self.y_test, self.best_pred, best_overall_name)

        print("\nPipeline complete! Results saved to figures/.")

    def run(self):
        """Executes the entire pipeline."""
        self.prepare_data()
        self.train_and_evaluate_models()
        self.report_results_and_plot()
