import os
import warnings
import pandas as pd
import optuna

from src.config import print_section
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_features
from src.models import (
    train_baselines, train_lightgbm, train_xgboost, train_lstm
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
        """Trains baselines and advanced models, and builds an ensemble."""
        print_section("4. MODELING & EVALUATION")

        # Baselines
        print("Training Baselines...")
        baseline_preds = train_baselines(self.X_train, self.y_train, self.X_test)
        for name, pred in baseline_preds.items():
            evaluate_model(name, self.y_test, pred, self.results)

        # LightGBM
        print("\nTraining LightGBM...")
        _, lgb_pred = train_lightgbm(self.X_train, self.y_train, self.X_test, self.y_test)
        evaluate_model("LightGBM", self.y_test, lgb_pred, self.results)

        # XGBoost
        print("Training XGBoost...")
        _, xgb_pred = train_xgboost(self.X_train, self.y_train, self.X_test, self.y_test)
        evaluate_model("XGBoost", self.y_test, xgb_pred, self.results)

        # LSTM
        print("Training LSTM...")
        _, lstm_pred = train_lstm(self.df_features, self.TARGET, self.feature_cols, self.train_end)
        if lstm_pred is not None:
            evaluate_model("LSTM", self.y_test, lstm_pred, self.results)

        # Ensemble
        if lstm_pred is not None and len(lstm_pred) == len(lgb_pred):
            print("Building Ensemble (LGB+XGB+LSTM)...")
            ensemble_pred = 0.5 * lgb_pred + 0.3 * xgb_pred + 0.2 * lstm_pred
            self.best_model_name = "Ensemble (LGB+XGB+LSTM)"
        else:
            print("Building Ensemble (LGB+XGB)...")
            ensemble_pred = 0.6 * lgb_pred + 0.4 * xgb_pred
            self.best_model_name = "Ensemble (LGB+XGB)"

        evaluate_model(self.best_model_name, self.y_test, ensemble_pred, self.results)
        self.best_pred = ensemble_pred

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
