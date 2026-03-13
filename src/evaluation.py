import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.config import FIGURES_DIR

def evaluate_model(name: str, y_true, y_pred, results_dict: dict):
    """Compute metrics and store in results dictionary."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    results_dict[name] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    print(f"  {name:25s} | MAE: {mae:8.1f} MW | RMSE: {rmse:8.1f} MW | MAPE: {mape:.2f}%")
    return mae, rmse, mape

def plot_actual_vs_predicted(y_true, y_pred, model_name, filename="08_actual_vs_predicted.png"):
    """Plot actual vs predicted for a 7-day test window."""
    sample_start = pd.Timestamp("2020-08-15")
    sample_end = sample_start + pd.Timedelta(hours=7*24-1)
    mask = (y_true.index >= sample_start) & (y_true.index <= sample_end)
    
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(y_true.index[mask], y_true.values[mask], label='Actual', color='navy', linewidth=2)
    ax.plot(y_true.index[mask], y_pred[mask], label=f'Predicted ({model_name})', color='coral', linestyle='--')
    ax.set_title('7-Day Forecast: Actual vs Predicted', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=150)
    plt.close()

def plot_residuals(y_true, y_pred, filename="10_residuals.png"):
    """Plot error residuals."""
    residuals = y_true.values - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(residuals, bins=50, color='steelblue', edgecolor='navy')
    axes[0].set_title('Residual Distribution', fontweight='bold')
    axes[0].axvline(x=0, color='red', linestyle='--')
    
    axes[1].scatter(y_pred, residuals, alpha=0.1, s=5, color='steelblue')
    axes[1].set_title('Residuals vs Predicted')
    axes[1].axhline(y=0, color='red', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=150)
    plt.close()

def plot_24h_forecast(y_true, y_pred, model_name, filename="11_24h_forecast_demo.png"):
    """Generate a 24-hour demand forecast for a sample day in July or August."""
    # Let's pick August 10th, 2020 as the sample day
    sample_day = pd.Timestamp("2020-08-10")
    end_of_day = sample_day + pd.Timedelta(hours=23)
    
    mask = (y_true.index >= sample_day) & (y_true.index <= end_of_day)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(y_true.index[mask], y_true.values[mask], label='Actual Demand', color='navy', marker='o', linewidth=2)
    ax.plot(y_true.index[mask], y_pred[mask], label=f'Predicted 24h Forecast ({model_name})', color='coral', marker='x', linestyle='--', linewidth=2)
    
    ax.set_title(f'24-Hour Demand Forecast Demo - {sample_day.strftime("%B %d, %Y")}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (Hours)', fontsize=12)
    ax.set_ylabel('Electricity Demand (MW)', fontsize=12)
    
    # Format x-axis to show hours clearly
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)
    
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=150)
    
    print(f"\n--- 24-Hour Demo Forecast ({sample_day.strftime('%Y-%m-%d')}) ---")
    demo_results = pd.DataFrame({
        'Time': y_true.index[mask].strftime('%H:%M'),
        'Actual (MW)': y_true.values[mask].round(1),
        'Predicted (MW)': y_pred[mask].round(1),
        'Error (%)': (np.abs(y_true.values[mask] - y_pred[mask]) / y_true.values[mask] * 100).round(2)
    }).set_index('Time')
    print(demo_results)
    print("-" * 60)

    plt.close()
