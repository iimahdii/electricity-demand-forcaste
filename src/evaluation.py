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

def plot_24h_forecast(y_true, y_pred, model_name="Best Model", filename="11_24h_forecast_demo.png"):
    """Plot an interactive 24-hour forecast for the best sample day in July or August."""
    import matplotlib.dates as mdates
    
    # Create DataFrame with true and predicted
    df_eval = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
    
    # Filter for July and August
    df_summer = df_eval[(df_eval.index.month == 7) | (df_eval.index.month == 8)].copy()
    
    if df_summer.empty:
        print("No July or August data found in the test set for 24h plot.")
        return
        
    # Find the day with the lowest Mean Absolute Error
    df_summer['Date'] = df_summer.index.date
    daily_errors = {}
    
    for date, group in df_summer.groupby('Date'):
        if len(group) == 24: # Only consider full 24-hour days
            mae = np.mean(np.abs(group['Actual'] - group['Predicted']))
            daily_errors[date] = mae
            
    if not daily_errors:
         print("No full 24-hour days found in July/August.")
         return
         
    best_date = min(daily_errors, key=daily_errors.get)
    best_day_data = df_summer[df_summer['Date'] == best_date].copy()
    
    # Print tabular results
    print(f"\n--- Best 24-Hour Forecast Found: {best_date} (Model: {model_name}) ---")
    best_day_data['Error (%)'] = np.abs(best_day_data['Actual'] - best_day_data['Predicted']) / best_day_data['Actual'] * 100
    print(best_day_data[['Actual', 'Predicted', 'Error (%)']].round(2))
    print("-" * 60)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(best_day_data.index, best_day_data['Actual'], label='Actual Demand', color='black', linewidth=2, marker='o')
    
    # Get just the date string for the title
    date_str = pd.Timestamp(best_date).strftime("%B %d, %Y")
    
    ax.plot(best_day_data.index, best_day_data['Predicted'], label=f'Predicted ({model_name})', color='darkorange', linewidth=2, marker='s', linestyle='--')
    
    ax.fill_between(best_day_data.index, best_day_data['Actual'], best_day_data['Predicted'], color='orange', alpha=0.15)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.title(f"24-Hour Electricity Demand Forecast Demo - {date_str}\nModel: {model_name} (Lowest Error Day in Summer)", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Demand (MW)", fontsize=12)
    plt.xlabel("Hour of Day", fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(frameon=True, fontsize=11, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=150)
    plt.close()
