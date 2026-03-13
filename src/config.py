"""
Centralized configuration settings for the forecasting pipeline.
"""
import os

# Paths
DATA_PATH = "Sample Dataset.csv"
FIGURES_DIR = "figures"

# Model Parameters
TRAIN_END = "2020-06-30"      # Training ends here
TEST_START = "2020-07-01"     # Test begins here
FORECAST_HORIZON = 24         # Hours to predict
RANDOM_STATE = 42

# Ensure figures directory exists
os.makedirs(FIGURES_DIR, exist_ok=True)

# Helper for printing sections
def print_section(title: str):
    """Pretty-print section headers."""
    width = 70
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}\n")
