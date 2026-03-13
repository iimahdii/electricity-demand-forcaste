import os
import sys

# Ensure the root directory is on the path so we can import src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import DATA_PATH, TRAIN_END, TEST_START
from src.pipeline import ElectricityDemandPipeline


def main():
    """Main entry point to execute the electricity demand forecasting pipeline."""
    pipeline = ElectricityDemandPipeline(
        data_path=DATA_PATH,
        train_end=TRAIN_END,
        test_start=TEST_START
    )
    pipeline.run()


if __name__ == "__main__":
    main()
