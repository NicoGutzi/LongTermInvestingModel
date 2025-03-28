import pandas as pd
import os
from src.utils.environment_loading import INTERIM_DATA_PATH, RAW_DATA_PATH

# Function to clean raw data
def clean_data(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    for file_name in os.listdir(input_path):
        if file_name.endswith(".csv"):
            data = pd.read_csv(os.path.join(input_path, file_name))
            # Example cleaning: forward-fill missing data
            data.fillna(method="ffill", inplace=True)
            output_file = os.path.join(output_path, file_name)
            data.to_csv(output_file, index=False)
            print(f"Cleaned data saved to {output_file}")

def main():
    clean_data(RAW_DATA_PATH, INTERIM_DATA_PATH)

if __name__ == "__main__":
    main()
