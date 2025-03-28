import pandas as pd
import os
from src.utils.environment_loading import INTERIM_DATA_PATH, PROCESSED_DATA_PATH

# Add features to the data
def add_features(data):
    data['50_day_MA'] = data['Close'].rolling(window=50).mean()
    data['200_day_MA'] = data['Close'].rolling(window=200).mean()
    data['Momentum'] = data['Close'].pct_change(periods=20)
    return data

def generate_features(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    for file_name in os.listdir(input_path):
        if file_name.endswith(".csv"):
            data = pd.read_csv(os.path.join(input_path, file_name))
            data = add_features(data)
            output_file = os.path.join(output_path, file_name)
            data.to_csv(output_file, index=False)
            print(f"Features added and data saved to {output_file}")

def main():
    generate_features(INTERIM_DATA_PATH, PROCESSED_DATA_PATH)

if __name__ == "__main__":
    main()