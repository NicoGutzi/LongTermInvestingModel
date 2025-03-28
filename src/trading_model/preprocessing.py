from typing import List
import pandas as pd
from sqlalchemy import text
from src.utils.database import engine  # engine from your database.py
from src.utils.environment_loading import load_config
from src.trading_model.feature_engineering import add_basic_features

def extract_price_data(index_id: str) -> pd.DataFrame:
    """
    Extracts daily price data for the specified index by joining the
    indices and raw_price_data tables in the database.
    """
    query = text("""
        SELECT r.date, r.open, r.high, r.low, r.close, r.volume
        FROM raw_price_data AS r
        WHERE r.index_id = :index_id
        ORDER BY r.date;
    """)
    df = pd.read_sql(query, engine, params={"index_id": index_id}, 
                       parse_dates=["date"], index_col="date")
    return df

def get_index_id(index_name: str) -> int:
    """
    Extract the indice id from the database. 
    """
    query = text("""
        SELECT index.id 
        FROM indices AS index
        WHERE index.name = :index_name
    """)
    df = pd.read_sql(query, engine, params={"index_name": index_name})
    return int(df['id'].iloc[0])

def extract_macro_data(country: str, metric: str) -> pd.DataFrame:
    """
    Extracts macroeconomic indicator data for a given country and metric from the database.
    """
    query = text("""
        SELECT date, value
        FROM macro_indicators
        WHERE country = :country AND metric = :metric
        ORDER BY date
    """)
    df = pd.read_sql(query, engine, params={"country": country, "metric": metric},
                       parse_dates=["date"], index_col="date")
    return df

def extract_all_macro_data(config: dict) -> dict:
    """
    Extracts all macroeconomic data based on the config.
    Returns a nested dictionary organized by category and country.
    """
    macro_data = {}

    # extract the different indicators from the config.yaml file
    macroeconomic_indicators = [key for key, _ in config["data_requests"].items() if key != "indices"]
    for category in macroeconomic_indicators:
        # request data for each indicator from db
        macro_data[category] = {}
        for country, _ in config["data_requests"][category].items():
            df = extract_macro_data(country, category)
            macro_data[category][country] = df
    return macro_data

def align_macro_data(macro_df: pd.DataFrame, price_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Resamples monthly macro data to daily frequency (using forward fill) and aligns it to the price data dates.
    """
    macro_daily = macro_df.resample('D').ffill()
    aligned = macro_daily.reindex(price_index, method='ffill')
    return aligned

def add_missingness_flags(df: pd.DataFrame, indicator_columns: List[str]) -> pd.DataFrame:
    """
    For each indicator column, adds a binary flag column indicating whether the value is missing.
    Then fills missing values with a placeholder (e.g., 0 or the mean of the column).
    """
    for col in indicator_columns:
        if df[col].isna().any():
            # Create a flag column where 1 means missing and 0 means present.
            flag_col = col + "_flag"
            df[flag_col] = df[col].isna().astype(int)
            # Option 1: Fill missing values with 0.
            df[col] = df[col].fillna(0)
    return df

    
def preprocess_index_data(index_name: str, config: dict, macro_data: dict) -> pd.DataFrame:
    """
    Processes data for a given index by extracting its daily price data, merging it with 
    macroeconomic data (which has been extracted once for all indices), and computing trading features.
    The macro data is merged into the price DataFrame and then basic features are computed.
    Weighted macro features are added as additional inputs.
    """
    indice_id = get_index_id(index_name)
    price_df = extract_price_data(indice_id)
    macro_features = {}
    indicator_cols = []

    for category in macro_data.keys():
        for country, macro_df in macro_data[category].items():
            if macro_df.empty:
                continue
            aligned_df = align_macro_data(macro_df, price_df.index)
            col_name = f"{category}_{country}"
            macro_features[col_name] = aligned_df["value"]
            indicator_cols.append(col_name)
    
    if macro_features:
        macro_df_combined = pd.concat(macro_features, axis=1)
        merged_df = price_df.join(macro_df_combined, how='left')
    else:
        merged_df = price_df.copy()

    # Add missingness flags for the macro indicators.
    merged_df = add_missingness_flags(merged_df, indicator_cols)
    
    # Compute basic features such as daily returns, moving averages, and volatility.
    processed_df = add_basic_features(merged_df)
    
    return processed_df

def preprocess_all_indices() -> dict:
    """
    Processes all indices defined in the config.
    Macro data is extracted once and then merged with each index's price data.
    Returns a dictionary mapping each index name to its processed DataFrame.
    """
    config = load_config()
    macro_data = extract_all_macro_data(config)
    processed_data = {}
    # Use the macro indicator categories from the config to iterate over indices.
    for index_name in config["data_requests"]["indices"].keys():
        print(f"Processing data for index: {index_name}")
        df = preprocess_index_data(index_name, config, macro_data)
        processed_data[index_name] = df
    return processed_data
