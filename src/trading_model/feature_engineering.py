import pandas as pd

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with daily price data (with 'close' column), compute:
      - Daily returns as the percentage change of closing prices.
      - 20-day moving average of the closing price.
      - 20-day rolling volatility based on daily returns.
    """
    df = df.copy()

    # Compute daily returns
    df['daily_return'] = df['close'].pct_change()

    # Compute 20-day moving average
    df['20d_ma'] = df['close'].rolling(window=20).mean()

    # Compute 20-day rolling volatility
    df['20d_vol'] = df['daily_return'].rolling(window=20).std()

    # Fill NaN values in the technical indicators
    df['daily_return'] = df['daily_return'].fillna(0)
    df['20d_ma'] = df['20d_ma'].ffill()  # Forward fill for moving average
    df.iloc[:20, df.columns.get_loc("20d_ma")] = 0 

    df['20d_vol'] = df['20d_vol'].ffill()  # Forward fill for volatility
    df.iloc[:20, df.columns.get_loc("20d_vol")] = 0 

    return df
