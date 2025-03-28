import pandas as pd

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with daily price data (with 'close' column), compute:
      - Daily returns as the percentage change of closing prices.
      - 20-day moving average of the closing price.
      - 20-day rolling volatility based on daily returns.
    """
    df = df.copy()
    df['daily_return'] = df['close'].pct_change()
    df['20d_ma'] = df['close'].rolling(window=20).mean()
    df['20d_vol'] = df['daily_return'].rolling(window=20).std()
    return df.dropna()
