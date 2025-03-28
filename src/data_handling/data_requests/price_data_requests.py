from typing import Dict
import pandas as pd
import yfinance as yf

# database related
from sqlalchemy.orm import Session
from src.utils.database import SessionLocal
from src.models import Indices

# Save data to database
from src.data_handling import data_saving
# from src.models.indices import Indices
from src.utils.environment_loading import load_config

import time
from src.utils.logger import logger

def fetch_historical_price_data(config: dict, index_name: str) -> pd.DataFrame:
    """
    Universal fetcher using config.yaml settings to get historcial price data.
    """
    index_config = config["data_requests"]["indices"][index_name]
    data = pd.DataFrame()

    yahoo_ticker = index_config.get('yahoo', None)
    if yahoo_ticker:
        data = _fetch_historical_data_via_yfinance(yahoo_ticker)
    
    return data

def _fetch_historical_data_via_yfinance(ticker: str) -> pd.DataFrame:
    """
    Fetch historical price data for a given ticker from Yahoo Finance.
    """
    try:
        logger.info(f"Fetching data for {ticker} via yfinance")
        # Fetch data with maximum period and daily interval and add delay
        time.sleep(1)
        data = yf.download(ticker, period='max', interval='1d', progress=False)
        
        if data.empty:
            logger.warning(f"No data fetched for {ticker} via yfinance.")
        else:
            logger.info(f"Successfully fetched data for {ticker} via yfinance.")
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {ticker} via yfinance: {e}")
        return pd.DataFrame()

def get_historical_price_data() -> None:
    config = load_config()
    indices_config = config.get('data_requests', {}).get('indices', {})

    session: Session = SessionLocal()
    try:
        for name, details in indices_config.items():
            index = session.query(Indices).filter(Indices.name == name).first()
            if not index:
                index = Indices(
                    name=name,
                    yahoo_symbol=details.get('yahoo')
                )
                session.add(index)
                session.commit()
                session.refresh(index)

            data = fetch_historical_price_data(config, name)
            if not data.empty:
                data_saving.save_raw_data_to_db(session, index, data)
            else:
                logger.warning(f"No data to save for {name}.")
            time.sleep(1)  # To prevent rate limiting
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        session.close()
