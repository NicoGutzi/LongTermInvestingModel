import os
import requests
import pandas as pd

# database
from sqlalchemy.orm import Session
from src.utils.database import SessionLocal

from src.utils.environment_loading import load_config
from src.data_handling.data_saving import save_macro_indicator_data, IndicatorConfig

FRED_BASE_URL = 'https://api.stlouisfed.org/fred/series/observations'

def fetch_cpi_data() -> None:
    """Fetch CPI data from FRED and return as a pandas DataFrame."""
    # Get FRED inflation rate id from config
    config = load_config()
    fred_data = config.get('data_requests', {}).get('inflation_rate', {})

    # database session
    session: Session = SessionLocal()

    url = FRED_BASE_URL

    for country_name, country_config in fred_data.items():
        series_id = country_config.get('series_id')
        params = {
        'series_id': series_id,
        'api_key': os.getenv("FRED_API_KEY"),
        'file_type': 'json'
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            observations = data.get('observations', [])
            df = pd.DataFrame(observations)

            # Convert columns to proper data types
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            inflation_config = IndicatorConfig(country=country_name, metric="inflation_rate")
            save_macro_indicator_data(session, df, inflation_config)
        else:
            raise Exception(f"Failed to fetch data: {response.status_code}")
        

def fetch_interest_rate_dollar() -> None:
    """
    Fetch the interest rate from the FRED
    """
    # Get FRED inflation rate id from config
    config = load_config()
    fred_data = config.get('data_requests', {}).get('interest_rate', {})

    # database session
    session: Session = SessionLocal()

    url = FRED_BASE_URL

    for country_name, country_config in fred_data.items():
        series_id = country_config.get('series_id')
        params = {
        'series_id': series_id,
        'api_key': os.getenv("FRED_API_KEY"),
        'file_type': 'json'
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            observations = data.get('observations', [])
            df = pd.DataFrame(observations)

            # Convert columns to proper data types
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            interest_config = IndicatorConfig(country=country_name, metric="interest_rate")
            save_macro_indicator_data(session, df, interest_config)
        else:
            raise Exception(f"Failed to fetch data: {response.status_code}")
        

def fetch_unemployment_rate() -> None:
    """
    Fetch the unemployment rate from FRED, process the JSON response into a DataFrame,
    and save the data using the unified saving function.
    """
    config = load_config()
    fred_data = config.get('data_requests', {}).get('unemployment_rate', {})
    session: Session = SessionLocal()
    url = FRED_BASE_URL

    for country_name, country_config in fred_data.items():
        series_id = country_config.get('series_id')
        params = {
            'series_id': series_id,
            'api_key': os.getenv("FRED_API_KEY"),
            'file_type': 'json'
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            observations = data.get('observations', [])
            df = pd.DataFrame(observations)

            # Convert columns to proper data types.
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            unemployment_config = IndicatorConfig(country=country_name, metric="unemployment_rate")
            save_macro_indicator_data(session, df, unemployment_config)
        else:
            raise Exception(f"Failed to fetch unemployment data for {country_name}: {response.status_code}")
