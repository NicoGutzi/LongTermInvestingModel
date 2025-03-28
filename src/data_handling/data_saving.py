from dataclasses import dataclass
from typing import Set, List
from datetime import date

from sqlalchemy.orm import Session
import pandas as pd
import logging
from src.models import Indices, RawPriceData, MacroeconomicIndicator

logger = logging.getLogger(__name__)

def save_raw_data_to_db(session: Session, index: Indices, data: pd.DataFrame) -> None:
    """
    Save the fetched data to the raw_price_data table.
    """
    try:
        # Query the database for existing dates for this index
        existing_dates = {
            record.date for record in session.query(RawPriceData.date)
            .filter(RawPriceData.index_id == index.id).all()
        }

        records = []  # List to store our new database records

        # Iterate over each row in the DataFrame
        for date, row in data.iterrows():
            # skip existing dates
            # Convert the pandas Timestamp to a datetime.date
            date_obj = date.date() if hasattr(date, "date") else date
            if date_obj in existing_dates:
                continue

            record = RawPriceData(
                index_id=index.id,      # Link to the specific index
                date=date,          # Use row.name to access the date
                open=float(row.Open.iloc[0]),   # Convert to native float
                high=float(row.High.iloc[0]),
                low=float(row.Low.iloc[0]),
                close=float(row.Close.iloc[0]),
                volume=float(row.Volume.iloc[0])
            )
            records.append(record)
        
        session.add_all(records)
        session.commit()
        logger.info(f"Saved {len(records)} records for {index.name} to the database.")
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving data for {index.name}: {e}")


@dataclass
class IndicatorConfig:
    country: str
    metric: str
    value_col: str = "value"  # Default column name in the DataFrame

class MacroIndicatorRepository:
    """
    Encapsulates all database operations related to macroeconomic indicators.
    
    This repository provides two main functionalities:
      1. Retrieving the set of dates for which records already exist in the database for a given country and metric.
      2. Bulk saving a list of new macroeconomic indicator records to the database.
    """
    def __init__(self, session: Session):
        self.session = session

    def get_existing_dates(self, country: str, metric: str) -> Set[date]:
        # Retrieves dates for which records already exist for the given country and metric.
        return {
            record.date for record in self.session.query(MacroeconomicIndicator.date)
            .filter(
                MacroeconomicIndicator.country == country,
                MacroeconomicIndicator.metric == metric
            )
            .all()
        }

    def bulk_save_indicators(self, indicators: List[MacroeconomicIndicator]) -> None:
        # Bulk saves a list of macroeconomic indicator records.
        try:
            self.session.bulk_save_objects(indicators)
            self.session.commit()
            logger.info(f"Saved {len(indicators)} records to the database.")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error saving macroeconomic indicator data: {e}")

def save_macro_indicator_data(session: Session, data: pd.DataFrame, config: IndicatorConfig) -> None:
    """
    Transforms a DataFrame containing macroeconomic indicator data into model objects and
    saves them using the repository.
    """
    indicator_repo = MacroIndicatorRepository(session)
    existing_dates = indicator_repo.get_existing_dates(config.country, config.metric)
    
    new_records = []
    for date_val, row in data.iterrows():
        # Ensure the date is a datetime.date object.
        date_obj = date_val.date() if hasattr(date_val, "date") else date_val
        if date_obj in existing_dates:
            continue  # Skip if a record for this date already exists.
        record = MacroeconomicIndicator(
            country=config.country,
            date=date_obj,
            metric=config.metric,
            value=float(row[config.value_col])
        )
        new_records.append(record)
    
    indicator_repo.bulk_save_indicators(new_records)