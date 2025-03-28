import sys
import os 

# Add the top-level project directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from src.utils import database
from src.data_handling.data_requests.price_data_requests import get_historical_price_data
from src.data_handling.data_requests.fred_requests import fetch_cpi_data, fetch_interest_rate_dollar, fetch_unemployment_rate

from src.utils.logger import logger

def run_data_ingest():
    """
    Run the data ingestion process.
    """
    try:
        logger.info("Initializing the database.")
        database.init_db()
        logger.info("Starting data ingestion process.")
        get_historical_price_data()
        logger.info("Data ingestion process completed successfully.")
        fetch_cpi_data()
        logger.info("Successfully fetched cpi data.")
        fetch_interest_rate_dollar()
        logger.info("Fetched interest rates.")
        fetch_unemployment_rate()
        logger.info("Fetched unemployment rates.")

    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_data_ingest()
