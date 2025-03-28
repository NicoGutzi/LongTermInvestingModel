from dateutil.relativedelta import relativedelta
import pytest
import pandas as pd
from sqlalchemy import func

from src.models.base import RawPriceData, MacroeconomicIndicator
from src.utils.database import SessionLocal

# Pytest fixture for providing a session
@pytest.fixture(scope="module")
def session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

def test_raw_price_data_daily(session):
    """
    Test that daily raw price data is complete.
    For a given index (e.g., index_id = 1), check that the number of records equals
    the number of business days (excluding weekends) between the minimum and maximum date.
    If there is a discrepancy, the test reports the missing dates.
    """
    index_id = 1  # Adjust this to a valid index in your test database.
    min_date, max_date, count = session.query(
        func.min(RawPriceData.date),
        func.max(RawPriceData.date),
        func.count(RawPriceData.date)
    ).filter(RawPriceData.index_id == index_id).one()

    # Generate holiday dates for each year in the range.
    years = range(min_date.year, max_date.year + 1)
    # Assume New Year's Day and Christmas Day are holidays:
    holidays = [pd.Timestamp(f"{year}-01-01") for year in years] + [pd.Timestamp(f"{year}-12-25") for year in years]
    

    # Create a custom business day frequency that excludes weekends and these holidays.
    custom_bd = pd.tseries.offsets.CustomBusinessDay(holidays=holidays)
    expected_dates = pd.date_range(start=min_date, end=max_date, freq=custom_bd)
    expected_count = len(expected_dates)

    # Retrieve the actual dates from the database
    actual_dates = [record[0] for record in session.query(RawPriceData.date)
                    .filter(RawPriceData.index_id == index_id).all()]
    # Convert actual dates to a DatetimeIndex for easy set operations
    actual_dates = pd.to_datetime(actual_dates)

    # Determine which expected dates are missing from the actual dates
    missing_dates = sorted(set(expected_dates) - set(actual_dates))

    error_msg = (
        f"Index {index_id}: Expected {expected_count} records (business days) between {min_date} and {max_date}, "
        f"but found {count}. Missing dates: {missing_dates}"
    )
    
    assert count == expected_count, error_msg

def test_macro_indicator_monthly(session):
    """
    Test that monthly macroeconomic indicator data is complete for a given country and metric.
    For example, check the USA inflation_rate indicator.
    """
    country = "USA"
    metric = "inflation_rate"
    min_date, max_date, count = session.query(
        func.min(MacroeconomicIndicator.date),
        func.max(MacroeconomicIndicator.date),
        func.count(MacroeconomicIndicator.date)
    ).filter(
        MacroeconomicIndicator.country == country,
        MacroeconomicIndicator.metric == metric
    ).one()

    # Calculate expected number of months using relativedelta
    rd = relativedelta(max_date, min_date)
    expected_months = rd.years * 12 + rd.months + 1  # include the starting month
    assert count == expected_months, (
        f"{country} {metric}: Expected {expected_months} monthly records between {min_date} and {max_date}, but found {count}"
    )

def test_macro_indicator_value_ranges(session):
    """
    Test that macroeconomic indicator values fall within plausible ranges.
    Adjust the thresholds as necessary.
    """
    indicators = session.query(MacroeconomicIndicator.metric).distinct().all()
    for (metric,) in indicators:
        min_val, max_val, avg_val = session.query(
            func.min(MacroeconomicIndicator.value),
            func.max(MacroeconomicIndicator.value),
            func.avg(MacroeconomicIndicator.value)
        ).filter(MacroeconomicIndicator.metric == metric).one()
        print(f"For {metric}: min={min_val}, max={max_val}, avg={avg_val}")
        
        # Example assertions (adjust ranges based on your domain knowledge)
        if metric == "inflation_rate":
            assert 0 <= min_val < max_val < 50, f"{metric} values seem off"
        elif metric == "interest_rate":
            assert 0 <= min_val < max_val < 20, f"{metric} values seem off"
        elif metric == "unemployment_rate":
            assert 0 <= min_val < max_val < 50, f"{metric} values seem off"
