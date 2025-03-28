from sqlalchemy import Column, Integer, String, UniqueConstraint, Date, Numeric, ForeignKey, Float
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Indices(Base):
    __tablename__ = 'indices'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    yahoo_symbol = Column(String, nullable=True)

class RawPriceData(Base):
    __tablename__ = 'raw_price_data'

    id = Column(Integer, primary_key=True, index=True)
    index_id = Column(Integer, ForeignKey('indices.id'), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Numeric, nullable=True)
    high = Column(Numeric, nullable=True)
    low = Column(Numeric, nullable=True)
    close = Column(Numeric, nullable=True)
    volume = Column(Integer, nullable=True)

    __table_args__ = (
        UniqueConstraint('index_id', 'date', name='uix_index_date'),
    )


class MacroeconomicIndicator(Base):
    __tablename__ = 'macro_indicators'
    country = Column(String, primary_key=True)
    date = Column(Date, primary_key=True)
    metric = Column(String, primary_key=True)  # e.g., 'inflation_rate', 'interest_rate'
    value = Column(Float, nullable=False)