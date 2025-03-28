from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.models import Base
import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(dotenv_path='.env')

# Create the SQLAlchemy engine
engine = create_engine(os.getenv("DATABASE_URL"), pool_pre_ping=True)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    try:
        Base.metadata.create_all(engine)
        logger.info("Database tables created successfully.")
    except Exception as e:
        logger.error(f"Error initializing the database: {e}")
        raise e