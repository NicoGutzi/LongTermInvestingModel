# tests/conftest.py

import pytest
from src.utils.database import SessionLocal

@pytest.fixture(scope="module")
def session():
    session = SessionLocal()
    yield session
    session.close()
