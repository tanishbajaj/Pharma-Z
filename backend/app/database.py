"""
Pharma-Z Database Setup

SQLAlchemy engine, session factory, and base model.
Uses async SQLite for development.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.config import settings

# ---------------------------------------------------------------------------
# Engine & Session
# ---------------------------------------------------------------------------

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},  # Required for SQLite
    echo=settings.debug,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all ORM models
Base = declarative_base()


# ---------------------------------------------------------------------------
# Dependency — use in FastAPI route functions
# ---------------------------------------------------------------------------

def get_database_session():
    """Yield a database session and ensure it is closed after use."""
    database_session = SessionLocal()
    try:
        yield database_session
    finally:
        database_session.close()
