from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.config import Config

# Create SQLAlchemy engine
engine = create_engine(Config.db_url(), echo=True)

# Create a configured "Session" class
SessionLocal = sessionmaker(bind=engine)


# Dependency to get session
def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()