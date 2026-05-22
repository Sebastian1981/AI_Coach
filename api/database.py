from pathlib import Path

from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine

_DB_FILE = Path(__file__).resolve().parent.parent / "data" / "training.db"
DATABASE_URL = f"sqlite:///{_DB_FILE}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


def create_db() -> None:
    SQLModel.metadata.create_all(engine)
    _migrate_db()


def _migrate_db() -> None:
    """Add new columns to existing tables without touching existing data."""
    new_columns = [
        ("trainingmodule", "jump_type",            "TEXT"),
        ("trainingmodule", "weight_vest",           "BOOLEAN"),
        ("trainingmodule", "additional_weight_kg",  "FLOAT"),
    ]
    with engine.connect() as conn:
        for table, column, col_type in new_columns:
            existing = {row[1] for row in conn.execute(text(f"PRAGMA table_info({table})"))}
            if column not in existing:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"))
        conn.commit()


def get_session():
    with Session(engine) as session:
        yield session
