from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

_DB_FILE = Path(__file__).resolve().parent.parent / "data" / "training.db"
DATABASE_URL = f"sqlite:///{_DB_FILE}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


def create_db() -> None:
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
