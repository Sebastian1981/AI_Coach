from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from api.database import get_session
from api.models import TrainingSession, TrainingSessionCreate, TrainingSessionRead

router = APIRouter()


@router.post("/", response_model=TrainingSessionRead, status_code=201)
def create_session(data: TrainingSessionCreate, db: Session = Depends(get_session)):
    obj = TrainingSession.model_validate(data)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


@router.get("/", response_model=List[TrainingSessionRead])
def list_sessions(db: Session = Depends(get_session)):
    return db.exec(
        select(TrainingSession).order_by(TrainingSession.date.desc())
    ).all()


@router.delete("/{session_id}")
def delete_session(session_id: int, db: Session = Depends(get_session)):
    obj = db.get(TrainingSession, session_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Not found")
    db.delete(obj)
    db.commit()
    return {"ok": True}
