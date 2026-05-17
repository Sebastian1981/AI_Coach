from datetime import date
from typing import List, Optional

from pydantic import BaseModel
from sqlmodel import Field, SQLModel


# ── SQLModel table models (persisted to SQLite) ───────────────────────────────

class TrainingDay(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    date: date


class TrainingModule(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    day_id: int = Field(foreign_key="trainingday.id", index=True)
    order: int = Field(default=1)
    sport: str
    training_type: str
    duration_min: Optional[float] = None
    heart_rate_bpm: Optional[str] = None   # "145" or "130-160"
    sets: Optional[int] = None
    duration_s: Optional[float] = None
    pause_s: Optional[float] = None
    series: Optional[int] = None
    series_pause_s: Optional[float] = None
    weight_kg: Optional[float] = None
    bodyweight: Optional[bool] = None
    notes: Optional[str] = None


class AthleteProfile(SQLModel, table=True):
    id: int = Field(default=1, primary_key=True)
    age: Optional[int] = None
    hr_max: Optional[int] = None
    hr_rest: Optional[int] = None
    weight_kg: Optional[float] = None
    vo2max: Optional[float] = None


# ── Pydantic request / response models ────────────────────────────────────────

class ProfileIn(BaseModel):
    age: Optional[int] = None
    hr_max: Optional[int] = None
    hr_rest: Optional[int] = None
    weight_kg: Optional[float] = None
    vo2max: Optional[float] = None


class ModuleIn(BaseModel):
    order: int = 1
    sport: str
    training_type: str
    duration_min: Optional[float] = None
    heart_rate_bpm: Optional[str] = None
    sets: Optional[int] = None
    duration_s: Optional[float] = None
    pause_s: Optional[float] = None
    series: Optional[int] = None
    series_pause_s: Optional[float] = None
    weight_kg: Optional[float] = None
    bodyweight: Optional[bool] = None
    notes: Optional[str] = None


class ModuleOut(ModuleIn):
    id: int


class DayIn(BaseModel):
    date: date
    modules: List[ModuleIn]


class DayOut(BaseModel):
    id: int
    date: date
    modules: List[ModuleOut]
