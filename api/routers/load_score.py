import math
import sys
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from api.database import get_session
from api.models import AthleteProfile, TrainingDay, TrainingModule

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

router = APIRouter()


def _serialize(df: pd.DataFrame) -> list:
    """Convert DataFrame to JSON-safe list of dicts."""
    records = []
    for rec in df.to_dict(orient="records"):
        cleaned = {}
        for k, v in rec.items():
            if hasattr(v, "item"):
                v = v.item()
            if isinstance(v, float) and math.isnan(v):
                v = None
            if hasattr(v, "isoformat"):
                v = str(v)[:10]
            cleaned[k] = v
        records.append(cleaned)
    return records


@router.get("/")
def get_load_score(db: Session = Depends(get_session)):
    from src.load_score import compute_session_scores

    days = db.exec(select(TrainingDay).order_by(TrainingDay.date)).all()
    if not days:
        return {"daily": [], "weekly": []}

    rows = []
    for day in days:
        mods = db.exec(
            select(TrainingModule)
            .where(TrainingModule.day_id == day.id)
            .order_by(TrainingModule.order)
        ).all()
        for i, m in enumerate(mods):
            # Treat missing sets/series as 1 when at least one interval dimension exists
            if m.sets is not None or m.series is not None:
                effective_sets = (m.series or 1) * (m.sets or 1)
            else:
                effective_sets = None
            rows.append({
                "date":            day.date,
                "activity_nr":     f"Day{day.id}-Mod{i + 1}",
                "sport":           m.sport,
                "training_type":   m.training_type,
                "duration_[min]":  m.duration_min,
                "hear_rate_[bpm]": m.heart_rate_bpm,
                "sets":            effective_sets,
                "reps":            m.reps,
                "duration_[s]":    m.duration_s,
                "is_maximal":      m.is_maximal,
                "is_explosive":    m.is_explosive,
                "jump_type":       m.jump_type,
                "series":          m.series,
                "sets_per_serie":  m.sets,
                "pause_s":         m.pause_s,
                "series_pause_s":  m.series_pause_s,
            })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    profile = db.get(AthleteProfile, 1)
    hr_max  = profile.hr_max  if profile and profile.hr_max  else 190
    hr_rest = profile.hr_rest if profile and profile.hr_rest else 60
    df_daily, df_weekly = compute_session_scores(df, hr_rest=hr_rest, hr_max=hr_max)

    return {
        "daily":  _serialize(df_daily),
        "weekly": _serialize(df_weekly),
    }
