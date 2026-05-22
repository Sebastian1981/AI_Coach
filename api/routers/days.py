from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from api.database import get_session
from api.models import DayIn, DayOut, ModuleOut, TrainingDay, TrainingModule

router = APIRouter()


@router.post("/", response_model=DayOut, status_code=201)
def create_day(data: DayIn, db: Session = Depends(get_session)):
    day = TrainingDay(date=data.date)
    db.add(day)
    db.flush()  # populate day.id

    saved_mods: list[TrainingModule] = []
    for i, m in enumerate(data.modules):
        mod = TrainingModule(
            day_id=day.id,
            order=i + 1,
            sport=m.sport,
            training_type=m.training_type,
            duration_min=m.duration_min,
            heart_rate_bpm=m.heart_rate_bpm,
            sets=m.sets,
            reps=m.reps,
            duration_s=m.duration_s,
            distance_m=m.distance_m,
            is_maximal=m.is_maximal,
            is_explosive=m.is_explosive,
            pause_s=m.pause_s,
            series=m.series,
            series_pause_s=m.series_pause_s,
            weight_kg=m.weight_kg,
            bodyweight=m.bodyweight,
            notes=m.notes,
            jump_type=m.jump_type,
            weight_vest=m.weight_vest,
            additional_weight_kg=m.additional_weight_kg,
        )
        db.add(mod)
        saved_mods.append(mod)

    db.commit()
    db.refresh(day)
    for mod in saved_mods:
        db.refresh(mod)

    return DayOut(
        id=day.id,
        date=day.date,
        modules=[
            ModuleOut(
                id=m.id, order=m.order, sport=m.sport, training_type=m.training_type,
                duration_min=m.duration_min, heart_rate_bpm=m.heart_rate_bpm,
                sets=m.sets, reps=m.reps, duration_s=m.duration_s, distance_m=m.distance_m, is_maximal=m.is_maximal, is_explosive=m.is_explosive, pause_s=m.pause_s,
                series=m.series, series_pause_s=m.series_pause_s,
                weight_kg=m.weight_kg, bodyweight=m.bodyweight, notes=m.notes,
                jump_type=m.jump_type, weight_vest=m.weight_vest, additional_weight_kg=m.additional_weight_kg,
            )
            for m in saved_mods
        ],
    )


@router.get("/", response_model=List[DayOut])
def list_days(db: Session = Depends(get_session)):
    days = db.exec(select(TrainingDay).order_by(TrainingDay.date.desc())).all()
    result = []
    for day in days:
        mods = db.exec(
            select(TrainingModule)
            .where(TrainingModule.day_id == day.id)
            .order_by(TrainingModule.order)
        ).all()
        result.append(DayOut(
            id=day.id,
            date=day.date,
            modules=[
                ModuleOut(
                    id=m.id, order=m.order, sport=m.sport, training_type=m.training_type,
                    duration_min=m.duration_min, heart_rate_bpm=m.heart_rate_bpm,
                    sets=m.sets, reps=m.reps, duration_s=m.duration_s, distance_m=m.distance_m, is_maximal=m.is_maximal, is_explosive=m.is_explosive, pause_s=m.pause_s,
                series=m.series, series_pause_s=m.series_pause_s,
                weight_kg=m.weight_kg, bodyweight=m.bodyweight, notes=m.notes,
                jump_type=m.jump_type, weight_vest=m.weight_vest, additional_weight_kg=m.additional_weight_kg,
                )
                for m in mods
            ],
        ))
    return result


@router.put("/{day_id}", response_model=DayOut)
def update_day(day_id: int, data: DayIn, db: Session = Depends(get_session)):
    day = db.get(TrainingDay, day_id)
    if not day:
        raise HTTPException(status_code=404, detail="Not found")
    day.date = data.date
    db.add(day)
    for mod in db.exec(select(TrainingModule).where(TrainingModule.day_id == day_id)).all():
        db.delete(mod)
    db.flush()

    saved_mods: list[TrainingModule] = []
    for i, m in enumerate(data.modules):
        mod = TrainingModule(
            day_id=day.id, order=i + 1, sport=m.sport,
            training_type=m.training_type, duration_min=m.duration_min,
            heart_rate_bpm=m.heart_rate_bpm, sets=m.sets, reps=m.reps, duration_s=m.duration_s, distance_m=m.distance_m, is_maximal=m.is_maximal, is_explosive=m.is_explosive,
            pause_s=m.pause_s, series=m.series, series_pause_s=m.series_pause_s,
            weight_kg=m.weight_kg, bodyweight=m.bodyweight, notes=m.notes,
            jump_type=m.jump_type, weight_vest=m.weight_vest, additional_weight_kg=m.additional_weight_kg,
        )
        db.add(mod)
        saved_mods.append(mod)

    db.commit()
    db.refresh(day)
    for mod in saved_mods:
        db.refresh(mod)

    return DayOut(
        id=day.id, date=day.date,
        modules=[
            ModuleOut(
                id=m.id, order=m.order, sport=m.sport, training_type=m.training_type,
                duration_min=m.duration_min, heart_rate_bpm=m.heart_rate_bpm,
                sets=m.sets, reps=m.reps, duration_s=m.duration_s, distance_m=m.distance_m, is_maximal=m.is_maximal, is_explosive=m.is_explosive, pause_s=m.pause_s,
                series=m.series, series_pause_s=m.series_pause_s,
                weight_kg=m.weight_kg, bodyweight=m.bodyweight, notes=m.notes,
                jump_type=m.jump_type, weight_vest=m.weight_vest, additional_weight_kg=m.additional_weight_kg,
            )
            for m in saved_mods
        ],
    )


@router.delete("/{day_id}")
def delete_day(day_id: int, db: Session = Depends(get_session)):
    day = db.get(TrainingDay, day_id)
    if not day:
        raise HTTPException(status_code=404, detail="Not found")
    for mod in db.exec(select(TrainingModule).where(TrainingModule.day_id == day_id)).all():
        db.delete(mod)
    db.delete(day)
    db.commit()
    return {"ok": True}
