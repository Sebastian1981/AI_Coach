from fastapi import APIRouter, Depends
from sqlmodel import Session

from api.database import get_session
from api.models import AthleteProfile, ProfileIn

router = APIRouter()


@router.get("/", response_model=ProfileIn)
def get_profile(db: Session = Depends(get_session)):
    profile = db.get(AthleteProfile, 1)
    return ProfileIn(
        age=profile.age if profile else None,
        hr_max=profile.hr_max if profile else None,
        hr_rest=profile.hr_rest if profile else None,
        weight_kg=profile.weight_kg if profile else None,
        vo2max=profile.vo2max if profile else None,
    )


@router.put("/", response_model=ProfileIn)
def update_profile(data: ProfileIn, db: Session = Depends(get_session)):
    profile = db.get(AthleteProfile, 1)
    if not profile:
        profile = AthleteProfile(id=1)
    profile.age = data.age
    profile.hr_max = data.hr_max
    profile.hr_rest = data.hr_rest
    profile.weight_kg = data.weight_kg
    profile.vo2max = data.vo2max
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile
