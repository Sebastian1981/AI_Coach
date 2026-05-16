"""load_score.py — Training load calculation and (future) combined load/recovery score.

Current:
    compute_endurance_performance()  – TRIMP + sport-weighted total load per session

Planned:
    compute_combined_load()  – merges training load with Garmin recovery metrics
                               (resting HR, HRV) to produce a readiness-adjusted score
"""

import re

import numpy as np
import pandas as pd


# Sport-specific correction factors for orthopaedic / neuromuscular overhead.
# Reference: cycling = 1.0  (lowest ground-reaction force, no impact)
SPORT_LOAD_FACTORS: dict[str, float] = {
    "run":       1.3,
    "laufen":    1.3,
    "bike":      1.0,
    "rad":       1.0,
    "radfahren": 1.0,
    "swim":      1.1,
    "schwimmen": 1.1,
}


def compute_endurance_performance(
    df: pd.DataFrame,
    hr_rest: int = 60,
    hr_max: int = 190,
    b: float = 1.92,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute TRIMP and total load per session and as weekly sums.

    TRIMP (Banister):
        = t [min] × ΔHR × exp(b × ΔHR)
        with ΔHR = (avg_HR − hr_rest) / (hr_max − hr_rest)
    Fallback without HR data: TRIMP = duration [min] (volume proxy).

    Total load = TRIMP × sport_factor
        Captures: cardiovascular + orthopaedic load (sport-specific).
        NOT captured: neuromuscular load from sprints / maximal strength
        (HR response too slow for short efforts).

    Parameters
    ----------
    df       : DataFrame from parse_training_log()
    hr_rest  : resting heart rate [bpm]  (default 60)
    hr_max   : maximum heart rate [bpm]  (default 190)
    b        : sex factor (1.92 = male, 1.67 = female)

    Returns
    -------
    df_session : load per training session
    df_weekly  : weekly aggregates
    """
    result = df.copy()

    # 1. Total active duration [min]
    #    Steady-state: duration_[min]
    #    Intervals:    sets × duration_[s] / 60
    dur_min      = pd.to_numeric(result.get("duration_[min]", pd.Series(dtype=float, index=result.index)), errors="coerce")
    sets         = pd.to_numeric(result.get("sets",           pd.Series(dtype=float, index=result.index)), errors="coerce")
    dur_s        = pd.to_numeric(result.get("duration_[s]",   pd.Series(dtype=float, index=result.index)), errors="coerce")
    interval_min = (sets * dur_s / 60).where(sets.notna() & dur_s.notna())

    total_dur = dur_min.fillna(0) + interval_min.fillna(0)
    result["total_duration_min"] = total_dur.replace(0.0, np.nan)

    # 2. Average HR – parse range ("115-145") or single value
    hr_col = next(
        (c for c in ("hear_rate_[bpm]", "heart_rate_[bpm]") if c in result.columns),
        None,
    )

    def _parse_hr(val) -> float:
        if pd.isna(val):
            return np.nan
        parts = re.split(r"[-–]", str(val).strip())
        try:
            return float(np.mean([float(p.strip()) for p in parts]))
        except ValueError:
            return np.nan

    result["avg_hr_bpm"] = result[hr_col].apply(_parse_hr) if hr_col else np.nan

    # 3. TRIMP (cardiovascular load)
    delta_hr = ((result["avg_hr_bpm"] - hr_rest) / (hr_max - hr_rest)).clip(lower=0.0)
    result["trimp"] = np.where(
        result["avg_hr_bpm"].notna(),
        result["total_duration_min"] * delta_hr * np.exp(b * delta_hr),
        result["total_duration_min"],   # fallback: pure volume
    )

    # 4. Sport factor
    result["sport_factor"] = (
        result["sport"].str.lower().str.strip().map(SPORT_LOAD_FACTORS).fillna(1.0)
    )

    # 5. Total load = sport-weighted TRIMP
    result["total_load"] = (result["trimp"] * result["sport_factor"]).replace(0.0, np.nan)

    # 6. ISO calendar week
    iso = result["date"].dt.isocalendar()
    result["year"]          = iso.year.astype("Int64")
    result["calendar_week"] = iso.week.astype("Int64")

    df_session = result[[
        "date", "year", "calendar_week",
        "activity_nr", "sport", "training_type",
        "total_duration_min", "avg_hr_bpm", "trimp",
        "sport_factor", "total_load",
    ]].copy()

    df_weekly = (
        df_session
        .groupby(["year", "calendar_week"], dropna=True)
        .agg(
            sessions           = ("total_load",          "count"),
            total_duration_min = ("total_duration_min",  "sum"),
            trimp_weekly       = ("trimp",                "sum"),
            total_load_weekly  = ("total_load",           "sum"),
        )
        .reset_index()
    )

    return df_session, df_weekly


def compute_combined_load(
    df_session: pd.DataFrame,
    df_wellness: pd.DataFrame,
) -> pd.DataFrame:
    """Merge training load with Garmin recovery metrics into a daily readiness score.

    Planned implementation:
    - Join df_session (TRIMP, total_load) with df_wellness (resting HR, HRV) on date
    - Compute ATL (acute training load, ~7-day EWM) and CTL (chronic, ~42-day EWM)
    - Apply HRV-based recovery modifier:
        recovery_modifier = f(hrv_last_night_avg / hrv_weekly_avg)
    - Return readiness_score = CTL × recovery_modifier

    Parameters
    ----------
    df_session  : output of compute_endurance_performance()[0]
    df_wellness : output of fetch_wellness_data() from src.wellness

    Returns
    -------
    DataFrame with columns: date, atl, ctl, tsb, hrv_modifier, readiness_score
    """
    raise NotImplementedError(
        "Combined load score not yet implemented. "
        "Requires both df_session and df_wellness."
    )
