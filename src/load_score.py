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


# ─────────────────────────────────────────────────────────────────────────────
# Multi-module session scoring
# ─────────────────────────────────────────────────────────────────────────────

_SPEED_TYPES: frozenset[str] = frozenset({
    "sprint", "sprints", "speed drills", "skippings", "power", "jumps",
})
_SPEED_SPORTS: frozenset[str] = frozenset({
    "speed drills", "sprint", "power",
})


def _classify_module(
    sport: str,
    training_type: str,
    avg_hr: float,
    hr_max: int,
) -> str:
    """Return stimulus category: 'speed' | 'aerobic' | 'lactate' | 'vo2max'."""
    if (training_type or "").lower().strip() in _SPEED_TYPES:
        return "speed"
    if (sport or "").lower().strip() in _SPEED_SPORTS:
        return "speed"
    if pd.isna(avg_hr):
        return "aerobic"
    pct = avg_hr / hr_max
    if pct < 0.75:
        return "aerobic"
    if pct < 0.87:
        return "threshold"
    return "lactate"


# Recovery-cost weights per stimulus zone.
# Rationale: recovery demand is non-linear with intensity.
# Zone multipliers reflect empirical recovery-time ratios relative to Zone 1.
#   Aerobic  (< 75 % HRmax)  → 1–2 days  → k = 1.0  (baseline)
#   Schwelle (75–87 % HRmax) → 2–3 days  → k = 2.5
#   Laktat   (> 87 % HRmax)  → 4–6 days  → k = 6.0
#   Speed/Strength (neuromusc.) → 3–6 days → k_NM = 1.0
#     (speed volume in [s × k_Sport] contributes directly)
_K_AEROBIC   : float = 1.0
_K_THRESHOLD : float = 2.5
_K_LACTATE   : float = 6.0
_K_SPEED     : float = 1.0   # [s × k_Sport] → same scale as recovery-weighted TRIMP


def compute_session_scores(
    df: pd.DataFrame,
    hr_rest: int = 60,
    hr_max: int = 190,
    b: float = 1.92,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-day and weekly training stimulus scores from a module DataFrame.

    Stimulus categories
    -------------------
    ausdauerreiz      – aerobic TRIMP (HR < 75 % HRmax)
    schwellenreiz     – threshold TRIMP (75–87 % HRmax, around LT2)
    laktatreiz        – high-intensity TRIMP (> 87 % HRmax, lactate tolerance)
    schnelligkeitsreiz – neuromuscular volume: sets × duration_s × k_Sport [s]
    regenerationsbedarf – recovery-weighted sum of all stimulus scores:
                        aer×1.0 + schw×2.5 + lak×6.0 + speed×1.0

    Parameters
    ----------
    df : DataFrame with columns expected by compute_endurance_performance()
         plus 'sets' and 'duration_[s]' for speed modules.

    Returns
    -------
    df_daily   : one row per training day
    df_weekly  : weekly aggregates
    """
    df_mod, _ = compute_endurance_performance(df, hr_rest=hr_rest, hr_max=hr_max, b=b)

    df_mod["category"] = [
        _classify_module(r.sport, r.training_type, r.avg_hr_bpm, hr_max)
        for r in df_mod.itertuples()
    ]

    # Speed volume from original df (HR too slow for short efforts)
    sets_s  = pd.to_numeric(df.get("sets",         pd.Series(dtype=float, index=df.index)), errors="coerce")
    dur_s_s = pd.to_numeric(df.get("duration_[s]", pd.Series(dtype=float, index=df.index)), errors="coerce")
    df_mod["speed_vol_s"] = (sets_s.fillna(0) * dur_s_s.fillna(0)).values

    # Aggregate per day
    records = []
    for date_val, grp in df_mod.groupby("date"):
        is_speed = grp["category"] == "speed"
        aer   = grp.loc[grp["category"] == "aerobic",    "total_load"].sum()
        schw  = grp.loc[grp["category"] == "threshold",  "total_load"].sum()
        lak   = grp.loc[grp["category"] == "lactate",    "total_load"].sum()
        speed = (grp.loc[is_speed, "speed_vol_s"] * grp.loc[is_speed, "sport_factor"]).sum()
        records.append({
            "date":               date_val,
            "module_count":       len(grp),
            "regenerationsbedarf": (aer   * _K_AEROBIC
                                   + schw  * _K_THRESHOLD
                                   + lak   * _K_LACTATE
                                   + speed * _K_SPEED),
            "ausdauerreiz":       aer,
            "schwellenreiz":      schw,
            "laktatreiz":         lak,
            "schnelligkeitsreiz": speed,
        })

    df_daily = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

    iso = pd.to_datetime(df_daily["date"]).dt.isocalendar()
    df_daily["year"]          = iso.year.astype("Int64")
    df_daily["calendar_week"] = iso.week.astype("Int64")

    df_weekly = (
        df_daily
        .groupby(["year", "calendar_week"], dropna=True)
        .agg(
            training_days         = ("date",                  "count"),
            total_modules         = ("module_count",          "sum"),
            regenerationsbedarf   = ("regenerationsbedarf",   "sum"),
            ausdauerreiz          = ("ausdauerreiz",          "sum"),
            schwellenreiz      = ("schwellenreiz",      "sum"),
            laktatreiz         = ("laktatreiz",         "sum"),
            schnelligkeitsreiz = ("schnelligkeitsreiz", "sum"),
        )
        .reset_index()
    )

    return df_daily, df_weekly
