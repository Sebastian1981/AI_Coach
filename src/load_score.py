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
    "hiking":    1.2,
    "wandern":   1.2,
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
    "sprint", "sprints",
    "steigerungsläufe", "acceleration run",
    "speed drills", "skippings", "a-skips",
    "hopserlauf", "kniehebelauf", "anfersen", "legcycling drill", "pawing drill", "seitgalopp",
    "treppenläufe (1er)", "treppenläufe (2er)",
    "jumps", "easy jumps", "explosive jumps", "jump runs", "sprungläufe",
    "sprünge",  # structured jump modules (with jump_type subcategory)
})
_SPEED_SPORTS: frozenset[str] = frozenset({
    "speed drills", "sprint",
})

# Sub-classification within the speed category
_SPRINT_TYPES: frozenset[str] = frozenset({"sprint", "sprints"})
_ACCEL_TYPES:  frozenset[str] = frozenset({"steigerungsläufe", "acceleration run"})
_DRILL_TYPES:  frozenset[str] = frozenset({
    "skippings", "a-skips", "speed drills",
    "hopserlauf", "kniehebelauf", "anfersen", "legcycling drill", "pawing drill", "seitgalopp",
    "treppenläufe (1er)", "treppenläufe (2er)",
})
_JUMP_TYPES:   frozenset[str] = frozenset({
    "jumps", "easy jumps", "explosive jumps", "jump runs", "sprungläufe", "sprünge",
    "treppenläufe (1er)", "treppenläufe (2er)",
})
_BIKE_SPORTS:  frozenset[str] = frozenset({"bike", "rad", "radfahren", "cycling", "fahrrad"})
_KRAFT_SPORTS: frozenset[str] = frozenset({"kraft"})


def _speed_intensity_factor(sport: str, training_type: str, is_maximal, row_jump_type=None) -> float:
    """Return intensity weight within the speed category (applied on top of k_sport).

    Effective factor (this value × k_sport):
      Sprint maximal       Laufen : 1.00 × 1.3 = 1.30
      Sprint submaximal    Laufen : 0.35 × 1.3 = 0.46
      Steigerungsläufe     Laufen : 0.22 × 1.3 = 0.29
      Jump Runs            Laufen : 0.35 × 1.3 = 0.46
      Explosive Jumps      Laufen : 0.50 × 1.3 = 0.65
      Easy Jumps           Laufen : 0.20 × 1.3 = 0.26
      Drills/skippings     Laufen : 0.15 × 1.3 = 0.20
      Sprint maximal       Rad    : 0.12 × 1.0 = 0.12
      Sprint submaximal    Rad    : 0.06 × 1.0 = 0.06
      Drill / Jump         Rad    : 0.05 × 1.0 = 0.05

    is_maximal=None (legacy data, no flag set) → treated as maximal for sprints.
    """
    sport_lc = (sport or "").lower().strip()
    type_lc  = (training_type or "").lower().strip()
    is_bike  = sport_lc in _BIKE_SPORTS

    if type_lc in _SPRINT_TYPES:
        maximal = is_maximal is not False   # True or None → maximal
        if is_bike:
            return 0.12 if maximal else 0.06
        return 1.0 if maximal else 0.35

    if type_lc in _ACCEL_TYPES:
        # Steigerungsläufe: start slow → less peak load than sub-max sprint
        return 0.05 if is_bike else 0.22

    if type_lc in _JUMP_TYPES:
        if is_bike:
            return 0.05
        # New structured jump_type field (training_type == 'sprünge')
        jump_sub = str(row_jump_type or "").lower().strip() if row_jump_type is not None else type_lc
        if jump_sub in ("explosive sprünge", "explosive jumps"):
            return 0.50   # plyometric: max effort, high neuromuscular cost
        if jump_sub in ("laufsprünge", "jump runs", "sprungläufe", "treppensprünge", "stair jumps"):
            return 0.35   # bounding / stair jumps: comparable to sub-max sprint
        if jump_sub in ("fußgelenkssprünge", "reactive jumps"):
            return 0.30   # reactive ankle jumps
        if jump_sub in ("einbeinsprünge", "single-leg jumps"):
            return 0.40   # unilateral: higher load per rep
        if jump_sub == "kraftsprünge":
            return 0.25   # power jumps ~50 % effort
        # Legacy names or no subtype
        if type_lc == "explosive jumps":
            return 0.50
        if type_lc in ("jump runs", "sprungläufe"):
            return 0.35
        return 0.20       # easy jumps / legacy 'jumps'

    if type_lc == "treppenläufe (2er)":
        return 0.05 if is_bike else 0.35   # two-step stair run: like sub-max sprint
    if type_lc == "treppenläufe (1er)":
        return 0.05 if is_bike else 0.25   # one-step skipping stair: drill-like

    # Drills, skippings, and any other speed type
    return 0.05 if is_bike else 0.15


def _classify_module(
    sport: str,
    training_type: str,
    avg_hr: float,
    hr_max: int,
) -> str:
    """Return stimulus category: 'speed' | 'aerobic' | 'threshold' | 'lactate' | 'kraft'."""
    if (sport or "").lower().strip() in _KRAFT_SPORTS:
        return "kraft"
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
# Based on Banister IR model (Morton et al., J Appl Physiol 1990) and tissue-specific
# fatigue decay constants (Howatson & van Someren, Sports Med 2008).
#   Aerobic  (< 75 % HRmax)  → 1–2 days  → k = 1.0  (baseline)
#   Schwelle (75–87 % HRmax) → 2–3 days  → k = 3.0  (updated from 2.5)
#   Laktat   (> 87 % HRmax)  → 4–6 days  → k = 6.0
#   Speed (neuromuscular)    → 3–5 days  → k = 4.0  (updated: impact + tendon > Kraft)
#   Kraft    (strength)      → 2–4 days  → k = 3.0
_K_AEROBIC   : float = 1.0
_K_THRESHOLD : float = 3.0   # updated: 2.5 → 3.0
_K_LACTATE   : float = 6.0
_K_SPEED     : float = 4.0   # updated: 1.0 → 4.0 (neuromuscular + impact load)
_K_KRAFT     : float = 3.0

# Recovery-time calibration (Banister fatigue-decay model: T_rec = ln(1 + L/L₀)).
# Two separate reference loads because the two subsystems use different units:
#
#   _L_REF_CARDIO  – TRIMP units (metabolic load).
#                    Calibrated: 20 min easy run at ~70 % HRmax → 1.0 day.
#                    TRIMP≈33, aer=33×1.3=42.9, L₀ = 42.9 / (e−1) ≈ 25.
#
#   _L_REF_NEURO   – eff-seconds × k_Sport × k_Speed units (neuromuscular load).
#                    Calibrated: 3 × 100 m near-maximal sprint (12 s each) → 3.0 days.
#                    neuro_wt = 3×12×1.0×1.3×4.0 = 187.2, L₀ = 187.2 / (e³−1) ≈ 10.
_L_REF_CARDIO: float = 25.0
_L_REF_NEURO:  float = 10.0
_JUMP_REP_TIME_S: float = 2.0  # effective seconds per jump rep (plyometric effort)


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
                        aer×1.0 + schw×3.0 + lak×6.0 + speed×4.0 + kraft×3.0
    recovery_days       – estimated days to full recovery (Banister fatigue-decay model,
                        Morton et al. 1990): two-system model (cardio vs. neuromusc.)

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

    # For jump modules: when reps are given but no duration_s, estimate effective
    # work time as reps × _JUMP_REP_TIME_S (≈2 s/rep for plyometric jumps).
    # This keeps speed_vol and block_time non-zero regardless of input style.
    _training_type_col = df.get("training_type", pd.Series([""] * len(df), dtype=str, index=df.index))
    _is_jump_row = _training_type_col.str.lower().str.strip().isin(_JUMP_TYPES)
    _reps_raw    = pd.to_numeric(df.get("reps", pd.Series(dtype=float, index=df.index)), errors="coerce")
    dur_s_effective = dur_s_s.where(
        dur_s_s.notna() | ~_is_jump_row,
        _reps_raw * _JUMP_REP_TIME_S,
    )

    # Per-module intensity weight: sprint max/sub vs. drill, and bike penalty
    _is_maximal  = df.get("is_maximal", pd.Series([None] * len(df), dtype=object, index=df.index))
    _jump_types  = df.get("jump_type",  pd.Series([None] * len(df), dtype=object, index=df.index))
    _spd_factors = [
        _speed_intensity_factor(r.sport, r.training_type, _is_maximal.iloc[i], _jump_types.iloc[i])
        for i, r in enumerate(df_mod.itertuples())
    ]
    df_mod["speed_vol_s"] = (sets_s.fillna(0) * dur_s_effective.fillna(0)).values * np.array(_spd_factors)

    # Kraft volume: effective_sets × reps × k_explosive × k_sport
    reps_s       = pd.to_numeric(df.get("reps", pd.Series(dtype=float, index=df.index)), errors="coerce")
    _is_explosive = df.get("is_explosive", pd.Series([None] * len(df), dtype=object, index=df.index))
    _kft_factors  = [1.5 if _is_explosive.iloc[i] is True else 1.0 for i in range(len(df))]
    df_mod["kraft_vol"] = (sets_s.fillna(0) * reps_s.fillna(0)).values * np.array(_kft_factors)

    # ── Cardiovascular TRIMP for speed modules with HR data ──────────────────
    # If the user records an average HR for a speed module (sprints or drills),
    # that HR already reflects rest periods too — meaning HR stayed elevated.
    # We use total block time (work + pauses) to compute a supplementary TRIMP
    # and add it to the appropriate HR zone (aerobic / threshold / lactate).
    # This captures the cardiovascular component of dense sprint/drill circuits.
    series_v   = pd.to_numeric(df.get("series",        pd.Series(dtype=float, index=df.index)), errors="coerce").fillna(1)
    sets_v     = pd.to_numeric(df.get("sets_per_serie", pd.Series(dtype=float, index=df.index)), errors="coerce").fillna(1)
    pause_v    = pd.to_numeric(df.get("pause_s",        pd.Series(dtype=float, index=df.index)), errors="coerce").fillna(0)
    s_pause_v  = pd.to_numeric(df.get("series_pause_s", pd.Series(dtype=float, index=df.index)), errors="coerce").fillna(0)
    dur_s_raw  = dur_s_effective.fillna(0)  # uses rep×time fallback for jump modules

    block_s = (series_v * sets_v * dur_s_raw
               + series_v * (sets_v - 1).clip(lower=0) * pause_v
               + (series_v - 1).clip(lower=0) * s_pause_v)
    block_min = (block_s / 60.0).replace(0.0, np.nan)

    is_drill_m = df_mod["category"] == "speed"
    has_hr_m   = df_mod["avg_hr_bpm"].notna()
    delta_hr_d = ((df_mod["avg_hr_bpm"] - hr_rest) / (hr_max - hr_rest)).clip(lower=0.0)
    drill_trimp_raw = np.where(
        is_drill_m & has_hr_m & block_min.notna().values,
        block_min.fillna(0).values * delta_hr_d * np.exp(b * delta_hr_d),
        0.0,
    )
    df_mod["drill_trimp_load"] = drill_trimp_raw * df_mod["sport_factor"].values
    hr_pct = df_mod["avg_hr_bpm"] / hr_max
    df_mod["drill_zone"] = np.where(
        is_drill_m & has_hr_m,
        np.where(hr_pct < 0.75, "aerobic", np.where(hr_pct < 0.87, "threshold", "lactate")),
        "none",
    )

    # Aggregate per day
    records = []
    for date_val, grp in df_mod.groupby("date"):
        is_speed   = grp["category"] == "speed"
        is_kraft   = grp["category"] == "kraft"
        drill_aer  = grp.loc[grp["drill_zone"] == "aerobic",    "drill_trimp_load"].sum()
        drill_schw = grp.loc[grp["drill_zone"] == "threshold",  "drill_trimp_load"].sum()
        drill_lak  = grp.loc[grp["drill_zone"] == "lactate",    "drill_trimp_load"].sum()
        aer   = grp.loc[grp["category"] == "aerobic",    "total_load"].sum() + drill_aer
        schw  = grp.loc[grp["category"] == "threshold",  "total_load"].sum() + drill_schw
        lak   = grp.loc[grp["category"] == "lactate",    "total_load"].sum() + drill_lak
        speed = (grp.loc[is_speed, "speed_vol_s"] * grp.loc[is_speed, "sport_factor"]).sum()
        kraft = (grp.loc[is_kraft, "kraft_vol"]   * grp.loc[is_kraft, "sport_factor"]).sum()
        cardio_wt = aer * _K_AEROBIC + schw * _K_THRESHOLD + lak * _K_LACTATE
        neuro_wt  = speed * _K_SPEED  + kraft * _K_KRAFT
        # Recovery estimate: Banister fatigue-decay model, two-system approach.
        # T_rec = τ · ln(1 + L/L₀), dominant system sets baseline, secondary adds 40%.
        t_cardio = float(np.log1p(cardio_wt / _L_REF_CARDIO)) if cardio_wt > 0 else 0.0
        t_neuro  = float(np.log1p(neuro_wt  / _L_REF_NEURO))  if neuro_wt  > 0 else 0.0
        t_dom    = max(t_cardio, t_neuro)
        t_sec    = min(t_cardio, t_neuro)
        records.append({
            "date":               date_val,
            "module_count":       len(grp),
            "regenerationsbedarf": cardio_wt + neuro_wt,
            "ausdauerreiz":       aer,
            "schwellenreiz":      schw,
            "laktatreiz":         lak,
            "schnelligkeitsreiz": speed,
            "kraftreiz":          kraft,
            "recovery_days":      round(t_dom + 0.4 * t_sec, 1),
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
            kraftreiz          = ("kraftreiz",          "sum"),
        )
        .reset_index()
    )

    return df_daily, df_weekly
