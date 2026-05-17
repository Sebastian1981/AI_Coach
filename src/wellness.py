import datetime

import pandas as pd


def fetch_wellness_data(client, date_start: str, date_end: str) -> pd.DataFrame:
    """Fetch daily resting HR and nightly HRV from Garmin Connect.

    Parameters
    ----------
    client     : raw Garmin instance (GarminClient.raw)
    date_start : start date "YYYY-MM-DD"
    date_end   : end date   "YYYY-MM-DD"

    Returns
    -------
    DataFrame with columns: date, resting_hr_bpm, max_hr_bpm,
        hrv_last_night_avg, hrv_last_night_5min_high, hrv_weekly_avg, hrv_status
    """
    current = datetime.date.fromisoformat(date_start)
    end     = datetime.date.fromisoformat(date_end)
    rows = []

    while current <= end:
        date_str = current.isoformat()
        row: dict = {"date": pd.Timestamp(date_str)}

        try:
            stats = client.get_stats(date_str)
            row["resting_hr_bpm"] = stats.get("restingHeartRate")
            row["max_hr_bpm"]     = stats.get("maxHeartRate")
        except Exception:
            row["resting_hr_bpm"] = None
            row["max_hr_bpm"]     = None

        try:
            hrv_data = client.get_hrv_data(date_str)
            summary  = (hrv_data or {}).get("hrvSummary", {})
            row["hrv_last_night_avg"]       = summary.get("lastNightAvg")
            row["hrv_last_night_5min_high"] = summary.get("lastNight5MinHigh")
            row["hrv_weekly_avg"]           = summary.get("weeklyAvg")
            row["hrv_status"]               = summary.get("status")
        except Exception:
            row["hrv_last_night_avg"]       = None
            row["hrv_last_night_5min_high"] = None
            row["hrv_weekly_avg"]           = None
            row["hrv_status"]               = None

        rows.append(row)
        current += datetime.timedelta(days=1)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df
