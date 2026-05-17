from pathlib import Path

import fitparse
import pandas as pd


def parse_fit_file(fit_path: Path) -> pd.DataFrame:
    """Parse a single .fit file into a DataFrame of record messages.

    Returns time-series 'record' messages with two additional columns
    'sport' and 'sub_sport' from the 'session' message.
    Unknown / vendor-specific fields are dropped.
    """
    fit = fitparse.FitFile(str(fit_path))

    # --- session meta (sport, sub_sport) ---
    sport = None
    sub_sport = None
    for msg in fit.get_messages("session"):
        for data in msg:
            if data.name == "sport":
                sport = str(data.value) if data.value is not None else None
            elif data.name == "sub_sport":
                sub_sport = str(data.value) if data.value is not None else None
        break  # only first session message needed

    # --- time-series records ---
    rows = []
    for msg in fit.get_messages("record"):
        row = {}
        for data in msg:
            if data.name is None:
                continue
            if isinstance(data.name, int):
                continue
            if str(data.name).startswith("unknown"):
                continue
            row[data.name] = data.value
        rows.append(row)

    df = pd.DataFrame(rows)
    df.insert(0, "sport", sport)
    df.insert(1, "sub_sport", sub_sport)
    return df
