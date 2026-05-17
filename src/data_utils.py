import pandas as pd


def expand_timestamp(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    ts = pd.to_datetime(df[col])
    df = df.copy()
    df["jahr"]    = ts.dt.year
    df["monat"]   = ts.dt.month
    df["tag"]     = ts.dt.day
    df["uhrzeit"] = ts.dt.strftime("%H:%M:%S")
    return df


def reorder_columns(df: pd.DataFrame, first_cols: list) -> pd.DataFrame:
    ordered = first_cols + [c for c in df.columns if c not in first_cols]
    return df[ordered]


def missing_data_report(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    missing = df.isna().sum()
    return pd.DataFrame({
        "fehlend":  missing,
        "anteil_%": (missing / total * 100).round(2),
    }).sort_values("anteil_%", ascending=False)


def drop_high_missing(df: pd.DataFrame, threshold: float = 50.0) -> pd.DataFrame:
    missing_pct = df.isna().mean() * 100
    keep = missing_pct[missing_pct <= threshold].index
    dropped = missing_pct[missing_pct > threshold].index.tolist()
    print(f"Gelöschte Spalten ({len(dropped)}): {dropped}")
    return df[keep]
