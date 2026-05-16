import re
from pathlib import Path

import pandas as pd
from docx import Document


def parse_training_log(file_path: str | Path) -> pd.DataFrame:
    """Parse a training-log .docx file into a tidy DataFrame.

    Each activity block becomes one row.  Key-value pairs (e.g. "Duration [min]: 45")
    become columns; bare labels (Sport / Training-Type headers) are captured
    positionally as 'sport' and 'training_type'.

    Parameters
    ----------
    file_path : path to the .docx training-log file

    Returns
    -------
    DataFrame with columns: date, activity_nr, sport, training_type, …
    """
    doc = Document(str(file_path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    records: list[dict] = []
    current_date: str | None = None
    current_activity_nr: str | None = None
    current_sport: str | None = None
    current_training_type: str | None = None
    current_kv: dict = {}

    def flush() -> None:
        if current_date and current_activity_nr:
            row: dict = {
                "date":          current_date,
                "activity_nr":   current_activity_nr,
                "sport":         current_sport,
                "training_type": current_training_type,
            }
            row.update(current_kv)
            records.append(row)

    activity_re   = re.compile(r'^Activity\s+\d+\s*:$', re.IGNORECASE)
    bare_label_re = re.compile(r'^[^\[\]:\n]+:$')
    kv_re         = re.compile(r'^(.+?)\s*(?:\[([^\]]+)\])?\s*:\s*(.*)$')

    for line in paragraphs:
        if line.startswith("Date:"):
            flush()
            current_date          = line.split(":", 1)[-1].strip()
            current_activity_nr   = None
            current_sport         = None
            current_training_type = None
            current_kv            = {}

        elif activity_re.match(line):
            flush()
            current_activity_nr   = line.rstrip(":").strip()
            current_sport         = None
            current_training_type = None
            current_kv            = {}

        elif bare_label_re.match(line):
            label = line.rstrip(":").strip()
            if current_sport is None:
                current_sport = label
            elif current_training_type is None:
                current_training_type = label

        else:
            m = kv_re.match(line)
            if m:
                key_raw, unit, value = m.group(1).strip(), m.group(2), m.group(3).strip()
                col = key_raw.lower().replace(" ", "_").replace("-", "_")
                if unit:
                    col = f"{col}_[{unit.lower()}]"
                current_kv[col] = value if value else None

    flush()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y", errors="coerce")
    return df
