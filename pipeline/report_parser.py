"""
Report Parser — reads the Indiana Chest X-ray reports CSV and returns
a list of ReportRecord objects.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import pandas as pd

from .models import ReportRecord


def load_reports(
    csv_path: str | Path,
    limit: Optional[int] = None,
) -> list[ReportRecord]:
    """
    Load radiology reports from the CSV.

    Parameters
    ----------
    csv_path : path to indiana_reports.csv
    limit    : optional cap on number of records returned

    Returns
    -------
    list[ReportRecord]  — only rows where at least one of findings/impression
                          is non-empty.
    """
    df = pd.read_csv(csv_path)

    # Fill NaN with empty strings for text columns
    text_cols = ["MeSH", "Problems", "image", "indication", "comparison", "findings", "impression"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # Keep only rows with at least some clinical content
    df = df[
        (df["findings"].str.strip() != "") | (df["impression"].str.strip() != "")
    ].copy()

    if limit is not None:
        df = df.head(limit)

    records: list[ReportRecord] = []
    for _, row in df.iterrows():
        records.append(
            ReportRecord(
                uid=int(row["uid"]),
                MeSH=str(row["MeSH"]),
                Problems=str(row["Problems"]),
                image=str(row["image"]),
                indication=str(row["indication"]),
                comparison=str(row["comparison"]),
                findings=str(row["findings"]),
                impression=str(row["impression"]),
            )
        )

    return records
