"""
Report Parser — reads the Indiana Chest X-ray reports CSV and returns
a list of ReportRecord objects. Optionally enriches records with
reference image filenames from indiana_projections.csv.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import pandas as pd

from .models import ImageProjection, ReportRecord


def load_projections(
    projections_csv: str | Path,
) -> dict[int, list[ImageProjection]]:
    """
    Load indiana_projections.csv and build a uid → list[ImageProjection] lookup.

    Parameters
    ----------
    projections_csv : path to indiana_projections.csv

    Returns
    -------
    dict mapping uid (int) → list of ImageProjection objects
    """
    df = pd.read_csv(projections_csv)

    lookup: dict[int, list[ImageProjection]] = {}
    for _, row in df.iterrows():
        uid = int(row["uid"])
        proj = ImageProjection(
            filename=str(row["filename"]),
            projection=str(row["projection"]),
        )
        lookup.setdefault(uid, []).append(proj)

    return lookup


def load_reports(
    csv_path: str | Path,
    limit: Optional[int] = None,
    projections: Optional[dict[int, list[ImageProjection]]] = None,
) -> list[ReportRecord]:
    """
    Load radiology reports from the CSV.

    Parameters
    ----------
    csv_path    : path to indiana_reports.csv
    limit       : optional cap on number of records returned
    projections : optional uid→projections lookup from load_projections().
                  When provided, each ReportRecord will have its
                  reference_images field populated.

    Returns
    -------
    list[ReportRecord]  — only rows where at least one of findings/impression is non-empty.
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
        uid = int(row["uid"])
        ref_images = projections.get(uid, []) if projections else []
        records.append(
            ReportRecord(
                uid=uid,
                MeSH=str(row["MeSH"]),
                Problems=str(row["Problems"]),
                image=str(row["image"]),
                indication=str(row["indication"]),
                comparison=str(row["comparison"]),
                findings=str(row["findings"]),
                impression=str(row["impression"]),
                reference_images=ref_images,
            )
        )

    return records
