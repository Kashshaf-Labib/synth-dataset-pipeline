"""
View Splitter — detects multi-view radiology reports and splits a
StructuredRadiologyPrompt into separate per-view prompts.

For example, a report describing "PA and Lateral" views will be split
into two prompts: one for the PA view and one for the Lateral view,
each generating a separate image.

When no multi-view pattern is detected, the original prompt is returned
unchanged in a single-element list (standard fallback).
"""

from __future__ import annotations

import re
from typing import Optional

from .models import StructuredRadiologyPrompt


# ── Regex patterns for detecting individual view types ────────────────────────

# Frontal views: PA (posteroanterior), AP (anteroposterior), or "Frontal"
_FRONTAL_PATTERN = re.compile(r"\b(PA|AP|Frontal)\b", re.IGNORECASE)

# Lateral views: "Lateral" or the abbreviation "LAT" (not inside "LATERAL")
_LATERAL_PATTERN = re.compile(r"\b(Lateral|LAT)\b", re.IGNORECASE)

# Multi-view phrase used for cleaning modality text
# Matches patterns like "PA and lateral", "AP/lateral", "Frontal, lateral"
_MULTI_VIEW_PHRASE = re.compile(
    r"\b(PA|AP|Frontal)\s*(?:and|[/,])\s*(Lateral|LAT)\b",
    re.IGNORECASE,
)


def _normalize_frontal(matched_text: str) -> str:
    """Normalize a frontal-view match to a consistent label."""
    upper = matched_text.upper()
    if upper in ("PA", "AP"):
        return upper
    return "Frontal"


def detect_views(text: str) -> list[str]:
    """
    Detect individual imaging views from a text description.

    Looks for the co-occurrence of a frontal view indicator (PA, AP, Frontal)
    AND a lateral view indicator (Lateral, LAT). Both must be present to
    indicate a multi-view report.

    Parameters
    ----------
    text : the image/modality description string (e.g. "Xray Chest PA and Lateral")

    Returns
    -------
    list[str] — e.g. ["PA", "Lateral"] if multi-view; empty list otherwise.
    """
    if not text or not text.strip():
        return []

    frontal_match = _FRONTAL_PATTERN.search(text)
    lateral_match = _LATERAL_PATTERN.search(text)

    if frontal_match and lateral_match:
        frontal_label = _normalize_frontal(frontal_match.group(1))
        return [frontal_label, "Lateral"]

    # Only one type found (or neither) → not a multi-view report
    return []


def _simplify_modality(modality: str, view: str) -> str:
    """
    Replace multi-view references in a modality string with a single view name.

    Example:
        _simplify_modality("PA and lateral chest X-ray", "PA")
        → "PA chest X-ray"
    """
    return _MULTI_VIEW_PHRASE.sub(view, modality)


def split_prompt_by_views(
    prompt: StructuredRadiologyPrompt,
    raw_image_text: str,
) -> list[tuple[Optional[str], StructuredRadiologyPrompt]]:
    """
    Split a StructuredRadiologyPrompt into per-view prompts.

    Detection priority:
      1. The raw ``image`` column from the CSV (most reliable source).
      2. The LLM-extracted ``plane_view`` field (fallback).

    Parameters
    ----------
    prompt          : the structured prompt extracted by the LLM
    raw_image_text  : the raw ``image`` column value from the CSV report

    Returns
    -------
    list of (view_name, modified_prompt) tuples.
    - If multi-view is detected:  e.g. [("PA", sp1), ("Lateral", sp2)]
    - If single/no view:          [(None, original_prompt)]
    """
    # Try the raw CSV text first (most explicit)
    views = detect_views(raw_image_text)

    # Fallback: try the LLM-extracted plane_view field
    if not views and prompt.plane_view:
        views = detect_views(prompt.plane_view)

    # No multi-view detected → return the original prompt unchanged
    if len(views) <= 1:
        return [(None, prompt)]

    # Multi-view detected → create one prompt per view
    result: list[tuple[Optional[str], StructuredRadiologyPrompt]] = []

    for view in views:
        modified = prompt.model_copy(deep=True)
        modified.view = view
        modified.plane_view = f"{view} view"

        # Clean up the modality text to reference only this view
        if modified.modality:
            modified.modality = _simplify_modality(modified.modality, view)

        result.append((view, modified))

    return result
