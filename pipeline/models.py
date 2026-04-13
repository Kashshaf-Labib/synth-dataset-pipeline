"""
Pydantic models for structured data flowing through the pipeline.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class ImageProjection(BaseModel):
    """A single projection image entry from indiana_projections.csv."""

    filename: str
    projection: str  # e.g. "Frontal", "Lateral"


class ReportRecord(BaseModel):
    """Represents a single row from the Indiana Chest X-ray reports CSV."""

    uid: int
    mesh: str = Field(default="", alias="MeSH")
    problems: str = Field(default="", alias="Problems")
    image: str = Field(default="")
    indication: str = Field(default="")
    comparison: str = Field(default="")
    findings: str = Field(default="")
    impression: str = Field(default="")

    # Optional: populated when indiana_projections.csv is provided
    reference_images: list[ImageProjection] = Field(default_factory=list)

    class Config:
        populate_by_name = True


class StructuredRadiologyPrompt(BaseModel):
    """
    Structured prompt extracted from a radiology report by the LLM.
    Fields that cannot be determined from the report should be None.
    """

    modality: Optional[str] = Field(
        default=None,
        description="Imaging modality and sequence, e.g. 'Chest X-ray', 'PA and lateral radiograph'",
    )
    anatomical_region: Optional[str] = Field(
        default=None,
        description="Primary anatomical region, e.g. 'Chest', 'Thorax', 'Lungs and mediastinum'",
    )
    plane_view: Optional[str] = Field(
        default=None,
        description="Imaging plane, view, or orientation, e.g. 'PA and lateral', 'Frontal and lateral'",
    )
    patient_demographics: Optional[str] = Field(
        default=None,
        description="Patient age and sex if available, e.g. '45-year-old male'",
    )
    findings: Optional[str] = Field(
        default=None,
        description="Structured clinical description of radiological findings from the report",
    )
    impression: Optional[str] = Field(
        default=None,
        description="Radiologist impression / conclusion",
    )
    anatomical_constraints: Optional[str] = Field(
        default=None,
        description="Any anatomical constraints or notes about surrounding structures",
    )
    imaging_characteristics: Optional[str] = Field(
        default=None,
        description="Specific imaging characteristics like tissue density, contrast patterns",
    )

    # Set by the view splitter (not the LLM) when a multi-view report is
    # split into separate per-view prompts (e.g. "PA", "Lateral").
    view: Optional[str] = Field(
        default=None,
        description="Specific view for this prompt when split from a multi-view report",
    )

    # Carries reference image info through to prompt formatting and generation
    reference_images: list[ImageProjection] = Field(default_factory=list)

    # Original image dimensions and matched API aspect ratio — set by the
    # pipeline when reference images are available, used to request the
    # correct aspect ratio from the generation API (no post-processing resize).
    source_dimensions: Optional[tuple[int, int]] = Field(
        default=None,
        description="(width, height) of the original reference image",
    )
    matched_aspect_ratio: Optional[str] = Field(
        default=None,
        description="Closest API-supported aspect ratio string, e.g. '3:4'",
    )