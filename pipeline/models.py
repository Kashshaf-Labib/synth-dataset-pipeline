"""
Pydantic models for structured data flowing through the pipeline.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


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