"""
Image Prompt Formatter — assembles a StructuredRadiologyPrompt into a
final text prompt ready for an image generation model.
"""

from __future__ import annotations

from .models import StructuredRadiologyPrompt


def format_image_prompt(sp: StructuredRadiologyPrompt) -> str:
    """
    Convert the structured prompt into the multi-section text prompt
    used by DALL-E / Gemini.

    Sections whose source data is None are omitted entirely.
    """
    sections: list[str] = []

    # ── MODALITY CONDITIONING ────────────────────────────────────────────
    if sp.modality or sp.plane_view:
        modality_str = sp.modality or "Medical imaging"
        view_str = sp.plane_view or "standard clinical view"
        sections.append(
            "[MODALITY CONDITIONING]\n"
            f"Radiology-grade {modality_str}, {view_str}, clinical "
            "diagnostic imaging acquired using real-world hospital protocols."
        )

    # ── ANATOMICAL PROMPTING ─────────────────────────────────────────────
    if sp.anatomical_region:
        sections.append(
            "[ANATOMICAL PROMPTING]\n"
            f"Focused on {sp.anatomical_region}, medically accurate human anatomy, "
            "correct spatial relationships between structures, normal anatomical proportions, "
            "realistic organ boundaries, symmetry where appropriate, no duplicated or missing organs."
        )

    # ── METADATA CONDITIONING ────────────────────────────────────────────
    if sp.patient_demographics:
        sections.append(
            "[METADATA CONDITIONING]\n"
            f"Image represents a {sp.patient_demographics} patient, with clinically "
            "realistic anatomical variations consistent with demographics."
        )

    # ── TEXTUAL PROMPTING — CLINICAL DESCRIPTION ─────────────────────────
    if sp.findings or sp.impression:
        parts: list[str] = []
        if sp.findings:
            parts.append(f"Radiological findings include: {sp.findings}")
        if sp.impression:
            parts.append(f"Impression: {sp.impression}")
        sections.append(
            "[TEXTUAL PROMPTING — CLINICAL DESCRIPTION]\n" + " ".join(parts)
        )

    # ── ANATOMICAL CONSTRAINTS ───────────────────────────────────────────
    constraints_text = sp.anatomical_constraints or (
        "Surrounding structures remain anatomically correct unless clinically affected"
    )
    sections.append(
        "[ANATOMICAL CONSTRAINTS]\n"
        f"{constraints_text}, no unrealistic deformation, "
        "physiologically plausible morphology."
    )

    # ── IMAGING CHARACTERISTICS ──────────────────────────────────────────
    characteristics_text = sp.imaging_characteristics or ""
    base_characteristics = (
        "realistic tissue density and contrast, hospital PACS radiology appearance, "
        "diagnostic grayscale medical imaging, consistent with real-world clinical datasets"
    )
    if characteristics_text:
        full_characteristics = f"{characteristics_text}. {base_characteristics}"
    else:
        full_characteristics = base_characteristics

    sections.append(
        "[IMAGING CHARACTERISTICS]\n" + full_characteristics + "."
    )

    return "\n\n".join(sections)
