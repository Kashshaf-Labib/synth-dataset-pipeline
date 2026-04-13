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
    When reference_images are present, a [REFERENCE IMAGES] section is
    prepended to inform the model to treat the attached images as
    anatomical/structural references.
    """
    sections: list[str] = []

    # ── REFERENCE IMAGES (when images are provided) ──────────────────────
    if sp.reference_images:
        view_list = ", ".join(
            f"{img.projection} ({img.filename})" for img in sp.reference_images
        )
        sections.append(
            "[REFERENCE IMAGES]\n"
            f"The following real chest X-ray images are provided as anatomical "
            f"and structural reference ({view_list}). "
            "Use these images to accurately replicate the patient's anatomy, "
            "body habitus, and image geometry. Apply the clinical findings "
            "described below to generate a realistic synthetic variant."
        )

    # ── MODALITY CONDITIONING ────────────────────────────────────────────
    if sp.modality or sp.plane_view:
        modality_str = sp.modality or "Medical imaging"
        view_str = sp.plane_view or "standard clinical view"
        sections.append(
            "[MODALITY CONDITIONING]\n"
            f"Radiology-grade {modality_str}, {view_str}, clinical "
            "diagnostic imaging acquired using real-world hospital protocols."
        )

    # ── SINGLE VIEW DIRECTIVE (when split from a multi-view report) ──────
    if sp.view:
        sections.append(
            "[SINGLE VIEW DIRECTIVE]\n"
            f"Generate ONLY a single {sp.view} view radiograph. "
            "Do NOT combine multiple views or projections side by side "
            "in one image. The output must show exactly one projection."
        )

    # ── IMAGE GEOMETRY (when source dimensions are known) ───────────────
    if sp.matched_aspect_ratio and sp.source_dimensions:
        w, h = sp.source_dimensions
        if w > h:
            orientation = "landscape"
        elif h > w:
            orientation = "portrait"
        else:
            orientation = "square"
        sections.append(
            "[IMAGE GEOMETRY]\n"
            f"Generate a {orientation}-orientation radiograph matching a "
            f"{sp.matched_aspect_ratio} aspect ratio "
            f"(original source: {w}×{h} pixels). "
            "Preserve the spatial proportions and field-of-view geometry "
            "of the original image."
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
    # NOTE: impression is intentionally excluded — it must NOT appear as
    # rendered text or an overlay in the image. Only radiological findings
    # are used to condition the visual appearance of the X-ray.
    if sp.findings:
        sections.append(
            "[TEXTUAL PROMPTING — CLINICAL DESCRIPTION]\n"
            f"Radiological findings include: {sp.findings}"
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

    # ── NEGATIVE / EXCLUSION CONSTRAINTS ────────────────────────────────
    sections.append(
        "[NEGATIVE CONSTRAINTS — STRICT]\n"
        "Do NOT render any text, letters, words, numbers, labels, captions, "
        "annotations, impression boxes, report panels, diagnostic summaries, "
        "overlaid text blocks, scrollbars, UI elements, or any written content "
        "anywhere in the image. The output must be a pure radiograph image only — "
        "no impression section, no findings text, no header bars with hospital "
        "names or dates, no patient information overlays. "
        "The image must look exactly like a raw, unadorned clinical X-ray film "
        "as it appears on a radiologist's lightbox or PACS viewer without any "
        "annotation layer."
    )

    return "\n\n".join(sections)
