"""
Image Generator — generates synthetic radiology images using DALL-E 3 or
Google Vertex AI Gemini, driven by the formatted text prompt.
Supports optional reference images as multimodal input for Gemini.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from io import BytesIO

from PIL import Image as PILImage

from . import config


# ─── DALL-E 3 ────────────────────────────────────────────────────────────────

def generate_image_dalle(
    prompt: str,
    uid: int,
    view_suffix: str | None = None,
    max_retries: int = 3,
) -> Path:
    """
    Generate an image with OpenAI DALL-E 3.

    Returns the path to the saved PNG file.
    """
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    filename = f"{uid}_{view_suffix}.png" if view_suffix else f"{uid}.png"
    output_path = config.IMAGES_DIR / filename

    for attempt in range(1, max_retries + 1):
        try:
            response = client.images.generate(
                model=config.DALLE_MODEL,
                prompt=prompt,
                size=config.DALLE_IMAGE_SIZE,
                quality=config.DALLE_IMAGE_QUALITY,
                response_format="b64_json",
                n=1,
            )
            image_data = base64.b64decode(response.data[0].b64_json)
            img = PILImage.open(BytesIO(image_data))
            img.save(output_path)
            return output_path

        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  ⚠ DALL-E attempt {attempt} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"DALL-E generation failed for uid={uid} after {max_retries} attempts: {e}"
                ) from e

    return output_path  # unreachable but satisfies type checker


# ─── Google Vertex AI Gemini Image Generation ────────────────────────────────

def generate_image_gemini(
    prompt: str,
    uid: int,
    image_paths: list[Path] | None = None,
    view_suffix: str | None = None,
    max_retries: int = 3,
) -> Path:
    """
    Generate an image with Google Gemini's native image generation via Vertex AI.

    Uses the google.genai Client with vertexai=True and the
    gemini-2.5-flash-image model with response_modalities=["TEXT", "IMAGE"].

    Parameters
    ----------
    prompt      : the formatted text prompt
    uid         : report UID, used for output filename
    image_paths : optional list of reference image Paths to include as
                  multimodal input. When provided, images are prepended
                  to the content as inline Parts before the text prompt.
    max_retries : number of retry attempts on failure

    Returns
    -------
    Path to the saved PNG file.
    """
    from google import genai
    from google.genai import types as genai_types

    client = genai.Client(
        vertexai=True,
        project=config.GCP_PROJECT_ID,
        location=config.GCP_IMAGE_LOCATION,
    )

    model_id = config.GEMINI_IMAGE_MODEL
    filename = f"{uid}_{view_suffix}.png" if view_suffix else f"{uid}.png"
    output_path = config.IMAGES_DIR / filename

    # ── Build multimodal contents ─────────────────────────────────────────
    if image_paths:
        contents: list = []
        for img_path in image_paths:
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            # Determine MIME type from extension
            suffix = img_path.suffix.lower()
            mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
            mime_type = mime_map.get(suffix, "image/png")
            contents.append(
                genai_types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
            )
        # Append text prompt as a final Part
        contents.append(genai_types.Part.from_text(text=prompt))
        ref_count = len(image_paths)
    else:
        # Text-only: pass as a plain string (same as before)
        contents = prompt
        ref_count = 0

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    candidate_count=1,
                ),
            )

            # Extract image data from response parts
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    img = PILImage.open(BytesIO(part.inline_data.data))
                    img.save(output_path)
                    ref_info = f", {ref_count} reference image(s)" if ref_count else ""
                    print(f"  ✓ Image saved: {output_path.name} (model: {model_id}{ref_info})")
                    return output_path

            raise RuntimeError("No image data found in response")

        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  ⚠ Gemini attempt {attempt} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Gemini image generation failed for uid={uid} after {max_retries} attempts: {e}"
                ) from e

    return output_path  # unreachable


# ─── Dispatcher ──────────────────────────────────────────────────────────────

def generate_image(
    prompt: str,
    uid: int,
    generator: str | None = None,
    image_paths: list[Path] | None = None,
    view_suffix: str | None = None,
) -> Path:
    """
    Generate an image using the configured (or specified) backend.

    Parameters
    ----------
    prompt      : the formatted image generation prompt
    uid         : report UID, used for filename
    generator   : "dalle" or "gemini" (overrides config)
    image_paths : optional reference image paths (Gemini only)

    Returns
    -------
    Path to the saved image file.
    """
    gen = (generator or config.IMAGE_GENERATOR).lower()

    if gen == "dalle":
        if image_paths:
            print(f"  ⚠ Reference images are not supported for DALL-E, ignoring {len(image_paths)} image(s).")
        return generate_image_dalle(prompt, uid, view_suffix=view_suffix)
    elif gen == "gemini":
        return generate_image_gemini(prompt, uid, image_paths=image_paths, view_suffix=view_suffix)
    else:
        raise ValueError(f"Unknown image generator: {gen}. Use 'dalle' or 'gemini'.")
