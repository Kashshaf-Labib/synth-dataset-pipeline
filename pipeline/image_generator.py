"""
Image Generator — generates synthetic radiology images using DALL-E 3 or
Google Vertex AI Imagen, driven by the formatted text prompt.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from io import BytesIO

from PIL import Image as PILImage

from . import config


# ─── DALL-E 3 ────────────────────────────────────────────────────────────────

def generate_image_dalle(prompt: str, uid: int, max_retries: int = 3) -> Path:
    """
    Generate an image with OpenAI DALL-E 3.

    Returns the path to the saved PNG file.
    """
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    output_path = config.IMAGES_DIR / f"{uid}.png"

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


# ─── Google Vertex AI Imagen ─────────────────────────────────────────────────

def generate_image_gemini(prompt: str, uid: int, max_retries: int = 3) -> Path:
    """
    Generate an image with Google Vertex AI Imagen model.

    Uses Vertex AI SDK with Application Default Credentials (ADC).
    Returns the path to the saved PNG file.
    """
    import vertexai
    from vertexai.preview.vision_models import ImageGenerationModel

    # Initialize Vertex AI with project/location from config
    vertexai.init(project=config.GCP_PROJECT_ID, location=config.GCP_LOCATION)

    output_path = config.IMAGES_DIR / f"{uid}.png"

    # Use Imagen model via Vertex AI
    imagen_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")

    for attempt in range(1, max_retries + 1):
        try:
            result = imagen_model.generate_images(
                prompt=prompt,
                number_of_images=1,
                aspect_ratio="1:1",
            )

            # Save the first generated image
            result.images[0].save(str(output_path))
            return output_path

        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  ⚠ Vertex AI attempt {attempt} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Vertex AI image generation failed for uid={uid} after {max_retries} attempts: {e}"
                ) from e

    return output_path  # unreachable


# ─── Dispatcher ──────────────────────────────────────────────────────────────

def generate_image(prompt: str, uid: int, generator: str | None = None) -> Path:
    """
    Generate an image using the configured (or specified) backend.

    Parameters
    ----------
    prompt    : the formatted image generation prompt
    uid       : report UID, used for filename
    generator : "dalle" or "gemini" (overrides config)

    Returns
    -------
    Path to the saved image file.
    """
    gen = (generator or config.IMAGE_GENERATOR).lower()

    if gen == "dalle":
        return generate_image_dalle(prompt, uid)
    elif gen == "gemini":
        return generate_image_gemini(prompt, uid)
    else:
        raise ValueError(f"Unknown image generator: {gen}. Use 'dalle' or 'gemini'.")
