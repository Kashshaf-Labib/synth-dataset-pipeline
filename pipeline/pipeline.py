"""
Pipeline Orchestrator — end-to-end flow:

  CSV  →  Report Parser  →  Structured Prompt (ChatGPT)  →  Image Prompt  →  DALL-E / Gemini
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

from . import config
from .report_parser import load_reports
from .prompt_builder import build_prompt_chain, extract_structured_prompt
from .image_prompt_formatter import format_image_prompt
from .image_generator import generate_image


def run_pipeline(
    csv_path: str | Path,
    limit: int | None = None,
    generator: str | None = None,
    skip_image_generation: bool = False,
) -> list[dict]:
    """
    Execute the full pipeline.

    Parameters
    ----------
    csv_path              : path to indiana_reports.csv
    limit                 : process only the first N reports
    generator             : "dalle" or "gemini" (overrides config)
    skip_image_generation : if True, stop after prompt generation (useful for testing)

    Returns
    -------
    list of metadata dicts for each processed report
    """
    gen = generator or config.IMAGE_GENERATOR

    print(f"📂 Loading reports from {csv_path} ...")
    records = load_reports(csv_path, limit=limit)
    print(f"   → {len(records)} reports loaded\n")

    # Build the LangChain chain once (reused across all records)
    print(f"🤖 Initializing ChatGPT chain (model: {config.CHAT_MODEL}) ...")
    chain = build_prompt_chain()
    print("   → Chain ready\n")

    results: list[dict] = []

    for record in tqdm(records, desc="Processing reports", unit="report"):
        entry: dict = {"uid": record.uid}

        # ── Step 1: Extract structured prompt via ChatGPT ────────────────
        try:
            structured = extract_structured_prompt(record, chain=chain)
            entry["structured_prompt"] = structured.model_dump()
        except Exception as e:
            print(f"\n  ✗ Failed to extract prompt for uid={record.uid}: {e}")
            entry["error_prompt"] = str(e)
            results.append(entry)
            continue

        # ── Step 2: Format the image generation prompt ───────────────────
        image_prompt = format_image_prompt(structured)
        entry["image_prompt"] = image_prompt

        # ── Step 3: Generate image ───────────────────────────────────────
        if not skip_image_generation:
            try:
                image_path = generate_image(image_prompt, record.uid, generator=gen)
                entry["image_path"] = str(image_path)
                tqdm.write(f"  ✓ uid={record.uid} → {image_path.name}")
            except Exception as e:
                print(f"\n  ✗ Image generation failed for uid={record.uid}: {e}")
                entry["error_image"] = str(e)
        else:
            tqdm.write(f"  ✓ uid={record.uid} → prompt generated (image skipped)")

        results.append(entry)

    # ── Save metadata ────────────────────────────────────────────────────
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Metadata saved to {config.METADATA_FILE}")
    print(f"🖼  Images saved to {config.IMAGES_DIR}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic Radiology Image Generation Pipeline"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(config.PROJECT_ROOT / "indiana_reports.csv"),
        help="Path to the Indiana reports CSV file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of reports to process (default: all)",
    )
    parser.add_argument(
        "--generator",
        type=str,
        choices=["dalle", "gemini"],
        default=None,
        help="Image generator backend (overrides .env config)",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Only generate prompts, skip image generation (for testing)",
    )

    args = parser.parse_args()

    run_pipeline(
        csv_path=args.csv,
        limit=args.limit,
        generator=args.generator,
        skip_image_generation=args.skip_images,
    )


if __name__ == "__main__":
    main()
