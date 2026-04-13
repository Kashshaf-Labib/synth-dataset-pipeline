"""
Pipeline Orchestrator — end-to-end flow:

  CSV  →  Report Parser  →  Structured Prompt (Gemini/GPT)  →  Image Prompt  →  DALL-E / Gemini
                         ↑
              Optional: indiana_projections.csv + images_dir
              → reference images passed as multimodal input
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image as PILImage
from tqdm import tqdm

from . import config
from .report_parser import load_projections, load_reports
from .prompt_builder import build_prompt_chain, extract_structured_prompt
from .image_prompt_formatter import format_image_prompt
from .image_generator import compute_best_aspect_ratio, generate_image
from .view_splitter import split_prompt_by_views


def run_pipeline(
    csv_path: str | Path,
    limit: int | None = None,
    generator: str | None = None,
    skip_image_generation: bool = False,
    projections_csv: str | Path | None = None,
    images_dir: str | Path | None = None,
) -> list[dict]:
    """
    Execute the full pipeline.

    Parameters
    ----------
    csv_path              : path to indiana_reports.csv
    limit                 : process only the first N reports
    generator             : "dalle" or "gemini" (overrides config)
    skip_image_generation : if True, stop after prompt generation (useful for testing)
    projections_csv       : optional path to indiana_projections.csv.
                            Required when images_dir is provided.
    images_dir            : optional path to the directory containing the
                            original X-ray PNG images (e.g. on Kaggle:
                            /kaggle/input/.../images_normalized/).
                            When provided along with projections_csv,
                            reference images are passed as multimodal input.

    Returns
    -------
    list of metadata dicts for each processed report
    """
    gen = generator or config.IMAGE_GENERATOR
    images_dir_path = Path(images_dir) if images_dir else None

    # ── Load projections lookup (optional) ───────────────────────────────
    projections = None
    if projections_csv and images_dir_path:
        print(f"🗂  Loading projections from {projections_csv} ...")
        projections = load_projections(projections_csv)
        print(f"   → {len(projections)} uids with projection data\n")
    elif images_dir_path and not projections_csv:
        print("⚠  --images-dir provided but --projections-csv is missing. Reference images disabled.\n")

    # ── Load reports ─────────────────────────────────────────────────────
    print(f"📂 Loading reports from {csv_path} ...")
    records = load_reports(csv_path, limit=limit, projections=projections)
    print(f"   → {len(records)} reports loaded\n")

    # ── Build the LangChain chain once (reused across all records) ────────
    print(f"🤖 Initializing LLM chain (model: {config.CHAT_MODEL}) ...")
    chain = build_prompt_chain()
    print("   → Chain ready\n")

    results: list[dict] = []

    for record in tqdm(records, desc="Processing reports", unit="report"):
        entry: dict = {"uid": record.uid}

        # ── Step 1: Extract structured prompt via LLM ────────────────────
        try:
            structured = extract_structured_prompt(record, chain=chain)
            # Carry reference image metadata into the structured prompt
            structured.reference_images = record.reference_images
            entry["structured_prompt"] = structured.model_dump(
                exclude={"reference_images", "view"}
            )
        except Exception as e:
            print(f"\n  ✗ Failed to extract prompt for uid={record.uid}: {e}")
            entry["error_prompt"] = str(e)
            results.append(entry)
            continue

        # ── Step 1.5: Split by views ─────────────────────────────────────
        view_prompts = split_prompt_by_views(structured, record.image)
        if len(view_prompts) > 1:
            view_names = [v for v, _ in view_prompts]
            tqdm.write(
                f"  📐 uid={record.uid} → {len(view_prompts)} views detected: {view_names}"
            )

        # ── Step 2: Resolve reference image paths (once per report) ──────
        ref_image_paths: list[Path] | None = None
        source_dimensions: tuple[int, int] | None = None
        matched_ratio: str | None = None

        if images_dir_path and record.reference_images:
            resolved = []
            for proj in record.reference_images:
                img_path = images_dir_path / proj.filename
                if img_path.exists():
                    resolved.append(img_path)
                else:
                    tqdm.write(f"  ⚠ Reference image not found: {img_path}")
            if resolved:
                ref_image_paths = resolved
                ref_views = ", ".join(
                    p.projection
                    for p in record.reference_images
                    if (images_dir_path / p.filename).exists()
                )
                tqdm.write(
                    f"  📎 uid={record.uid} → {len(resolved)} reference image(s) ({ref_views})"
                )

                # ── Read original image dimensions ────────────────────────
                try:
                    with PILImage.open(resolved[0]) as ref_img:
                        source_dimensions = ref_img.size  # (width, height)
                    matched_ratio = compute_best_aspect_ratio(*source_dimensions)
                    tqdm.write(
                        f"  📐 uid={record.uid} → source: {source_dimensions[0]}×{source_dimensions[1]}, "
                        f"matched aspect ratio: {matched_ratio}"
                    )
                except Exception as dim_err:
                    tqdm.write(
                        f"  ⚠ Could not read dimensions from {resolved[0].name}: {dim_err}"
                    )

            entry["reference_images"] = [
                {"filename": p.filename, "projection": p.projection}
                for p in record.reference_images
            ]
            if source_dimensions:
                entry["source_dimensions"] = {"width": source_dimensions[0], "height": source_dimensions[1]}
                entry["matched_aspect_ratio"] = matched_ratio

        # ── Propagate dimension info into structured prompt ───────────────
        if source_dimensions and matched_ratio:
            structured.source_dimensions = source_dimensions
            structured.matched_aspect_ratio = matched_ratio

        # ── Step 3: Generate images for each view ────────────────────────
        views_data: list[dict] = []
        for view_name, view_prompt in view_prompts:
            view_entry: dict = {"view": view_name}

            # Format the image generation prompt for this specific view
            image_prompt = format_image_prompt(view_prompt)
            view_entry["image_prompt"] = image_prompt

            if not skip_image_generation:
                try:
                    image_path = generate_image(
                        image_prompt,
                        record.uid,
                        generator=gen,
                        image_paths=ref_image_paths,
                        view_suffix=view_name,
                        source_dimensions=source_dimensions,
                    )
                    view_entry["image_path"] = str(image_path)
                    tqdm.write(
                        f"  ✓ uid={record.uid} [{view_name or 'single'}] → {image_path.name}"
                    )
                except Exception as e:
                    print(
                        f"\n  ✗ Image generation failed for uid={record.uid}"
                        f" [{view_name or 'single'}]: {e}"
                    )
                    view_entry["error_image"] = str(e)
            else:
                tqdm.write(
                    f"  ✓ uid={record.uid} [{view_name or 'single'}]"
                    " → prompt generated (image skipped)"
                )

            views_data.append(view_entry)

        entry["views"] = views_data
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
        "--projections-csv",
        type=str,
        default=None,
        help=(
            "Path to indiana_projections.csv. "
            "Required when --images-dir is provided to enable reference image input."
        ),
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help=(
            "Directory containing original X-ray PNG images. "
            "On Kaggle: /kaggle/input/chest-xrays-indiana-university/images/images_normalized. "
            "When provided with --projections-csv, images are passed as multimodal input to Gemini."
        ),
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
        projections_csv=args.projections_csv,
        images_dir=args.images_dir,
    )


if __name__ == "__main__":
    main()
