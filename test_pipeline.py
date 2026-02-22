"""Quick verification script for the pipeline modules."""

# 1. Config
from pipeline.config import IMAGES_DIR, OUTPUT_DIR, CHAT_MODEL
print(f"[OK] config — CHAT_MODEL={CHAT_MODEL}, IMAGES_DIR={IMAGES_DIR}")

# 2. Models
from pipeline.models import ReportRecord, StructuredRadiologyPrompt
print("[OK] models")

# 3. Report parser
from pipeline.report_parser import load_reports
records = load_reports("indiana_reports.csv", limit=3)
assert len(records) == 3, f"Expected 3, got {len(records)}"
print(f"[OK] report_parser — loaded {len(records)} records")
print(f"     Sample: uid={records[0].uid}, findings='{records[0].findings[:60]}...'")

# 4. Image prompt formatter
from pipeline.image_prompt_formatter import format_image_prompt
sp = StructuredRadiologyPrompt(
    modality="Chest X-ray",
    anatomical_region="Chest and lungs",
    plane_view="PA and lateral views",
    findings="Normal heart size. No pleural effusion.",
    impression="No acute disease.",
)
prompt = format_image_prompt(sp)
assert "[MODALITY CONDITIONING]" in prompt
assert "[ANATOMICAL PROMPTING]" in prompt
assert "[TEXTUAL PROMPTING" in prompt
print(f"[OK] image_prompt_formatter — {len(prompt)} chars")
print()
print("=== SAMPLE FORMATTED PROMPT ===")
print(prompt)

# 5. Image generator import (no API call)
from pipeline.image_generator import generate_image
print()
print("[OK] image_generator — imported")

# 6. Pipeline orchestrator import
from pipeline.pipeline import run_pipeline
print("[OK] pipeline — imported")

print()
print("=== ALL MODULES VERIFIED SUCCESSFULLY ===")
