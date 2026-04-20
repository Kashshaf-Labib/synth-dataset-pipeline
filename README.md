# Synthetic Radiology Image Generation Pipeline

A modular, end-to-end pipeline for generating clinically conditioned synthetic chest X-ray images from structured radiology reports. The system leverages large language models (LLMs) for structured information extraction and state-of-the-art generative models (DALL-E 3, Gemini) for radiograph synthesis, with optional multimodal reference image conditioning.

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Architecture](#architecture)
- [Pipeline Stages](#pipeline-stages)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Basic Execution](#basic-execution)
  - [Advanced Options](#advanced-options)
  - [Kaggle Environment](#kaggle-environment)
- [Data Requirements](#data-requirements)
- [Module Reference](#module-reference)
  - [Report Parser](#report-parser)
  - [Prompt Builder](#prompt-builder)
  - [View Splitter](#view-splitter)
  - [Image Prompt Formatter](#image-prompt-formatter)
  - [Image Generator](#image-generator)
  - [Pipeline Orchestrator](#pipeline-orchestrator)
- [Output Format](#output-format)
- [Aspect Ratio Preservation](#aspect-ratio-preservation)
- [Testing](#testing)
- [Limitations](#limitations)
- [License](#license)

---

## Overview

This pipeline automates the generation of synthetic chest X-ray images conditioned on structured clinical findings extracted from real radiology reports. It is designed for use in research contexts where paired real/synthetic medical image datasets are needed, such as training and evaluating synthetic image detection models.

The system processes reports from the Indiana University Chest X-ray Collection, extracts structured radiological attributes via an LLM, composes domain-specific image generation prompts, and produces synthetic radiograph images through either OpenAI DALL-E 3 or Google Gemini image generation APIs.

## Motivation

Synthetic medical image datasets are increasingly important for:

- **Synthetic image detection research**: Training classifiers to distinguish AI-generated medical images from authentic clinical acquisitions.
- **Data augmentation**: Expanding limited medical imaging datasets while preserving clinical fidelity.
- **Privacy-preserving research**: Generating de-identified synthetic alternatives to sensitive patient data.
- **Model robustness evaluation**: Stress-testing diagnostic systems against adversarial synthetic inputs.

This pipeline addresses the need for a systematic, reproducible method to generate clinically coherent synthetic radiographs at scale, with full provenance tracking from source report to generated image.

---

## Architecture

The pipeline follows a sequential, multi-stage processing architecture:

```
indiana_reports.csv
        |
        v
+------------------+
|  Report Parser   |  --> ReportRecord objects (Pydantic models)
+------------------+
        |
        v
+------------------+
|  Prompt Builder  |  --> StructuredRadiologyPrompt (LLM extraction via LangChain)
+------------------+
        |
        v
+------------------+
|  View Splitter   |  --> Per-view prompts (PA, Lateral, etc.)
+------------------+
        |
        v
+------------------+
|  Image Prompt    |  --> Formatted multi-section text prompt
|  Formatter       |
+------------------+
        |
        v
+------------------+
| Image Generator  |  --> Synthetic PNG radiograph
| (DALL-E / Gemini)|
+------------------+
        |
        v
  output/images/         output/metadata.json
```

When optional reference images are provided (via `indiana_projections.csv` and an images directory), the pipeline operates in **multimodal mode**: original X-ray images are passed alongside the text prompt to the Gemini API, enabling anatomically guided generation.

---

## Pipeline Stages

### Stage 1: Report Parsing

Raw CSV records are loaded, filtered for clinical content (at least one of `findings` or `impression` must be non-empty), and converted to typed `ReportRecord` Pydantic models. If projection metadata is provided, reference image information is attached to each record.

### Stage 2: Structured Prompt Extraction

Each `ReportRecord` is passed to an LLM (Gemini, GPT-4, or a custom Vertex AI endpoint) via a LangChain chain. The LLM extracts eight structured attributes from free-text radiology reports:

| Attribute                  | Description                                                    |
|----------------------------|----------------------------------------------------------------|
| `modality`                 | Imaging modality and sequence (e.g., "Chest X-ray")            |
| `anatomical_region`        | Primary anatomical region (e.g., "Chest", "Thorax")            |
| `plane_view`               | Imaging plane or view (e.g., "PA and lateral")                 |
| `patient_demographics`     | Patient age and sex, if available                              |
| `findings`                 | Clinical findings in original medical terminology              |
| `impression`               | Radiologist impression / conclusion                            |
| `anatomical_constraints`   | Notes about surrounding structures or spatial relationships    |
| `imaging_characteristics`  | Tissue density, contrast patterns, appearance descriptors      |

The extraction uses structured output parsing with Pydantic validation to guarantee schema conformance.

### Stage 3: View Splitting

Multi-view reports (e.g., "PA and Lateral") are automatically detected via regex pattern matching on both the raw CSV `image` column and the LLM-extracted `plane_view` field. When detected, the structured prompt is split into separate per-view prompts, each generating an independent image. Single-view reports pass through unchanged.

Supported view patterns:
- Frontal views: PA (posteroanterior), AP (anteroposterior), Frontal
- Lateral views: Lateral, LAT

### Stage 4: Image Prompt Formatting

The structured prompt is assembled into a multi-section text prompt optimized for image generation models. The prompt comprises the following sections, each included only when the corresponding data is available:

| Section                               | Purpose                                                            |
|---------------------------------------|--------------------------------------------------------------------|
| `[REFERENCE IMAGES]`                  | Instructions for using attached reference images                   |
| `[MODALITY CONDITIONING]`             | Specifies imaging modality and clinical acquisition protocol       |
| `[SINGLE VIEW DIRECTIVE]`             | Constrains output to a single projection when views are split      |
| `[IMAGE GEOMETRY]`                    | Specifies target aspect ratio and orientation                      |
| `[ANATOMICAL PROMPTING]`              | Defines anatomical region and structural accuracy requirements     |
| `[METADATA CONDITIONING]`             | Patient demographics for anatomically consistent variation         |
| `[TEXTUAL PROMPTING]`                 | Clinical findings to be visually represented                       |
| `[ANATOMICAL CONSTRAINTS]`            | Morphological plausibility constraints                             |
| `[IMAGING CHARACTERISTICS]`           | Tissue density, contrast, and PACS-grade appearance requirements   |
| `[NEGATIVE CONSTRAINTS -- STRICT]`    | Explicit exclusion of text overlays, annotations, and UI elements  |

The `impression` field is intentionally excluded from the image prompt to prevent the generation model from rendering diagnostic text within the radiograph.

### Stage 5: Image Generation

The formatted prompt is dispatched to the selected image generation backend:

- **DALL-E 3** (OpenAI): Text-only prompts; supports configurable image size and quality.
- **Gemini** (Google): Supports both text-only and multimodal prompts with inline reference images and native aspect ratio control.

Both backends implement retry logic with exponential backoff (up to 3 attempts per image).

---

## Repository Structure

```
.
|-- pipeline/                      # Core pipeline package
|   |-- __init__.py                # Package marker
|   |-- __main__.py                # Module entry point (python -m pipeline)
|   |-- config.py                  # Configuration and environment variable loading
|   |-- models.py                  # Pydantic data models (ReportRecord, StructuredRadiologyPrompt)
|   |-- report_parser.py           # CSV parsing and report loading
|   |-- prompt_builder.py          # LLM-based structured prompt extraction (LangChain)
|   |-- view_splitter.py           # Multi-view detection and prompt splitting
|   |-- image_prompt_formatter.py  # Multi-section image generation prompt assembly
|   |-- image_generator.py         # DALL-E 3 and Gemini image generation backends
|   |-- pipeline.py                # End-to-end pipeline orchestrator and CLI
|
|-- indiana_reports.csv            # Source radiology reports (Indiana University dataset)
|-- indiana_projections.csv        # Image projection metadata (uid -> filename, projection)
|-- requirements.txt               # Python dependencies
|-- .env.example                   # Environment variable template
|-- .gitignore                     # Git ignore rules
|-- test_pipeline.py               # Module integration verification script
|-- test_aspect_ratio.py           # Aspect ratio computation unit tests
|-- output/                        # Generated outputs (git-ignored)
|   |-- images/                    # Synthetic radiograph PNGs
|   |-- metadata.json              # Per-report generation metadata
```

---

## Requirements

- Python 3.10 or later
- API access to at least one of:
  - OpenAI API (for DALL-E 3 and/or GPT models)
  - Google Cloud / Vertex AI (for Gemini models)
  - Google AI API key (for Kaggle environments)

### Python Dependencies

| Package                         | Version   | Purpose                                          |
|---------------------------------|-----------|--------------------------------------------------|
| `langchain`                     | >= 0.3.0  | LLM chain orchestration                         |
| `langchain-openai`              | >= 0.3.0  | OpenAI LLM integration                          |
| `langchain-google-genai`        | >= 2.0.0  | Google Generative AI integration (Kaggle)        |
| `langchain-google-vertexai`     | >= 2.0.0  | Google Vertex AI integration (local/cloud)       |
| `openai`                        | >= 1.0.0  | DALL-E 3 image generation                        |
| `google-generativeai`           | >= 0.8.0  | Gemini native image generation                   |
| `google-cloud-aiplatform`       | >= 1.60.0 | Vertex AI endpoint support                       |
| `pandas`                        | >= 2.0.0  | CSV data loading and manipulation                |
| `python-dotenv`                 | >= 1.0.0  | Environment variable management                  |
| `Pillow`                        | >= 10.0.0 | Image I/O and dimension extraction               |
| `pydantic`                      | >= 2.0.0  | Data validation and typed models                 |
| `tqdm`                          | >= 4.60.0 | Progress bar display                             |

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Kashshaf-Labib/synth-dataset-pipeline.git
   cd synth-dataset-pipeline
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # Linux / macOS
   source .venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and provide the required API keys and configuration values (see [Configuration](#configuration)).

---

## Configuration

All configuration is managed through environment variables, loaded from a `.env` file at the project root. The following table lists all supported variables:

### API Keys

| Variable          | Required | Description                            |
|-------------------|----------|----------------------------------------|
| `OPENAI_API_KEY`  | Conditional | OpenAI API key (required for DALL-E or GPT models) |
| `GOOGLE_API_KEY`  | Conditional | Google AI API key (required for Gemini on Kaggle)   |

### Google Cloud / Vertex AI

| Variable              | Required | Default        | Description                                    |
|-----------------------|----------|----------------|------------------------------------------------|
| `GCP_PROJECT_ID`      | Conditional | --          | Google Cloud project ID (for Vertex AI)        |
| `GCP_LOCATION`        | No       | `us-central1`  | Vertex AI region for LLM inference             |
| `GCP_IMAGE_LOCATION`  | No       | `global`       | Vertex AI region for image generation          |
| `VERTEX_ENDPOINT_ID`  | No       | --             | Custom Vertex AI endpoint (e.g., MedGemma)     |

### Model Selection

| Variable              | Default                  | Options                            |
|-----------------------|--------------------------|------------------------------------|
| `CHAT_MODEL`          | `gemini-2.0-flash`       | Any Gemini, GPT, or custom model   |
| `IMAGE_GENERATOR`     | `gemini`                 | `dalle`, `gemini`                  |
| `GEMINI_IMAGE_MODEL`  | `gemini-2.5-flash-image` | Gemini models with image output    |
| `DALLE_MODEL`         | `dall-e-3`               | `dall-e-3`                         |
| `DALLE_IMAGE_SIZE`    | `1024x1024`              | `1024x1024`, `1792x1024`, `1024x1792` |
| `DALLE_IMAGE_QUALITY` | `standard`               | `standard`, `hd`                   |

### Environment Detection

The pipeline automatically detects Kaggle environments via the `KAGGLE_KERNEL_RUN_TYPE` variable and adjusts its authentication path accordingly (using API key-based `ChatGoogleGenerativeAI` instead of Vertex AI service account authentication).

---

## Usage

### Basic Execution

Run the pipeline as a Python module:

```bash
python -m pipeline --csv indiana_reports.csv --limit 10 --generator gemini
```

### Advanced Options

```
usage: pipeline [-h] [--csv CSV] [--projections-csv PROJECTIONS_CSV]
                [--images-dir IMAGES_DIR] [--limit LIMIT] [--offset OFFSET]
                [--generator {dalle,gemini}] [--skip-images]

Synthetic Radiology Image Generation Pipeline

options:
  --csv CSV                    Path to the Indiana reports CSV file
                               (default: indiana_reports.csv)

  --projections-csv PATH       Path to indiana_projections.csv. Required when
                               --images-dir is provided to enable reference
                               image input.

  --images-dir DIR             Directory containing original X-ray PNG images.
                               When provided with --projections-csv, images
                               are passed as multimodal input to Gemini.

  --limit N                    Limit the number of reports to process
                               (default: all)

  --offset N                   Skip the first N reports in the dataset.
                               Useful for resuming batch processing.

  --generator {dalle,gemini}   Image generator backend
                               (overrides .env config)

  --skip-images                Only generate prompts, skip image generation
                               (useful for testing and prompt evaluation)
```

### Example: Multimodal Generation with Reference Images

```bash
python -m pipeline \
  --csv indiana_reports.csv \
  --projections-csv indiana_projections.csv \
  --images-dir /path/to/images_normalized/ \
  --generator gemini \
  --limit 50
```

In this mode, original X-ray images are attached to the Gemini API request as multimodal input, enabling the model to use anatomical structure from the real images as a reference during synthesis.

### Example: Prompt-Only Mode (No API Calls for Image Generation)

```bash
python -m pipeline --csv indiana_reports.csv --limit 5 --skip-images
```

This generates the structured prompts and formatted image prompts without calling any image generation API, useful for prompt engineering and debugging.

### Example: Batch Processing with Offset

```bash
# Process reports 100-199
python -m pipeline --csv indiana_reports.csv --offset 100 --limit 100

# Process reports 200-299
python -m pipeline --csv indiana_reports.csv --offset 200 --limit 100
```

### Kaggle Environment

On Kaggle, the pipeline automatically:
- Detects the Kaggle runtime via `KAGGLE_KERNEL_RUN_TYPE`
- Uses `ChatGoogleGenerativeAI` with a plain API key (avoiding circular import issues with `kaggle_gcp.py`)
- Uses API key-based `genai.Client` for image generation

A companion notebook (`synth-dataset-notebook.ipynb`) is provided for Kaggle execution.

---

## Data Requirements

### Indiana Chest X-ray Reports (`indiana_reports.csv`)

The primary input is the Indiana University Chest X-ray report dataset. The CSV must contain the following columns:

| Column       | Type   | Description                                          |
|--------------|--------|------------------------------------------------------|
| `uid`        | int    | Unique report identifier                             |
| `MeSH`       | str    | MeSH descriptor terms for the report                 |
| `Problems`   | str    | Clinical problem list                                |
| `image`      | str    | Imaging modality and view description                |
| `indication` | str    | Clinical indication for the study                    |
| `comparison` | str    | Prior comparison studies                             |
| `findings`   | str    | Radiological findings (free text)                    |
| `impression` | str    | Radiologist impression / conclusion (free text)      |

Reports where both `findings` and `impression` are empty are automatically filtered out.

### Projection Metadata (`indiana_projections.csv`) -- Optional

Maps report UIDs to individual image files and their projection types:

| Column       | Type   | Description                                          |
|--------------|--------|------------------------------------------------------|
| `uid`        | int    | Report UID (foreign key to reports CSV)              |
| `filename`   | str    | Image filename (e.g., `CXR1_1_IM-0001-3001.png`)    |
| `projection` | str    | Projection type (e.g., `Frontal`, `Lateral`)         |

---

## Module Reference

### Report Parser

**File:** `pipeline/report_parser.py`

- `load_projections(projections_csv)` -- Builds a `uid -> list[ImageProjection]` lookup dictionary from the projections CSV.
- `load_reports(csv_path, limit, offset, projections)` -- Loads and filters radiology reports, optionally enriching them with reference image metadata.

### Prompt Builder

**File:** `pipeline/prompt_builder.py`

- `build_prompt_chain()` -- Constructs a LangChain chain for structured extraction. Automatically selects between OpenAI, Vertex AI Gemini, or Google Generative AI backends based on the configured model name and runtime environment.
- `extract_structured_prompt(record, chain)` -- Unified entry point that routes to either the LangChain chain or a custom Vertex AI endpoint.
- `extract_structured_prompt_endpoint(record)` -- Directly calls a custom Vertex AI endpoint (e.g., deployed MedGemma) for extraction.

### View Splitter

**File:** `pipeline/view_splitter.py`

- `detect_views(text)` -- Identifies multi-view patterns (frontal + lateral co-occurrence) in modality description strings.
- `split_prompt_by_views(prompt, raw_image_text)` -- Splits a `StructuredRadiologyPrompt` into per-view prompts when multi-view patterns are detected. Returns the original prompt unchanged for single-view reports.

### Image Prompt Formatter

**File:** `pipeline/image_prompt_formatter.py`

- `format_image_prompt(sp)` -- Converts a `StructuredRadiologyPrompt` into a multi-section formatted text prompt for image generation. Handles reference image instructions, modality conditioning, view directives, geometry specifications, anatomical constraints, and strict negative constraints.

### Image Generator

**File:** `pipeline/image_generator.py`

- `compute_best_aspect_ratio(width, height)` -- Finds the closest supported API aspect ratio for given pixel dimensions.
- `generate_image_dalle(prompt, uid, ...)` -- Generates images using OpenAI DALL-E 3 with configurable size and quality.
- `generate_image_gemini(prompt, uid, image_paths, ...)` -- Generates images using Google Gemini with optional multimodal reference image input and native aspect ratio control.
- `generate_image(prompt, uid, generator, ...)` -- Dispatcher that routes to the appropriate backend.

### Pipeline Orchestrator

**File:** `pipeline/pipeline.py`

- `run_pipeline(csv_path, limit, offset, generator, ...)` -- Executes the complete end-to-end pipeline: report loading, LLM extraction, view splitting, prompt formatting, image generation, and metadata persistence.
- `main()` -- CLI entry point with argument parsing.

---

## Output Format

All outputs are written to the `output/` directory:

### Generated Images

Saved to `output/images/` as PNG files with the naming convention:

- Single-view reports: `{uid}.png`
- Multi-view reports: `{uid}_{view}.png` (e.g., `1_PA.png`, `1_Lateral.png`)

### Metadata (`output/metadata.json`)

A JSON array where each element corresponds to one processed report:

```json
{
  "uid": 1,
  "structured_prompt": {
    "modality": "Xray Chest PA and Lateral",
    "anatomical_region": "Chest",
    "plane_view": "PA and Lateral",
    "patient_demographics": null,
    "findings": "The cardiac silhouette and mediastinum size are within normal limits...",
    "impression": "Normal chest x-ray.",
    "anatomical_constraints": null,
    "imaging_characteristics": null
  },
  "reference_images": [
    {"filename": "CXR1_1_IM-0001-3001.png", "projection": "Frontal"}
  ],
  "source_dimensions": {"width": 2048, "height": 2500},
  "matched_aspect_ratio": "4:5",
  "views": [
    {
      "view": "PA",
      "image_prompt": "[MODALITY CONDITIONING]\nRadiology-grade Xray Chest PA...",
      "image_path": "output/images/1_PA.png"
    },
    {
      "view": "Lateral",
      "image_prompt": "[MODALITY CONDITIONING]\nRadiology-grade Xray Chest Lateral...",
      "image_path": "output/images/1_Lateral.png"
    }
  ]
}
```

Fields such as `reference_images`, `source_dimensions`, and `matched_aspect_ratio` are present only when reference images are provided through the multimodal input mode.

---

## Aspect Ratio Preservation

When reference images are provided, the pipeline extracts the original image dimensions and matches them to the closest API-supported aspect ratio. This ratio is passed natively to the generation API, avoiding any post-processing resize that could introduce artifacts.

### Supported Aspect Ratios

**Gemini:**

| Ratio   | Decimal | Orientation |
|---------|---------|-------------|
| `1:1`   | 1.000   | Square      |
| `4:3`   | 1.333   | Landscape   |
| `3:4`   | 0.750   | Portrait    |
| `16:9`  | 1.778   | Landscape   |
| `9:16`  | 0.563   | Portrait    |
| `3:2`   | 1.500   | Landscape   |
| `2:3`   | 0.667   | Portrait    |
| `5:4`   | 1.250   | Landscape   |
| `4:5`   | 0.800   | Portrait    |
| `21:9`  | 2.333   | Ultrawide   |

**DALL-E 3:**

| Size          | Decimal | Orientation |
|---------------|---------|-------------|
| `1024x1024`   | 1.000   | Square      |
| `1792x1024`   | 1.750   | Landscape   |
| `1024x1792`   | 0.571   | Portrait    |

---

## Testing

### Module Verification

Run the integration test to verify all modules load correctly and basic functionality works:

```bash
python test_pipeline.py
```

This script verifies:
- Configuration loading
- Pydantic model instantiation
- CSV report parsing
- Image prompt formatting (section presence assertions)
- Module import integrity for image generator and pipeline orchestrator

### Aspect Ratio Unit Tests

```bash
python test_aspect_ratio.py
```

Validates the `compute_best_aspect_ratio` function against known dimension-to-ratio mappings.

---

## Limitations

- **Image fidelity**: Generated images are conditioned on text descriptions and optional reference images. Clinical accuracy of synthesized pathology is not guaranteed and should not be used for diagnostic purposes.
- **De-identified reports**: The Indiana dataset contains de-identification placeholders (`XXXX`) which may introduce noise into LLM extraction. The prompt builder instructs the LLM to treat these as null values.
- **DALL-E reference images**: Multimodal reference image input is supported only for the Gemini backend. DALL-E operates in text-only mode.
- **Rate limits**: Both OpenAI and Google APIs enforce rate limits. The pipeline includes retry logic with exponential backoff but does not implement global rate throttling across reports.
- **View detection**: The view splitter uses regex-based heuristics for multi-view detection. Reports with non-standard view descriptions may not be correctly split.

---

## License

This project is developed as part of academic thesis research. Please refer to the repository for licensing terms.
