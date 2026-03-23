"""
Prompt Builder — uses LangChain + Vertex AI Gemini (or a custom Vertex AI
endpoint) to extract a StructuredRadiologyPrompt from a raw radiology report.
"""

from __future__ import annotations

import json
import re

from langchain_core.prompts import ChatPromptTemplate

from .config import (
    OPENAI_API_KEY,
    GCP_PROJECT_ID,
    GCP_LOCATION,
    CHAT_MODEL,
    VERTEX_ENDPOINT_ID,
)
from .models import ReportRecord, StructuredRadiologyPrompt


# ── System prompt for structured extraction ──────────────────────────────────
_SYSTEM_PROMPT = """\
You are a radiology AI assistant. Given a radiology report, extract the following
properties into a structured JSON object. Only include properties that are
explicitly mentioned or can be directly inferred from the report. If a property
is not present in the report, set it to null — do NOT make up information.

Properties to extract:
1. **modality** — The imaging modality and sequence (e.g. "Chest X-ray", "PA and lateral radiograph").
2. **anatomical_region** — The primary anatomical region being imaged (e.g. "Chest", "Thorax").
3. **plane_view** — The imaging plane, view, or orientation (e.g. "PA and lateral", "Frontal and lateral views").
4. **patient_demographics** — Patient age and sex if mentioned (e.g. "65-year-old female"). De-identified placeholders like "XXXX-year-old XXXX" should be set to null.
5. **findings** — The clinical findings as described in the report. Preserve medical terminology. Clean up any "XXXX" placeholders.
6. **impression** — The radiologist's impression / conclusion.
7. **anatomical_constraints** — Any notes about anatomical constraints, surrounding structures, or spatial relationships mentioned.
8. **imaging_characteristics** — Any specific imaging characteristics such as tissue density, contrast patterns, or appearance descriptors.

Return valid JSON matching the schema exactly.
"""

_USER_TEMPLATE = """\
=== RADIOLOGY REPORT (UID: {uid}) ===

Image/Modality: {image}
Indication: {indication}
Comparison: {comparison}
MeSH Terms: {mesh}
Problems: {problems}

FINDINGS:
{findings}

IMPRESSION:
{impression}
"""


def _is_vertex_model(model_name: str) -> bool:
    """Check if the model name refers to a Google / Vertex AI model."""
    vertex_prefixes = ("gemini", "medlm", "med-palm", "medpalm", "claude", "google/")
    return model_name.lower().startswith(vertex_prefixes)


def _use_custom_endpoint() -> bool:
    """Check if a custom Vertex AI endpoint should be used."""
    return bool(VERTEX_ENDPOINT_ID)


def _extract_json_from_text(text: str) -> dict:
    """Extract a JSON object from model response text (handles markdown fences)."""
    # Try to find JSON in markdown code blocks first
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1).strip())
    # Try direct JSON parse
    # Find the first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return json.loads(text[start : end + 1])
    raise ValueError(f"Could not extract JSON from response: {text[:200]}")


# ─── Standard LangChain path (Gemini / OpenAI) ──────────────────────────────

def build_prompt_chain():
    """
    Create a LangChain chain that extracts a StructuredRadiologyPrompt
    from a ReportRecord.  Automatically selects OpenAI or Vertex AI Gemini
    based on the CHAT_MODEL config value.

    Returns None if a custom endpoint is configured (handled separately).
    """
    if _use_custom_endpoint():
        # Custom endpoint — handled by extract_structured_prompt_endpoint()
        return None

    if _is_vertex_model(CHAT_MODEL):
        from langchain_google_vertexai import ChatVertexAI

        llm = ChatVertexAI(
            model_name=CHAT_MODEL,
            project=GCP_PROJECT_ID,
            location=GCP_LOCATION,
            temperature=0.0,
        )
    else:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=CHAT_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0.0,
        )

    # Use structured output to guarantee valid Pydantic model
    structured_llm = llm.with_structured_output(StructuredRadiologyPrompt)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("human", _USER_TEMPLATE),
        ]
    )

    chain = prompt | structured_llm
    return chain


# ─── Custom Vertex AI Endpoint path ─────────────────────────────────────────

def _call_vertex_endpoint(user_message: str) -> str:
    """Send a prompt to the custom MedGemma Vertex AI endpoint and return raw text."""
    from google.cloud import aiplatform

    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_LOCATION, api_transport="rest")
    endpoint = aiplatform.Endpoint(endpoint_name=VERTEX_ENDPOINT_ID, location=GCP_LOCATION)

    messages = [
        {
            "role": "system",
            "content": _SYSTEM_PROMPT + "\n\nReturn ONLY valid JSON, no other text.",
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

    instances = [
        {
            "@requestFormat": "chatCompletions",
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0,
        },
    ]

    response = endpoint.predict(
        instances=instances, use_dedicated_endpoint=True
    )

    # Response format: predictions["choices"][0]["message"]["content"]
    raw_text = response.predictions["choices"][0]["message"]["content"]
    print(f"  [DEBUG] Endpoint raw response (first 300 chars): {raw_text[:300]}")
    return raw_text


def extract_structured_prompt_endpoint(record: ReportRecord) -> StructuredRadiologyPrompt:
    """Extract structured prompt using a custom deployed Vertex AI endpoint."""
    user_msg = _USER_TEMPLATE.format(
        uid=record.uid,
        image=record.image,
        indication=record.indication,
        comparison=record.comparison,
        mesh=record.mesh,
        problems=record.problems,
        findings=record.findings,
        impression=record.impression,
    )

    raw_response = _call_vertex_endpoint(user_msg)
    parsed = _extract_json_from_text(raw_response)
    return StructuredRadiologyPrompt(**parsed)


# ─── Unified entry point ────────────────────────────────────────────────────

def extract_structured_prompt(
    record: ReportRecord,
    chain=None,
) -> StructuredRadiologyPrompt:
    """
    Given a ReportRecord, call the LLM to produce a StructuredRadiologyPrompt.

    Parameters
    ----------
    record : ReportRecord
    chain  : optional pre-built chain (avoids re-creating per call).
             Ignored when using a custom endpoint.

    Returns
    -------
    StructuredRadiologyPrompt
    """
    # Custom endpoint path
    if _use_custom_endpoint():
        return extract_structured_prompt_endpoint(record)

    # Standard LangChain path
    if chain is None:
        chain = build_prompt_chain()

    result = chain.invoke(
        {
            "uid": record.uid,
            "image": record.image,
            "indication": record.indication,
            "comparison": record.comparison,
            "mesh": record.mesh,
            "problems": record.problems,
            "findings": record.findings,
            "impression": record.impression,
        }
    )

    return result

