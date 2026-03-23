"""
Prompt Builder — uses LangChain + Vertex AI Gemini to extract a StructuredRadiologyPrompt
from a raw radiology report.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from .config import OPENAI_API_KEY, GCP_PROJECT_ID, GCP_LOCATION, CHAT_MODEL
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


def _is_gemini_model(model_name: str) -> bool:
    """Check if the model name refers to a Google / Vertex AI model."""
    vertex_prefixes = ("gemini", "medlm", "med-palm", "medpalm", "claude")
    return model_name.lower().startswith(vertex_prefixes)


def build_prompt_chain():
    """
    Create a LangChain chain that extracts a StructuredRadiologyPrompt
    from a ReportRecord.  Automatically selects OpenAI or Vertex AI Gemini
    based on the CHAT_MODEL config value.
    """
    if _is_gemini_model(CHAT_MODEL):
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


def extract_structured_prompt(
    record: ReportRecord,
    chain=None,
) -> StructuredRadiologyPrompt:
    """
    Given a ReportRecord, call the LLM to produce a StructuredRadiologyPrompt.

    Parameters
    ----------
    record : ReportRecord
    chain  : optional pre-built chain (avoids re-creating per call)

    Returns
    -------
    StructuredRadiologyPrompt
    """
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
