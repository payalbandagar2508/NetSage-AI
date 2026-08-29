"""
llm_engine.py
LLM integration layer for the Network Troubleshooting Diagnosis project.

Uses:
    Groq API
    Model: openai/gpt-oss-120b

Install:
    pip install openai pydantic python-dotenv

.env:
    GROQ_API_KEY=your_groq_api_key
    GROQ_MODEL=openai/gpt-oss-120b
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError


# ---------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROMPT_PATH = BASE_DIR / "diagnose_prompt.md"

load_dotenv()


# ---------------------------------------------------------
# PYDANTIC MODELS
# ---------------------------------------------------------

class Evidence(BaseModel):
    source: Literal[
        "deterministic_checker",
        "show_output",
        "symptom",
        "topology",
        "inference"
    ]

    detail: str


class Diagnosis(BaseModel):
    case_id: str

    diagnosis: str

    fault_category: Literal[
        "VLAN",
        "DHCP",
        "DNS",
        "Routing",
        "ACL",
        "NAT",
        "Wireless",
        "Unknown"
    ]

    severity: Literal[
        "Low",
        "Medium",
        "High",
        "Critical",
        "Unknown"
    ]

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )

    root_cause: str

    explanation: str

    recommended_actions: list[str]

    verification_commands: list[str]

    evidence: list[Evidence]

    deterministic_status: Literal[
        "PASS",
        "FAIL",
        "WARN",
        "NOT_AVAILABLE"
    ]

    needs_human_review: bool


# ---------------------------------------------------------
# LOAD SYSTEM PROMPT
# ---------------------------------------------------------

def load_system_prompt(
    path: Path = PROMPT_PATH
) -> str:

    if not path.exists():
        raise FileNotFoundError(
            f"Diagnosis prompt not found: {path}"
        )

    return path.read_text(
        encoding="utf-8"
    ).strip()


# ---------------------------------------------------------
# BUILD USER INPUT
# ---------------------------------------------------------

def _build_user_input(
    case_id: str,
    symptom: str,
    topology_note: str = "",
    show_outputs: str = "",
    deterministic_results: list[dict[str, Any]] | None = None,
) -> str:

    deterministic_results = deterministic_results or []

    return f"""
CASE ID:
{case_id}

SYMPTOM:
{symptom}

TOPOLOGY / ENVIRONMENT:
{topology_note or "Not provided"}

SHOW COMMAND OUTPUT:
{show_outputs or "Not provided"}

DETERMINISTIC CHECKER RESULTS:
{json.dumps(deterministic_results, indent=2)}

IMPORTANT RULES:

1. Treat the supplied information as evidence.
2. Do not invent IP addresses.
3. Do not invent interfaces.
4. Do not invent routes.
5. Do not invent VLANs.
6. Do not invent configuration.
7. Do not contradict deterministic checker results without explaining why.
8. If the evidence is insufficient, set needs_human_review=true.
9. Return only the requested JSON structure.
""".strip()


# ---------------------------------------------------------
# DIAGNOSE CASE
# ---------------------------------------------------------

def diagnose_case(
    case_id: str,
    symptom: str,
    topology_note: str = "",
    show_outputs: str = "",
    deterministic_results: list[dict[str, Any]] | None = None,
    *,
    client: OpenAI | None = None,
    model: str | None = None,
) -> dict[str, Any]:

    # -----------------------------------------------------
    # GET GROQ API KEY
    # -----------------------------------------------------

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key and client is None:
        raise RuntimeError(
            "GROQ_API_KEY is not set. "
            "Please add GROQ_API_KEY to your .env file."
        )

    # -----------------------------------------------------
    # CREATE GROQ CLIENT
    # -----------------------------------------------------

    if client is None:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )

    # -----------------------------------------------------
    # GET MODEL
    # -----------------------------------------------------

    model = model or os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-120b"
    )

    # -----------------------------------------------------
    # LOAD PROMPT
    # -----------------------------------------------------

    system_prompt = load_system_prompt()

    # -----------------------------------------------------
    # BUILD USER MESSAGE
    # -----------------------------------------------------

    user_input = _build_user_input(
        case_id=case_id,
        symptom=symptom,
        topology_note=topology_note,
        show_outputs=show_outputs,
        deterministic_results=deterministic_results,
    )

    # -----------------------------------------------------
    # CREATE JSON SCHEMA
    # -----------------------------------------------------

    schema = Diagnosis.model_json_schema()

    # -----------------------------------------------------
    # CALL GROQ API
    # -----------------------------------------------------

    try:

        response = client.chat.completions.create(
            model=model,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_input,
                },
            ],

            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "network_diagnosis",
                    "schema": schema,
                    "strict": True,
                },
            },

            temperature=0.2,

            max_completion_tokens=3000,
        )

    except Exception as e:

        raise RuntimeError(
            f"Groq API request failed: {str(e)}"
        ) from e

    # -----------------------------------------------------
    # GET MODEL OUTPUT
    # -----------------------------------------------------

    if not response.choices:
        raise RuntimeError(
            "Groq returned no response choices."
        )

    message = response.choices[0].message

    content = message.content

    if not content:
        raise RuntimeError(
            "Groq returned an empty response."
        )

    # -----------------------------------------------------
    # PARSE JSON
    # -----------------------------------------------------

    try:

        result = json.loads(content)

    except json.JSONDecodeError as e:

        raise RuntimeError(
            f"Groq returned invalid JSON: {e}\n\n"
            f"Raw response:\n{content}"
        ) from e

    # -----------------------------------------------------
    # VALIDATE WITH PYDANTIC
    # -----------------------------------------------------

    try:

        diagnosis = Diagnosis.model_validate(result)

    except ValidationError as e:

        raise RuntimeError(
            f"Diagnosis validation failed:\n{e}"
        ) from e

    # -----------------------------------------------------
    # RETURN NORMAL PYTHON DICTIONARY
    # -----------------------------------------------------

    return diagnosis.model_dump()


# ---------------------------------------------------------
# JSON WRAPPER
# ---------------------------------------------------------

def diagnose_case_json(
    *args: Any,
    **kwargs: Any
) -> str:

    result = diagnose_case(
        *args,
        **kwargs
    )

    return json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    )


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    sample = diagnose_case(

        case_id="CASE_021",

        symptom=(
            "Router-1 cannot reach "
            "Subnet 172.16.2.0/24 behind Router-2."
        ),

        topology_note=(
            "Static routing environment "
            "between Router-1 and Router-2."
        ),

        show_outputs=(
            "Router-1# show ip route\n"
            "S* 0.0.0.0/0 [1/0] via 203.0.113.1\n"
            "10.0.0.0/8 is directly connected\n"
            "(Note: no route for 172.16.2.0/24)"
        ),

        deterministic_results=[
            {
                "status": "FAIL",
                "check": "Routing",
                "message": (
                    "Missing route for 172.16.2.0/24."
                )
            }
        ],
    )

    print(
        json.dumps(
            sample,
            indent=2,
            ensure_ascii=False
        )
    )