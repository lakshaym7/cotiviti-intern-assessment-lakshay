"""
Healthcare Claims Review — Agentic AI Pipeline
Single-page Streamlit proof-of-concept demonstrating a 4-agent pipeline for
explainable, confidence-scored radiology claim adjudication with human-in-the-loop
escalation. All patient data is entirely synthetic and fictional.
"""

import json
import os
import re
import time

import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data import (
    CODING_RULES,
    MOCK_EXTRACTIONS,
    MOCK_SYNTHESES,
    POLICY_CHUNKS,
    SAMPLE_CLAIMS,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Claims Review AI • Cotiviti POC",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — minimal but polished
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global font */
    html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

    /* Sidebar note preview box */
    .note-box {
        background: #1e1e2e;
        border: 1px solid #383850;
        border-radius: 8px;
        padding: 14px;
        font-size: 12px;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        white-space: pre-wrap;
        color: #cdd6f4;
        max-height: 340px;
        overflow-y: auto;
        line-height: 1.5;
    }

    /* Agent card label */
    .agent-label {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 2px;
    }

    /* Rule check rows */
    .check-row {
        padding: 6px 10px;
        border-radius: 6px;
        margin-bottom: 4px;
        font-size: 13px;
    }
    .check-pass  { background: #1a2f1a; border-left: 3px solid #4CAF50; }
    .check-warn  { background: #2f2a14; border-left: 3px solid #FFC107; }
    .check-fail  { background: #2f1a1a; border-left: 3px solid #F44336; }

    /* Verdict card */
    .verdict-card {
        padding: 28px 32px;
        border-radius: 14px;
        text-align: center;
        margin-top: 8px;
    }
    .verdict-approve { background: #0f2210; border: 2px solid #4CAF50; }
    .verdict-deny    { background: #220f0f; border: 2px solid #F44336; }
    .verdict-escalate{ background: #1f1c0c; border: 2px solid #FFC107; }

    .verdict-title {
        font-size: 36px;
        font-weight: 800;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }
    .verdict-approve  .verdict-title { color: #4CAF50; }
    .verdict-deny     .verdict-title { color: #F44336; }
    .verdict-escalate .verdict-title { color: #FFC107; }

    .verdict-subtitle { color: #888; font-size: 13px; margin-bottom: 16px; }

    /* Mock mode badge */
    .mock-badge {
        display: inline-block;
        background: #7c3f00;
        color: #FFC107;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.1em;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid #FFC107;
        text-transform: uppercase;
    }

    /* Override notice */
    .override-box {
        background: #1f1c0c;
        border: 2px solid #FFC107;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 12px;
    }
    .override-title {
        color: #FFC107;
        font-weight: 700;
        font-size: 14px;
        margin-bottom: 6px;
    }
    .override-body { color: #ccc; font-size: 13px; line-height: 1.6; }

    /* Citation pill */
    .citation {
        display: inline-block;
        background: #252540;
        border: 1px solid #4444aa;
        border-radius: 5px;
        padding: 2px 10px;
        font-size: 12px;
        color: #aaaaee;
        margin: 2px 4px;
        font-family: monospace;
    }

    /* Confidence bar container */
    .conf-bar-bg {
        background: #2a2a2a;
        border-radius: 99px;
        height: 10px;
        width: 100%;
        margin: 10px 0 16px;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 10px;
        border-radius: 99px;
    }

    /* Policy snippet */
    .policy-snippet {
        background: #1a1a2e;
        border-left: 3px solid #6666dd;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        font-size: 12.5px;
        color: #ccc;
        line-height: 1.6;
        margin-bottom: 10px;
    }
    .policy-title {
        font-size: 11px;
        font-weight: 700;
        color: #8888ee;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 5px;
    }

    /* Section divider */
    hr.thin { border: none; border-top: 1px solid #2a2a2a; margin: 18px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# LLM wrapper — Anthropic API with transparent mock fallback
# ---------------------------------------------------------------------------
def _api_key() -> str:
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()


def _mock_mode() -> bool:
    return not bool(_api_key())


# ---------------------------------------------------------------------------
# Input guardrails — validate the physician note before the pipeline runs
# ---------------------------------------------------------------------------
_PROMPT_INJECTION_PATTERNS = [
    "ignore previous", "ignore all", "disregard", "forget your instructions",
    "you are now", "system:", "act as", "jailbreak", "new persona",
    "pretend you are", "override instructions",
]

_MEDICAL_SIGNALS = [
    "patient", "diagnosis", "physician", "procedure", "clinical", "icd",
    "cpt", "complaint", "history", "examination", "impression", "ordered",
    "imaging", "radiology", "treatment", "symptom", "finding",
]


def validate_input(note: str) -> list[str]:
    """Return a list of guardrail violation messages. Empty list = input is safe."""
    issues = []
    stripped = note.strip()

    if len(stripped) < 80:
        issues.append(
            "Note is too short (< 80 characters). Please paste a complete physician note."
        )

    note_lower = stripped.lower()

    if any(pat in note_lower for pat in _PROMPT_INJECTION_PATTERNS):
        issues.append(
            "Input contains patterns that resemble a prompt injection attempt and has been blocked."
        )

    if len(stripped) >= 80 and not any(sig in note_lower for sig in _MEDICAL_SIGNALS):
        issues.append(
            "Note does not appear to contain clinical content. "
            "Please paste a physician note with diagnosis, procedure, or patient history."
        )

    return issues


# ---------------------------------------------------------------------------
# Output guardrails — validate LLM responses before using them downstream
# ---------------------------------------------------------------------------
_ICD10_RE = re.compile(r"^[A-Z]\d{2}(\.\d{1,4}[A-Z0-9]?)?$")
_CPT_RE = re.compile(r"^\d{5}$|^[A-Z]\d{4}$")
_VALID_RECOMMENDATIONS = {"approve", "deny", "escalate"}


def validate_extraction(data: dict) -> list[str]:
    """Warn if LLM extraction output has format anomalies."""
    warnings = []
    icd = data.get("icd10_suggested", "")
    cpt = data.get("cpt_suggested", "")

    if icd and icd not in ("Unknown", "N/A") and not _ICD10_RE.match(icd.upper()):
        warnings.append(f"Extracted ICD-10 code '{icd}' does not match expected format — verify manually.")

    if cpt and cpt not in ("Unknown", "N/A") and not _CPT_RE.match(cpt.upper()):
        warnings.append(f"Extracted CPT/HCPCS code '{cpt}' does not match expected format — verify manually.")

    for field in ("diagnosis_findings", "body_site", "procedure_ordered"):
        if not data.get(field, "").strip():
            warnings.append(f"Extraction field '{field}' is empty — LLM may have returned incomplete output.")

    return warnings


def validate_synthesis(data: dict) -> list[str]:
    """Warn if LLM synthesis output has invalid values."""
    warnings = []

    rec = data.get("recommendation", "")
    if rec.lower() not in _VALID_RECOMMENDATIONS:
        warnings.append(
            f"Synthesis returned unexpected recommendation '{rec}'. "
            "Expected: Approve, Deny, or Escalate. Defaulting to Escalate."
        )

    conf = data.get("confidence", -1)
    try:
        conf_int = int(conf)
        if not (0 <= conf_int <= 100):
            warnings.append(f"Confidence score {conf_int} is outside 0–100 range — clamping to nearest bound.")
    except (TypeError, ValueError):
        warnings.append(f"Confidence value '{conf}' is not a valid integer.")

    if not data.get("explanation", "").strip():
        warnings.append("Synthesis produced no explanation — output may be incomplete.")

    return warnings


def call_llm(prompt: str) -> str:
    """Call Claude claude-sonnet-4-6. Raises RuntimeError if in mock mode (caller handles fallback)."""
    if _mock_mode():
        raise RuntimeError("MOCK_MODE")
    try:
        import anthropic  # local import so missing package doesn't break mock mode

        client = anthropic.Anthropic(api_key=_api_key())
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except ImportError:
        raise RuntimeError("anthropic package not installed")


def _parse_json_response(text: str) -> dict:
    """Extract JSON from an LLM response, tolerating markdown code fences."""
    text = text.strip()
    # Strip ```json ... ``` fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find the first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


# ---------------------------------------------------------------------------
# Agent 1: Extraction Agent (LLM)
# ---------------------------------------------------------------------------
def extraction_agent(note: str, claim_id: str | None) -> dict:
    if _mock_mode():
        key = claim_id if claim_id in MOCK_EXTRACTIONS else "CUSTOM"
        return MOCK_EXTRACTIONS[key]

    prompt = f"""You are a medical coding assistant. Extract structured clinical information from the physician note below.

Return a JSON object with EXACTLY these fields (no extra fields, no markdown):
{{
  "diagnosis_findings": "<main clinical findings or diagnosis in plain English>",
  "body_site": "<specific anatomical location>",
  "laterality": "<Right | Left | Bilateral | N/A | Unspecified>",
  "procedure_ordered": "<the imaging study or procedure ordered>",
  "icd10_suggested": "<ICD-10 code from the note, or most appropriate if not stated>",
  "cpt_suggested": "<CPT or HCPCS code from the note, or most appropriate if not stated>",
  "documentation_completeness": "<1–2 sentence assessment of documentation quality and any gaps>"
}}

Return ONLY valid JSON. No preamble, no explanation, no markdown fences.

PHYSICIAN NOTE:
{note}"""

    try:
        raw = call_llm(prompt)
        return _parse_json_response(raw)
    except Exception:
        # Graceful degradation — return minimal structure
        return {
            "diagnosis_findings": "Extraction failed — see raw note",
            "body_site": "Unknown",
            "laterality": "Unknown",
            "procedure_ordered": "Unknown",
            "icd10_suggested": "Unknown",
            "cpt_suggested": "Unknown",
            "documentation_completeness": "LLM extraction error; manual review required.",
        }


# ---------------------------------------------------------------------------
# Agent 2: Coding Validation Agent (DETERMINISTIC — rule-based, no LLM)
# ---------------------------------------------------------------------------

# Keyword maps for required-documentation checks
_DOC_KEYWORDS: dict[str, list[str]] = {
    "acute injury or trauma mechanism": ["fall", "injury", "trauma", "accident", "struck", "hit", "twisted"],
    "laterality specified": ["right", "left", "bilateral"],
    "clinical indication (fracture suspicion, tenderness, or swelling)": [
        "fracture", "tenderness", "swelling", "edema", "ecchymosis", "pain", "unable to bear"
    ],
    "patient age documented (50–80)": ["year-old", "years old", " yo", "age "],
    "pack-year smoking history (≥20 pack-years)": ["pack-year", "pack year", "packs per day", "ppd"],
    "shared decision-making counseling documented": ["shared decision", "counseling", "counsel", "decision-making"],
    "conservative treatment failure documented (PT, NSAIDs, or activity modification)": [
        "physical therapy", " pt ", "nsaid", "ibuprofen", "naproxen", "conservative",
        "activity modification", "rest", "ice", "failed",
    ],
    "specific laterality specified": ["right", "left"],
    "objective clinical findings on examination": [
        "tenderness", "swelling", "edema", "ecchymosis", "effusion", "crepitus",
        "limited range", "decreased range", "restricted motion", "point tender",
    ],
    "clinical suspicion or Wells score documented": [
        "wells", "suspicion", "probability", "d-dimer", "dvt", "pulmonary embolism",
        "dyspnea", "tachycardia",
    ],
    "contrast administration protocol noted": [
        "contrast administered", "contrast given", "iv contrast",
        "pre-medication protocol completed", "premedication completed",
        "contrast allergy protocol", "iodine allergy protocol",
        "with contrast", "contrast clearance",
    ],
}


_NEGATION_WINDOW = 70   # characters to look back before a keyword for negation
_NEGATION_AFTER = 45    # characters to look ahead after a keyword for post-negation

# Phrases that negate a keyword when appearing BEFORE it
_NEGATION_BEFORE = [
    "no ", "not ", "without ", "none ", "never ", "absent",
    "no prior", "no history", "not performed", "not documented",
    "denied", "denies", "no documented", "non-",
]
# Phrases that negate a keyword when appearing immediately AFTER it
_NEGATION_AFTER_PHRASES = [
    "not completed", "not documented", "not performed",
    "not administered", "not given", "not done",
]


def _check_doc(requirement: str, note_lower: str) -> bool:
    """Return True if a keyword for this requirement is found and not negated in note.

    Uses word-boundary regex to avoid substring false positives (e.g. 'rest' in
    'requests'), checks a window before the match for negation prefixes, and a
    short window after the match for post-keyword negation ('...not completed').
    """
    keywords = _DOC_KEYWORDS.get(requirement, [])
    if not keywords:
        keywords = [w for w in requirement.lower().split() if len(w) > 4]

    for kw in keywords:
        pattern = r"\b" + re.escape(kw.strip()) + r"\b"
        for m in re.finditer(pattern, note_lower):
            pos, end = m.start(), m.end()
            before = note_lower[max(0, pos - _NEGATION_WINDOW) : pos]
            after = note_lower[end : end + _NEGATION_AFTER]
            if any(neg in before for neg in _NEGATION_BEFORE):
                continue
            if any(neg in after for neg in _NEGATION_AFTER_PHRASES):
                continue
            return True  # keyword present and not negated in either direction
    return False


def coding_validation_agent(extraction: dict, note: str) -> dict:
    icd10 = extraction.get("icd10_suggested", "").upper().strip()
    cpt = extraction.get("cpt_suggested", "").upper().strip()
    laterality = extraction.get("laterality", "").lower()
    note_lower = note.lower()

    # --- Find matching rule ---
    matched_rule = None
    for rule in CODING_RULES:
        for prefix in rule["icd10_prefixes"]:
            if icd10.startswith(prefix.upper()):
                matched_rule = rule
                break
        if matched_rule:
            break

    if matched_rule is None:
        return {
            "rule_matched": None,
            "rule_description": "No matching rule found",
            "valid_pair": False,
            "has_coding_conflict": False,
            "has_denial_indicator": True,
            "conflicts": [f"ICD-10 code '{icd10}' not found in the coding rule table."],
            "warnings": [],
            "present_docs": [],
            "missing_docs": [],
            "summary_status": "UNKNOWN",
        }

    valid_cpts = [c.upper() for c in matched_rule["valid_cpt_codes"]]
    is_denial_type = matched_rule["conflict_type"] == "denial"
    is_mismatch_type = matched_rule["conflict_type"] == "coding_mismatch"
    cpt_valid = cpt in valid_cpts

    # Determine conflict flags
    if is_denial_type:
        # Routine exam / no imaging supported → denial indicator, NOT a coding conflict
        has_coding_conflict = False
        has_denial_indicator = True
    elif not cpt_valid:
        # Wrong CPT for the diagnosis → coding mismatch
        has_coding_conflict = True
        has_denial_indicator = False
    else:
        has_coding_conflict = False
        has_denial_indicator = False

    # --- Check required documentation ---
    required_docs = matched_rule.get("required_documentation", [])
    present_docs, missing_docs = [], []
    for req in required_docs:
        if _check_doc(req, note_lower):
            present_docs.append(req)
        else:
            missing_docs.append(req)

    # --- Laterality mismatch check ---
    # ICD-10-CM encodes laterality (1=right, 2=left) at the 6th character position
    # ONLY for musculoskeletal (M) and trauma/injury (S) code categories.
    # Z, I, F, G and other categories use that position for other purposes,
    # so we restrict the check to codes starting with M or S to avoid false positives.
    _icd_nodot = icd10.replace(".", "")
    _lat_char = _icd_nodot[5] if len(_icd_nodot) >= 6 else ""
    _laterality_category = icd10[0] if icd10 else ""
    laterality_mismatch = (
        "bilateral" in laterality
        and _laterality_category in ("M", "S")
        and _lat_char in ("1", "2")
        and "bilateral" not in icd10.lower()
    )

    # Build conflicts / warnings lists
    conflicts, warnings = [], []

    if is_denial_type:
        conflicts.append(matched_rule.get("conflict_reason", "This diagnosis does not support imaging."))
    elif not cpt_valid:
        expected = ", ".join(matched_rule["valid_cpt_codes"]) if matched_rule["valid_cpt_codes"] else "none"
        conflicts.append(
            f"CPT {cpt} is not valid for ICD-10 {icd10}. "
            f"Expected: {expected}. "
            + (matched_rule.get("conflict_reason") or "")
        )

    if laterality_mismatch:
        warnings.append(
            "Laterality discrepancy: bilateral complaint but laterality-specific ICD-10 code used. "
            "Confirm the correct side was imaged and coded."
        )

    # Overall status — missing_docs count as warnings even though they render separately
    if conflicts:
        summary_status = "CONFLICT" if has_coding_conflict else "DENIAL"
    elif warnings or missing_docs:
        summary_status = "WARNING"
    else:
        summary_status = "VALID"

    # Effective valid_pair
    pair_is_valid = cpt_valid and not is_denial_type

    return {
        "rule_matched": matched_rule["id"],
        "rule_description": (
            f"{matched_rule['icd10_description']} → {matched_rule['cpt_description']}"
        ),
        "valid_pair": pair_is_valid,
        "has_coding_conflict": has_coding_conflict,
        "has_denial_indicator": has_denial_indicator,
        "conflicts": conflicts,
        "warnings": warnings,
        "present_docs": present_docs,
        "missing_docs": missing_docs,
        "summary_status": summary_status,
    }


# ---------------------------------------------------------------------------
# Agent 3: Policy Retrieval Agent (TF-IDF RAG — no vector DB)
# ---------------------------------------------------------------------------
def policy_retrieval_agent(extraction: dict, top_k: int = 2) -> dict:
    query = " ".join(
        filter(
            None,
            [
                extraction.get("diagnosis_findings", ""),
                extraction.get("body_site", ""),
                extraction.get("procedure_ordered", ""),
                extraction.get("icd10_suggested", ""),
                extraction.get("cpt_suggested", ""),
            ],
        )
    )

    documents = [chunk["text"] for chunk in POLICY_CHUNKS]

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(documents + [query])

    query_vec = tfidf_matrix[-1]
    doc_vecs = tfidf_matrix[:-1]
    scores = cosine_similarity(query_vec, doc_vecs).flatten()

    MIN_SIMILARITY = 0.03  # filter out near-zero matches that add no signal
    top_indices = np.argsort(scores)[::-1][:top_k]
    retrieved = []
    for rank, idx in enumerate(top_indices):
        score = float(scores[idx])
        # Always include the top result; drop subsequent results below threshold
        if rank > 0 and score < MIN_SIMILARITY:
            break
        retrieved.append(
            {
                "chunk": POLICY_CHUNKS[idx],
                "similarity_score": round(score, 4),
            }
        )

    return {"query": query[:200] + ("…" if len(query) > 200 else ""), "retrieved": retrieved}


# ---------------------------------------------------------------------------
# Agent 4: Synthesis Agent (LLM)
# ---------------------------------------------------------------------------
def synthesis_agent(
    extraction: dict,
    validation: dict,
    policy_result: dict,
    claim_id: str | None,
) -> dict:
    if _mock_mode():
        key = claim_id if claim_id in MOCK_SYNTHESES else "CUSTOM"
        return MOCK_SYNTHESES[key]

    policy_text = "\n\n".join(
        f"[{r['chunk']['id']}] {r['chunk']['title']} (similarity {r['similarity_score']}):\n{r['chunk']['text']}"
        for r in policy_result.get("retrieved", [])
    )

    prompt = f"""You are a healthcare payment integrity analyst reviewing a radiology insurance claim.

=== AGENT 1 — EXTRACTED CLINICAL DATA ===
{json.dumps(extraction, indent=2)}

=== AGENT 2 — CODING VALIDATION RESULT ===
{json.dumps(validation, indent=2)}

=== AGENT 3 — RETRIEVED POLICY SNIPPETS ===
{policy_text}

Based on the above, produce a final reimbursement recommendation. Return a JSON object with EXACTLY these 5 fields:
{{
  "recommendation": "Approve" | "Deny" | "Escalate",
  "confidence": <integer 0–100>,
  "explanation": "<2–3 sentence plain-English rationale citing specific policy ID and coding rule ID>",
  "policy_cited": "<policy ID, e.g. POL_001>",
  "rule_cited": "<rule ID, e.g. RULE_001>"
}}

Decision guidelines:
- Approve: valid code pair, complete documentation, clear medical necessity
- Deny: no medical necessity, invalid code pair for routine/unrelated encounter, no clinical indication
- Escalate: ambiguous, conflicting information, missing key documentation requiring clinical reviewer
- If documentation is incomplete OR there are coding conflicts OR warnings → confidence should be < 70
- Cite the single most relevant policy snippet ID and coding rule ID

Return ONLY valid JSON. No preamble, no explanation, no markdown.
"""

    try:
        raw = call_llm(prompt)
        result = _parse_json_response(raw)
        # Ensure confidence is an int
        result["confidence"] = int(result.get("confidence", 50))
        return result
    except Exception:
        return {
            "recommendation": "Escalate",
            "confidence": 40,
            "explanation": "LLM synthesis encountered an error. Manual review required.",
            "policy_cited": "POL_006",
            "rule_cited": "RULE_001",
        }


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _conf_bar(confidence: int, color: str) -> str:
    return (
        f'<div class="conf-bar-bg">'
        f'<div class="conf-bar-fill" style="width:{confidence}%; background:{color};"></div>'
        f"</div>"
    )


def _verdict_color(rec: str) -> tuple[str, str, str]:
    """Returns (css_class, hex_color, emoji)."""
    rec = rec.lower()
    if "approve" in rec:
        return "verdict-approve", "#4CAF50", "✓"
    if "deny" in rec:
        return "verdict-deny", "#F44336", "✗"
    return "verdict-escalate", "#FFC107", "⚠"


def _render_check(icon: str, label: str, css_class: str) -> str:
    return f'<div class="check-row {css_class}">{icon} {label}</div>'


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
def main() -> None:
    # ---- Sidebar ----
    with st.sidebar:
        st.markdown("## 🏥 Claims Review AI")
        st.markdown("*Agentic AI Pipeline — Healthcare POC*")
        st.markdown('<hr class="thin">', unsafe_allow_html=True)

        if _mock_mode():
            st.markdown(
                '<span class="mock-badge">⚠ MOCK MODE — no API key</span>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Set `ANTHROPIC_API_KEY` in your environment for live LLM calls. "
                "Pre-written responses are used for sample claims."
            )
            st.markdown('<hr class="thin">', unsafe_allow_html=True)

        mode = st.radio("Input mode", ["Sample Claim", "Custom Note"], horizontal=True)

        claim_id: str | None = None

        if mode == "Sample Claim":
            options = {f"{c['title']}": c for c in SAMPLE_CLAIMS}
            selected_title = st.selectbox("Select a claim", list(options.keys()))
            selected_claim = options[selected_title]
            claim_id = selected_claim["id"]

            st.markdown(
                f"**Expected outcome:** `{selected_claim['expected_outcome']}`  \n"
                f"**Modality:** {selected_claim['modality']}"
            )
            note = selected_claim["physician_note"]

            st.markdown("**Physician Note Preview**")
            st.markdown(
                f'<div class="note-box">{note}</div>', unsafe_allow_html=True
            )
        else:
            note = st.text_area(
                "Paste physician note",
                height=340,
                placeholder=(
                    "Paste a free-text physician note here.\n\n"
                    "Include: patient history, physical exam findings,\n"
                    "clinical impression, and diagnosis + procedure codes."
                ),
            )
            if _mock_mode():
                st.info(
                    "Custom notes in mock mode return a generic placeholder response. "
                    "Add an API key for live extraction.",
                    icon="ℹ️",
                )

        st.markdown('<hr class="thin">', unsafe_allow_html=True)
        run_button = st.button("🚀  Run Pipeline", type="primary", use_container_width=True)

    # ---- Main panel ----
    st.markdown("# Healthcare Claims Review")
    st.markdown(
        "A **4-agent AI pipeline** that reads a radiology physician note and produces an "
        "explainable, confidence-scored reimbursement recommendation. "
        "Each step runs in sequence — select a claim from the sidebar and click **Run Pipeline**."
    )

    if not run_button:
        st.info("← Select a claim and click **Run Pipeline** to begin.", icon="👈")
        return

    if not note or not note.strip():
        st.warning("Please enter or select a physician note before running.")
        return

    # --- Input guardrails ---
    input_issues = validate_input(note)
    if input_issues:
        st.error("**Input validation failed — pipeline blocked.**")
        for issue in input_issues:
            st.markdown(f"- {issue}")
        return

    # ================================================================
    # PIPELINE EXECUTION
    # ================================================================

    # --- Agent 1 ---
    with st.status(
        "**Agent 1 — Extraction** &nbsp;&nbsp;`LLM` &nbsp;·&nbsp; Parsing physician note…",
        expanded=True,
    ) as s1:
        time.sleep(1.6 if not _mock_mode() else 0.9)
        extraction = extraction_agent(note, claim_id)
        s1.update(
            label="**Agent 1 — Extraction** &nbsp;&nbsp;`LLM` &nbsp;·&nbsp; ✅ Clinical data extracted",
            state="complete",
            expanded=True,
        )

        st.markdown('<p class="agent-label" style="color:#6ec6ff;">📋 Extracted Fields</p>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Diagnosis / Findings**  \n{extraction.get('diagnosis_findings','—')}")
            st.markdown(f"**Body Site**  \n{extraction.get('body_site','—')}")
            st.markdown(f"**Laterality**  \n{extraction.get('laterality','—')}")
        with col_b:
            st.markdown(f"**Procedure Ordered**  \n{extraction.get('procedure_ordered','—')}")
            icd = extraction.get("icd10_suggested", "—")
            cpt = extraction.get("cpt_suggested", "—")
            st.markdown(f"**ICD-10 (suggested)** &nbsp;<code>{icd}</code>", unsafe_allow_html=True)
            st.markdown(f"**CPT / HCPCS (suggested)** &nbsp;<code>{cpt}</code>", unsafe_allow_html=True)
        st.markdown(
            f"**Documentation Assessment**  \n{extraction.get('documentation_completeness','—')}"
        )

        # Output guardrail — validate extraction fields
        ext_issues = validate_extraction(extraction)
        if ext_issues:
            for w in ext_issues:
                st.warning(f"Output guardrail: {w}", icon="⚠️")

    # --- Agent 2 ---
    with st.status(
        "**Agent 2 — Coding Validation** &nbsp;&nbsp;`RULE-BASED` &nbsp;·&nbsp; Checking ICD-10/CPT rule table…",
        expanded=True,
    ) as s2:
        time.sleep(1.1 if not _mock_mode() else 0.6)
        validation = coding_validation_agent(extraction, note)
        status_emoji = {"VALID": "✅", "WARNING": "⚠️", "CONFLICT": "🚨", "DENIAL": "🚫", "UNKNOWN": "❓"}.get(
            validation["summary_status"], "❓"
        )
        s2.update(
            label=(
                f"**Agent 2 — Coding Validation** &nbsp;&nbsp;`RULE-BASED` &nbsp;·&nbsp; "
                f"{status_emoji} {validation['summary_status']}"
            ),
            state="complete",
            expanded=True,
        )

        st.markdown(
            '<p class="agent-label" style="color:#a8d8a8;">🔒 Deterministic Rule Check — No LLM Involved</p>',
            unsafe_allow_html=True,
        )

        rule_id = validation.get("rule_matched") or "—"
        rule_desc = validation.get("rule_description") or "No rule matched"
        st.markdown(f"**Rule applied:** `{rule_id}` — {rule_desc}")

        pair_icon = "✅" if validation["valid_pair"] else "❌"
        pair_label = "Valid ICD-10 / CPT code pair" if validation["valid_pair"] else "Invalid ICD-10 / CPT code pair"
        pair_css = "check-pass" if validation["valid_pair"] else "check-fail"
        html_checks = _render_check(pair_icon, pair_label, pair_css)

        for doc in validation.get("present_docs", []):
            html_checks += _render_check("✅", f"Documentation present: {doc}", "check-pass")
        for doc in validation.get("missing_docs", []):
            html_checks += _render_check("⚠️", f"Missing: {doc}", "check-warn")
        for conflict in validation.get("conflicts", []):
            html_checks += _render_check("🚨", conflict, "check-fail")
        for warn in validation.get("warnings", []):
            html_checks += _render_check("⚠️", warn, "check-warn")

        st.markdown(html_checks, unsafe_allow_html=True)

    # --- Agent 3 ---
    with st.status(
        "**Agent 3 — Policy Retrieval** &nbsp;&nbsp;`TF-IDF RAG` &nbsp;·&nbsp; Searching payer policy corpus…",
        expanded=True,
    ) as s3:
        time.sleep(1.0 if not _mock_mode() else 0.5)
        policy_result = policy_retrieval_agent(extraction)
        s3.update(
            label=(
                f"**Agent 3 — Policy Retrieval** &nbsp;&nbsp;`TF-IDF RAG` &nbsp;·&nbsp; "
                f"✅ {len(policy_result['retrieved'])} snippet(s) retrieved"
            ),
            state="complete",
            expanded=True,
        )

        st.markdown(
            '<p class="agent-label" style="color:#c3b1e1;">📚 Retrieved Policy Snippets (cosine similarity)</p>',
            unsafe_allow_html=True,
        )
        st.caption(f"Query: *{policy_result['query']}*")

        for rank, item in enumerate(policy_result["retrieved"], 1):
            chunk = item["chunk"]
            score = item["similarity_score"]
            score_pct = min(int(score * 100 / 0.6 * 100), 100)  # rough normalisation for display
            st.markdown(
                f'<div class="policy-snippet">'
                f'<div class="policy-title">#{rank} &nbsp;{chunk["id"]} — {chunk["title"]}'
                f'&nbsp;&nbsp;<span style="color:#666;font-weight:400;">similarity {score:.3f}</span></div>'
                f"{chunk['text']}"
                f"</div>",
                unsafe_allow_html=True,
            )

    # --- Agent 4 ---
    with st.status(
        "**Agent 4 — Synthesis** &nbsp;&nbsp;`LLM` &nbsp;·&nbsp; Generating recommendation…",
        expanded=True,
    ) as s4:
        time.sleep(2.2 if not _mock_mode() else 1.0)
        synthesis = synthesis_agent(extraction, validation, policy_result, claim_id)
        s4.update(
            label=(
                f"**Agent 4 — Synthesis** &nbsp;&nbsp;`LLM` &nbsp;·&nbsp; "
                f"✅ Recommendation: {synthesis.get('recommendation','—')} "
                f"(confidence {synthesis.get('confidence','?')}%)"
            ),
            state="complete",
            expanded=True,
        )

        st.markdown(
            '<p class="agent-label" style="color:#f4a261;">🧠 Raw LLM Output (before override check)</p>',
            unsafe_allow_html=True,
        )
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            raw_rec = synthesis.get("recommendation", "—")
            raw_conf = synthesis.get("confidence", 0)
            css_cls, hex_col, _ = _verdict_color(raw_rec)
            st.markdown(
                f'<div style="text-align:center; padding:14px; border-radius:10px; '
                f'background:#1a1a1a; border:1px solid #333;">'
                f'<div style="font-size:22px; font-weight:700; color:{hex_col};">{raw_rec}</div>'
                f'<div style="font-size:13px; color:#888; margin-top:4px;">Confidence: {raw_conf}%</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_s2:
            st.markdown(f"**Explanation:** {synthesis.get('explanation','—')}")
            pol_cite = synthesis.get("policy_cited", "—")
            rule_cite = synthesis.get("rule_cited", "—")
            st.markdown(
                f'Citations: <span class="citation">{pol_cite}</span> <span class="citation">{rule_cite}</span>',
                unsafe_allow_html=True,
            )

        # Output guardrail — validate synthesis fields
        syn_issues = validate_synthesis(synthesis)
        if syn_issues:
            for w in syn_issues:
                st.warning(f"Output guardrail: {w}", icon="⚠️")

    # ================================================================
    # FINAL VERDICT + HUMAN-IN-THE-LOOP OVERRIDE
    # ================================================================
    st.markdown('<hr class="thin">', unsafe_allow_html=True)
    st.markdown("## Final Adjudication Verdict")

    raw_rec = synthesis.get("recommendation", "Escalate")
    # Clamp confidence defensively in case LLM returned out-of-range value
    raw_conf = max(0, min(100, int(synthesis.get("confidence", 0))))
    has_coding_conflict = validation.get("has_coding_conflict", False)

    # Hard override logic — human-in-the-loop safety mechanism
    override_triggered = raw_conf < 70 or has_coding_conflict
    override_reason_parts = []
    if raw_conf < 70:
        override_reason_parts.append(f"confidence score is {raw_conf}% (threshold: 70%)")
    if has_coding_conflict:
        override_reason_parts.append("coding validation flagged a billing code mismatch")

    final_rec = "Escalate to Human Reviewer" if override_triggered else raw_rec
    final_conf = raw_conf

    css_class, hex_col, emoji = _verdict_color(final_rec)

    # Confidence bar colour
    bar_col = "#4CAF50" if final_conf >= 70 else "#FFC107" if final_conf >= 40 else "#F44336"

    # Render verdict card
    st.markdown(
        f'<div class="verdict-card {css_class}">'
        f'<div class="verdict-title">{emoji}&nbsp; {final_rec.upper()}</div>'
        f'<div class="verdict-subtitle">AI pipeline recommendation</div>'
        + _conf_bar(final_conf, bar_col)
        + f'<div style="font-size:15px; color:#bbb;">Confidence Score: <strong style="color:{hex_col};">{final_conf}%</strong></div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # Override notice (visually prominent when triggered)
    if override_triggered:
        reason_str = " and ".join(override_reason_parts)
        llm_already_escalated = "escalate" in raw_rec.lower()
        if llm_already_escalated:
            override_body = (
                f"The pipeline safety guardrail has independently <strong>confirmed and enforced</strong> "
                f"escalation to a human reviewer because <strong>{reason_str}</strong>. "
                f"The LLM also recommended escalation — but this guardrail runs unconditionally: "
                f"even if the LLM had said Approve or Deny, the deterministic rule would have "
                f"overridden it. It cannot be bypassed by the model."
            )
        else:
            override_body = (
                f"The pipeline hard-coded safety rule has <strong>overridden</strong> the AI's raw "
                f"recommendation (<strong>{raw_rec}</strong>) and forced escalation to a human reviewer "
                f"because <strong>{reason_str}</strong>. "
                f"This override is non-negotiable and cannot be bypassed by the LLM — it is a "
                f"deterministic guardrail built into the pipeline architecture."
            )
        st.markdown(
            f'<div class="override-box">'
            f'<div class="override-title">🔒 HUMAN-IN-THE-LOOP OVERRIDE TRIGGERED</div>'
            f'<div class="override-body">{override_body}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    # Explanation & citations
    st.markdown('<hr class="thin">', unsafe_allow_html=True)
    col_e1, col_e2 = st.columns([3, 1])
    with col_e1:
        st.markdown("#### Plain-English Explanation")
        st.markdown(synthesis.get("explanation", "—"))

    with col_e2:
        st.markdown("#### Evidence Citations")
        pol_cite = synthesis.get("policy_cited", "—")
        rule_cite = synthesis.get("rule_cited", "—")
        st.markdown(
            f'<span class="citation">📄 {pol_cite}</span><br>'
            f'<span class="citation">📋 {rule_cite}</span>',
            unsafe_allow_html=True,
        )

        # Show the referenced policy title if we can find it
        for chunk in POLICY_CHUNKS:
            if chunk["id"] == pol_cite:
                st.caption(f"*{chunk['title']}*")
                break
        for rule in CODING_RULES:
            if rule["id"] == rule_cite:
                st.caption(f"*{rule['icd10_description']}*")
                break

    # Footer
    st.markdown('<hr class="thin">', unsafe_allow_html=True)
    st.caption(
        "⚠️ This is a proof-of-concept demonstration using entirely synthetic, fictional data. "
        "No real patient information is used or represented. Not intended for clinical or production use."
    )
    if _mock_mode():
        st.markdown(
            '<span class="mock-badge">⚠ MOCK MODE ACTIVE — set ANTHROPIC_API_KEY for live LLM calls</span>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
