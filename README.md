# Healthcare Claims Review — Agentic AI Pipeline (POC)

A single-page Streamlit proof-of-concept demonstrating how a multi-agent AI pipeline
can produce **explainable, confidence-scored reimbursement recommendations** for
radiology insurance claims, with a hard-coded human-in-the-loop escalation guardrail.

All patient data is **entirely synthetic and fictional**.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Set your Anthropic API key for live LLM calls
#    If omitted, the app runs in MOCK MODE with pre-written responses.
export ANTHROPIC_API_KEY="sk-ant-..."   # macOS / Linux
set ANTHROPIC_API_KEY=sk-ant-...        # Windows CMD
$env:ANTHROPIC_API_KEY="sk-ant-..."     # Windows PowerShell

# 3. Launch the app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 4-Agent Architecture

The pipeline processes each claim through four sequential agents, each shown as a
distinct expandable card in the UI:

| # | Agent | Type | What it does |
|---|-------|------|--------------|
| 1 | **Extraction** | LLM (`claude-sonnet-4-6`) | Parses the raw physician note and extracts diagnosis, body site, laterality, procedure, and suggested ICD-10/CPT codes. |
| 2 | **Coding Validation** | Deterministic rule-based | Checks the extracted code pair against a local ICD-10/CPT rule table. Flags missing required documentation, invalid pairs, and billing mismatches. **No LLM — explicit safety net.** |
| 3 | **Policy Retrieval** | TF-IDF cosine similarity (RAG) | Retrieves the most relevant snippet(s) from a small local corpus of 8 synthetic payer policy chunks using scikit-learn TF-IDF — no vector database. |
| 4 | **Synthesis** | LLM (`claude-sonnet-4-6`) | Consumes all three prior outputs and generates a final recommendation (Approve / Deny / Escalate), a 0–100 confidence score, and a plain-English explanation citing the specific policy snippet and coding rule used. |

### Escalation Logic (Human-in-the-Loop Override)

After the Synthesis Agent produces its raw recommendation, a **hard-coded override rule** runs:

> **If `confidence < 70` OR the Coding Validation Agent flagged a billing code mismatch → force the recommendation to `Escalate to Human Reviewer`**, regardless of what the LLM said.

This override is deterministic, cannot be bypassed by the model, and is displayed as a
visually prominent yellow alert in the UI whenever it fires. It is the primary
human-in-the-loop safety mechanism in the pipeline.

---

## Sample Claims

Five synthetic claims span three imaging modalities:

| Claim | Modality | Expected Outcome | Why |
|-------|----------|-----------------|-----|
| CLAIM001 | X-Ray — Right Foot | **Approve** | Clear fracture, valid codes, complete docs |
| CLAIM002 | CT — LDCT Lung Screening | **Approve** | All eligibility criteria met |
| CLAIM003 | MRI — Left Hip | **Escalate** | Missing conservative therapy docs, laterality discrepancy, low confidence |
| CLAIM004 | X-Ray — Knee | **Deny** | Routine exam code (Z00.00) doesn't support imaging — no clinical indication |
| CLAIM005 | CT — Chest / PE Workup | **Escalate** | Billing code mismatch: PE diagnosis requires CTPA (71275), billed as non-contrast (71250) → hard override |

Use **CLAIM001** or **CLAIM002** for a clean "Approve" demo, and **CLAIM003** or **CLAIM005**
to demonstrate the escalation path and override banner on camera.

---

## Mock Mode

If `ANTHROPIC_API_KEY` is not set, the app runs in **MOCK MODE**: the two LLM agents
(Extraction and Synthesis) return pre-written canned responses for each sample claim.
The rule-based Coding Validation and TF-IDF Policy Retrieval agents always run live.
A yellow `MOCK MODE` badge appears in the sidebar and footer.

---

## File Structure

```
d:\cotiviti\
├── Cotiviti_Presentation.pptx
├── Report.docx
├── video_presentation.mp4
│
└── POC_demo_code\
    ├── app.py
    ├── data.py
    ├── tests.py
    ├── requirements.txt
    └── README.md
```
