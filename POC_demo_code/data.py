"""
Static data: sample claims, coding rules, policy chunks, and mock LLM responses.
All patient data is entirely synthetic and fictional — no real individuals represented.
"""

# ---------------------------------------------------------------------------
# 5 Synthetic Sample Claims
# ---------------------------------------------------------------------------
SAMPLE_CLAIMS = [
    {
        "id": "CLAIM001",
        "title": "Right Foot X-Ray — Suspected 5th Metatarsal Fracture",
        "modality": "X-Ray • Foot",
        "expected_outcome": "Approve",
        "physician_note": (
            "DATE OF SERVICE: 2026-06-20\n"
            "PATIENT: J.D. | DOB: 1985-03-14 | MRN: 00145782\n"
            "ORDERING PHYSICIAN: Dr. Sarah Chen, MD (NPI: 1234567890)\n\n"
            "CHIEF COMPLAINT: Right foot pain following a fall from a ladder 3 days prior.\n\n"
            "CLINICAL HISTORY: 41-year-old male presenting with persistent right foot pain, "
            "swelling, and point tenderness over the lateral midfoot. Patient unable to bear "
            "weight on the right foot since the injury. No prior foot injuries or surgeries. "
            "No anticoagulation therapy.\n\n"
            "PHYSICAL EXAMINATION: Point tenderness at the base of the right 5th metatarsal. "
            "Mild ecchymosis and edema over the lateral midfoot. Neurovascular status intact "
            "distally. Skin intact.\n\n"
            "CLINICAL IMPRESSION: Suspected avulsion fracture, right 5th metatarsal base "
            "(Jones-type injury). Rule out Lisfranc ligamentous disruption.\n\n"
            "PROCEDURE ORDERED: Radiologic examination, right foot, minimum 3 views "
            "(AP, lateral, oblique).\n\n"
            "DIAGNOSIS CODE: S92.3510A — Nondisplaced fracture of fifth metatarsal bone, "
            "right foot, initial encounter\n"
            "PROCEDURE CODE: 73630 — Radiologic examination, foot; minimum 3 views"
        ),
    },
    {
        "id": "CLAIM002",
        "title": "LDCT Lung Cancer Screening — Eligible Active Smoker",
        "modality": "CT • Thorax (LDCT)",
        "expected_outcome": "Approve",
        "physician_note": (
            "DATE OF SERVICE: 2026-06-18\n"
            "PATIENT: R.M. | DOB: 1963-09-22 | MRN: 00298341\n"
            "ORDERING PHYSICIAN: Dr. James Patel, MD (NPI: 9876543210)\n\n"
            "CHIEF COMPLAINT: Annual preventive health visit — lung cancer screening request.\n\n"
            "CLINICAL HISTORY: 62-year-old male, active smoker, 40 pack-year history "
            "(currently 1 PPD × 40 years). No personal history of lung cancer. No hemoptysis, "
            "unexplained weight loss, or persistent cough. Shared decision-making counseling "
            "regarding LDCT benefits, limitations, and follow-up protocol provided and "
            "documented today.\n\n"
            "SCREENING ELIGIBILITY CHECKLIST:\n"
            "  • Age 62 years (criterion: 50–80 years) ✓\n"
            "  • Smoking history 40 pack-years (criterion: ≥20 pack-years) ✓\n"
            "  • Currently smoking (criterion: current or quit ≤15 years ago) ✓\n"
            "  • No signs or symptoms of lung cancer ✓\n"
            "  • Shared decision-making counseling documented ✓\n\n"
            "PROCEDURE ORDERED: Low-dose CT (LDCT) of the thorax for lung cancer screening.\n\n"
            "DIAGNOSIS CODE: Z87.891 — Personal history of nicotine dependence\n"
            "PROCEDURE CODE: G0297 — Low-dose CT scan (LDCT) for lung cancer screening"
        ),
    },
    {
        "id": "CLAIM003",
        "title": "Left Hip MRI — Vague Bilateral Complaint, Missing Conservative Tx Docs",
        "modality": "MRI • Hip",
        "expected_outcome": "Escalate",
        "physician_note": (
            "DATE OF SERVICE: 2026-06-21\n"
            "PATIENT: T.K. | DOB: 1972-11-08 | MRN: 00512967\n"
            "ORDERING PHYSICIAN: Dr. Amy Rodriguez, MD (NPI: 5551234567)\n\n"
            "CHIEF COMPLAINT: Hip discomfort.\n\n"
            "CLINICAL HISTORY: Patient reports bilateral hip discomfort for several months. "
            "No prior imaging or workup documented in this chart. Patient requests imaging. "
            "No prior physical therapy, NSAID trial, or activity modification attempted.\n\n"
            "PHYSICAL EXAMINATION: Mild discomfort to palpation over bilateral hip region. "
            "Gait unremarkable. No specific objective musculoskeletal findings noted.\n\n"
            "CLINICAL IMPRESSION: Hip pain, bilateral. MRI requested by patient.\n\n"
            "PROCEDURE ORDERED: MRI, left hip joint without contrast.\n\n"
            "DIAGNOSIS CODE: M25.552 — Pain in left hip\n"
            "PROCEDURE CODE: 73721 — Magnetic resonance imaging, any joint of lower extremity"
        ),
    },
    {
        "id": "CLAIM004",
        "title": "Knee X-Ray — No Clinical Indication (Routine Physical)",
        "modality": "X-Ray • Knee",
        "expected_outcome": "Deny",
        "physician_note": (
            "DATE OF SERVICE: 2026-06-17\n"
            "PATIENT: L.S. | DOB: 1991-07-30 | MRN: 00634182\n"
            "ORDERING PHYSICIAN: Dr. Kevin Moore, MD (NPI: 7779876543)\n\n"
            "CHIEF COMPLAINT: Annual physical examination.\n\n"
            "CLINICAL HISTORY: 34-year-old female presenting for annual wellness visit. "
            "No musculoskeletal complaints. Patient is entirely asymptomatic. Normal BMI. "
            "Active lifestyle. No prior knee injuries, surgeries, or arthritis history.\n\n"
            "PHYSICAL EXAMINATION: Vitals normal. General examination unremarkable. "
            "Knee examination not performed — no musculoskeletal complaints raised.\n\n"
            "ASSESSMENT & PLAN: Annual preventive care. Routine labs ordered. "
            "Knee X-ray ordered.\n\n"
            "DIAGNOSIS CODE: Z00.00 — Encounter for general adult medical examination "
            "without abnormal findings\n"
            "PROCEDURE CODE: 73560 — Radiologic examination, knee; 1 or 2 views"
        ),
    },
    {
        "id": "CLAIM005",
        "title": "Chest CT — PE Workup with Contrast/Code Billing Conflict",
        "modality": "CT • Thorax",
        "expected_outcome": "Escalate",
        "physician_note": (
            "DATE OF SERVICE: 2026-06-22\n"
            "PATIENT: B.W. | DOB: 1958-04-15 | MRN: 00789023\n"
            "ORDERING PHYSICIAN: Dr. Marcus Thompson, MD (NPI: 3334567890)\n\n"
            "CHIEF COMPLAINT: Acute onset dyspnea and pleuritic chest pain.\n\n"
            "CLINICAL HISTORY: 67-year-old male presenting to the ED with acute onset of "
            "dyspnea, pleuritic chest pain, and tachycardia (HR 108). Recent 14-hour "
            "international flight 48 hours prior. Wells score for PE: 6 points (high "
            "probability). D-dimer markedly elevated at 4.2 mcg/mL FEU.\n\n"
            "PHYSICAL EXAMINATION: HR 108 bpm, SpO2 92% on room air, BP 118/74. "
            "Right lower extremity mild tenderness and swelling. No rales or wheeze.\n\n"
            "CLINICAL IMPRESSION: High clinical probability of acute pulmonary embolism. "
            "CT pulmonary angiography ordered urgently.\n\n"
            "RADIOLOGY TECHNICIAN NOTE: Patient reports allergy to shellfish. Formal "
            "pre-medication protocol for iodinated contrast not completed at time of scan. "
            "Procedure performed as non-contrast CT chest per verbal instruction.\n\n"
            "DIAGNOSIS CODE: I26.99 — Other pulmonary embolism without acute cor pulmonale\n"
            "PROCEDURE CODE: 71250 — Computed tomography, thorax; without contrast material"
        ),
    },
]

# ---------------------------------------------------------------------------
# Coding Rules: ICD-10 / CPT Pair Validation Table
# ---------------------------------------------------------------------------
CODING_RULES = [
    {
        "id": "RULE_001",
        "icd10_prefixes": ["S92.35", "S92.30", "S92.31", "S92.32", "S92.34", "S92.3"],
        "icd10_description": "Fractures of metatarsal bones / acute foot injury",
        "valid_cpt_codes": ["73630", "73620", "73600"],
        "cpt_description": "Radiologic examination, foot (minimum 2–3+ views)",
        "required_documentation": [
            "acute injury or trauma mechanism",
            "laterality specified",
            "clinical indication (fracture suspicion, tenderness, or swelling)",
        ],
        "valid_pair": True,
        "conflict_type": None,
        "conflict_reason": None,
    },
    {
        "id": "RULE_002",
        "icd10_prefixes": ["Z87.891", "F17.2", "F17.21", "Z87.8"],
        "icd10_description": "Personal history of nicotine dependence / tobacco use disorder",
        "valid_cpt_codes": ["G0297"],
        "cpt_description": "Low-dose CT (LDCT) for lung cancer screening",
        "required_documentation": [
            "patient age documented (50–80)",
            "pack-year smoking history (≥20 pack-years)",
            "shared decision-making counseling documented",
        ],
        "valid_pair": True,
        "conflict_type": None,
        "conflict_reason": None,
    },
    {
        "id": "RULE_003",
        "icd10_prefixes": ["M25.55", "M25.56", "M16.0", "M16.1", "M16.9"],
        "icd10_description": "Hip pain / primary/secondary osteoarthritis of hip",
        "valid_cpt_codes": ["73721", "73500", "73510", "73520", "73525"],
        "cpt_description": "MRI or X-ray of the hip joint",
        "required_documentation": [
            "conservative treatment failure documented (PT, NSAIDs, or activity modification)",
            "specific laterality specified",
            "objective clinical findings on examination",
        ],
        "valid_pair": True,
        "conflict_type": "missing_documentation",
        "conflict_reason": (
            "Hip MRI (CPT 73721) requires documented failure of conservative therapy "
            "(min. 4–6 weeks); vague complaints without objective findings or prior "
            "treatment course are insufficient to establish medical necessity."
        ),
    },
    {
        "id": "RULE_004",
        "icd10_prefixes": ["Z00.0", "Z00.1", "Z00.8", "Z00.9", "Z00.00", "Z00.01"],
        "icd10_description": "Routine general medical / preventive examination",
        "valid_cpt_codes": [],
        "cpt_description": "No imaging procedure covered under routine exam without separate clinical indication",
        "required_documentation": [],
        "valid_pair": False,
        "conflict_type": "denial",
        "conflict_reason": (
            "ICD-10 Z00.xx (routine/preventive examination) does not support standalone "
            "diagnostic imaging orders. A separate, documented clinical indication with "
            "its own diagnosis code is required."
        ),
    },
    {
        "id": "RULE_005",
        "icd10_prefixes": ["I26.0", "I26.9", "I26.01", "I26.09", "I26.99"],
        "icd10_description": "Pulmonary embolism",
        "valid_cpt_codes": ["71275"],
        "cpt_description": "CT angiography, thorax (CTPA — contrast mandatory for PE workup)",
        "required_documentation": [
            "clinical suspicion or Wells score documented",
            "contrast administration protocol noted",
        ],
        "valid_pair": True,
        "conflict_type": "coding_mismatch",
        "conflict_reason": (
            "PE diagnosis (I26.xx) requires CT pulmonary angiography with contrast (CPT 71275). "
            "Non-contrast thoracic CT (CPT 71250) is clinically insufficient and represents a "
            "billing code mismatch — the contrast element is integral to diagnostic adequacy."
        ),
    },
]

# ---------------------------------------------------------------------------
# Policy Chunks: Synthetic Payer Policy Snippets (used for TF-IDF RAG)
# ---------------------------------------------------------------------------
POLICY_CHUNKS = [
    {
        "id": "POL_001",
        "title": "Foot and Ankle Imaging — Acute Traumatic Injury",
        "text": (
            "Radiographic evaluation of the foot (CPT 73600–73630) is medically necessary "
            "following acute trauma when there is clinical suspicion of fracture, dislocation, "
            "or significant soft-tissue injury. A minimum of three views (AP, lateral, oblique) "
            "is required for adequate anatomical assessment of metatarsal pathology. Clinical "
            "documentation must specify the mechanism of injury, affected laterality (right vs. "
            "left foot), and physical examination findings consistent with osseous injury "
            "including point tenderness localization. Ottawa Foot Rules criteria should be "
            "referenced when applicable. Claims lacking trauma documentation will not meet "
            "medical necessity criteria."
        ),
    },
    {
        "id": "POL_002",
        "title": "Lung Cancer Screening — LDCT Eligibility and Annual Coverage",
        "text": (
            "Low-dose CT (LDCT) lung cancer screening (HCPCS G0297) is a covered annual "
            "preventive benefit for beneficiaries aged 50–80 who have a smoking history of "
            "at least 20 pack-years and currently smoke or have quit within the past 15 years. "
            "Pre-service shared decision-making counseling must be documented prior to the "
            "initial LDCT screening. Claims must be billed with an appropriate ICD-10 diagnosis "
            "code reflecting tobacco use history (e.g., Z87.891, F17.210). Missing eligibility "
            "criteria documentation — including age, pack-year history, or counseling — will "
            "result in claim denial. Subsequent annual screenings require documentation of "
            "continued eligibility."
        ),
    },
    {
        "id": "POL_003",
        "title": "Hip Joint Imaging — Medical Necessity and Conservative Therapy Requirement",
        "text": (
            "MRI of the hip joint (CPT 73721) is covered for non-traumatic hip pain when "
            "there is documented failure of conservative therapy — minimum 4–6 weeks of "
            "physical therapy, NSAID treatment, or structured activity modification — and "
            "when the clinical note contains objective physical examination findings supporting "
            "the imaging indication. Clinical documentation must include specific laterality, "
            "duration and character of symptoms, and prior treatment attempts. Vague, "
            "non-specific complaints such as 'hip discomfort' or 'bilateral hip pain' without "
            "objective findings or a documented conservative treatment course do not establish "
            "medical necessity for MRI. Prior authorization may be required."
        ),
    },
    {
        "id": "POL_004",
        "title": "Imaging Orders During Routine Preventive Examinations",
        "text": (
            "Diagnostic imaging procedures are not covered as incidental or supplemental "
            "orders during a routine preventive care visit coded with ICD-10 Z00.xx, unless "
            "a separately documented, clinically distinct indication exists for the imaging "
            "study independent of the wellness examination. Imaging orders placed without a "
            "supporting acute or chronic diagnosis code and accompanying clinical documentation "
            "will be denied as not medically necessary. The ordering provider must submit a "
            "separate encounter-level diagnosis to support any imaging ordered during or "
            "adjacent to a preventive visit."
        ),
    },
    {
        "id": "POL_005",
        "title": "CT Pulmonary Angiography — Standard of Care for Pulmonary Embolism",
        "text": (
            "CT pulmonary angiography (CTPA, CPT 71275) is the first-line imaging modality "
            "for evaluation of suspected pulmonary embolism and requires intravenous iodinated "
            "contrast administration as an inherent component of the study. Claims submitted "
            "using non-contrast thoracic CT codes (CPT 71250) for a pulmonary embolism "
            "diagnosis (ICD-10 I26.xx) will be denied as miscoded, as non-contrast CT is not "
            "diagnostically sufficient for PE evaluation. If a documented iodine or contrast "
            "allergy exists, a formal pre-medication protocol or documented rationale for "
            "alternative imaging must appear in the clinical record before the claim will be "
            "processed. Code pair mismatches between PE diagnosis and non-contrast CT billing "
            "require corrected claim submission."
        ),
    },
    {
        "id": "POL_006",
        "title": "General Documentation Requirements for All Imaging Authorization",
        "text": (
            "All diagnostic imaging authorizations and claims require the following minimum "
            "documentation: (1) a specific, documented clinical indication with supporting "
            "ICD-10 diagnosis code; (2) ordering physician name, credentials, and 10-digit NPI; "
            "(3) date of service; (4) anatomical laterality where clinically applicable; and "
            "(5) clinical notes establishing medical necessity. Claims lacking any of these "
            "elements are subject to suspension pending receipt of additional documentation "
            "within 30 days. Failure to provide complete documentation within the cure period "
            "will result in claim denial. Repeated incomplete submissions may trigger a "
            "provider education review."
        ),
    },
    {
        "id": "POL_007",
        "title": "Code Pair Edits and Billing Accuracy in Radiology",
        "text": (
            "Claims with ICD-10 and CPT code pairs that do not conform to published code pair "
            "edit tables will be automatically flagged for clinical review prior to adjudication. "
            "Discrepancies between the clinically ordered procedure type and the billed procedure "
            "code — particularly contrast versus non-contrast CT, laterality mismatches, or "
            "procedure scope mismatches — will result in claim denial or a request for corrected "
            "billing. Providers are responsible for ensuring the billed CPT accurately reflects "
            "the procedure as documented in the signed radiology report. Systematic billing "
            "discrepancies may trigger a provider audit."
        ),
    },
    {
        "id": "POL_008",
        "title": "Bilateral Imaging Studies — Modifier Requirements",
        "text": (
            "When bilateral imaging studies are performed, providers must submit claims with "
            "modifier -50 (bilateral procedure) or as separate line items using RT (right) and "
            "LT (left) modifiers. Claims for bilateral hip, knee, shoulder, or extremity imaging "
            "without proper modifier assignment will be processed as unilateral studies and may "
            "result in underpayment or denial of one side. When the clinical indication specifies "
            "bilateral pathology but a unilateral ICD-10 code is submitted, the claim will be "
            "flagged for review to confirm the laterality of the performed study."
        ),
    },
]

# ---------------------------------------------------------------------------
# Mock LLM Responses (used when ANTHROPIC_API_KEY is absent)
# ---------------------------------------------------------------------------
MOCK_EXTRACTIONS = {
    "CLAIM001": {
        "diagnosis_findings": (
            "Suspected avulsion fracture of the right 5th metatarsal base (Jones-type); "
            "rule out Lisfranc ligamentous disruption. Acute onset following fall from ladder."
        ),
        "body_site": "Right foot — 5th metatarsal base",
        "laterality": "Right",
        "procedure_ordered": "Radiologic examination, right foot, minimum 3 views (AP, lateral, oblique)",
        "icd10_suggested": "S92.3510A",
        "cpt_suggested": "73630",
        "documentation_completeness": (
            "Comprehensive. Mechanism of injury (fall from ladder) documented; specific laterality "
            "(right foot) confirmed; objective exam findings (point tenderness, ecchymosis, inability "
            "to bear weight) present; ICD-10 and CPT codes explicitly stated and clinically consistent."
        ),
    },
    "CLAIM002": {
        "diagnosis_findings": (
            "Annual lung cancer screening for an eligible high-risk patient — 62-year-old active "
            "smoker with 40 pack-year history. No current signs or symptoms of lung cancer."
        ),
        "body_site": "Thorax / bilateral lung fields",
        "laterality": "N/A (bilateral thorax)",
        "procedure_ordered": "Low-dose CT (LDCT) of the thorax for lung cancer screening",
        "icd10_suggested": "Z87.891",
        "cpt_suggested": "G0297",
        "documentation_completeness": (
            "Excellent. All USPSTF eligibility criteria explicitly checklist-confirmed in note: age "
            "62 (within 50–80), 40 pack-years (≥20), active smoker, asymptomatic, shared decision-"
            "making counseling documented. Appropriate ICD-10 and HCPCS codes present."
        ),
    },
    "CLAIM003": {
        "diagnosis_findings": (
            "Bilateral hip pain of several months duration; no specific diagnosis established. "
            "No prior imaging, physical therapy, NSAID use, or conservative treatment documented."
        ),
        "body_site": "Hip joint — bilateral complaint, left hip imaging ordered",
        "laterality": "Bilateral complaint — left hip only ordered",
        "procedure_ordered": "MRI, left hip joint without contrast",
        "icd10_suggested": "M25.552",
        "cpt_suggested": "73721",
        "documentation_completeness": (
            "Poor. No conservative treatment failure documented (no PT, NSAIDs, or activity "
            "modification history). Objective clinical findings are minimal ('mild discomfort to "
            "palpation'). Laterality discrepancy: bilateral complaint coded with left-specific "
            "ICD-10 (M25.552). Duration noted ('several months') but insufficient alone."
        ),
    },
    "CLAIM004": {
        "diagnosis_findings": (
            "Annual wellness visit — no musculoskeletal complaints, no knee symptoms, no clinical "
            "indication for knee imaging documented anywhere in the note."
        ),
        "body_site": "Knee (unspecified laterality)",
        "laterality": "Unspecified",
        "procedure_ordered": "Radiologic examination, knee, 1 or 2 views",
        "icd10_suggested": "Z00.00",
        "cpt_suggested": "73560",
        "documentation_completeness": (
            "Critically deficient for imaging. Diagnosis code Z00.00 is a routine preventive exam "
            "code. No clinical indication for knee X-ray exists anywhere in the note; physician "
            "note explicitly states 'no musculoskeletal complaints' and 'knee examination not "
            "performed.' This is an unsupported imaging order."
        ),
    },
    "CLAIM005": {
        "diagnosis_findings": (
            "High clinical probability of acute pulmonary embolism: Wells score 6, markedly "
            "elevated D-dimer 4.2 mcg/mL, recent long-haul flight, right lower extremity "
            "swelling. CT pulmonary angiography ordered clinically but performed without "
            "contrast; billed as non-contrast CT."
        ),
        "body_site": "Thorax / pulmonary vasculature",
        "laterality": "N/A",
        "procedure_ordered": (
            "CT thorax without contrast material (ordered as CTA, performed without contrast "
            "due to undocumented shellfish allergy; billed as CPT 71250)"
        ),
        "icd10_suggested": "I26.99",
        "cpt_suggested": "71250",
        "documentation_completeness": (
            "Critical gap. Clinical intent (CTA for PE) does not match billed procedure "
            "(non-contrast CT, CPT 71250). PE diagnosis (I26.99) mandates CPT 71275 "
            "(CTPA with contrast). Contrast allergy pre-medication protocol absent. "
            "Significant mismatch between clinical order and executed/billed procedure."
        ),
    },
    "CUSTOM": {
        "diagnosis_findings": "Clinical findings extracted from custom physician note (mock mode active — enable API key for live extraction)",
        "body_site": "See physician note",
        "laterality": "See physician note",
        "procedure_ordered": "See physician note",
        "icd10_suggested": "Unknown",
        "cpt_suggested": "Unknown",
        "documentation_completeness": "Unable to assess in mock mode. Configure ANTHROPIC_API_KEY for live LLM extraction.",
    },
}

MOCK_SYNTHESES = {
    "CLAIM001": {
        "recommendation": "Approve",
        "confidence": 94,
        "explanation": (
            "This claim meets all criteria for approval. The mechanism of injury (fall from "
            "ladder), objective examination findings (point tenderness, ecchymosis, inability "
            "to bear weight), and specific laterality (right foot) are well-documented. Per "
            "Policy POL_001, foot radiography is medically necessary for acute trauma with "
            "fracture suspicion. Coding Rule RULE_001 confirms S92.3510A paired with CPT "
            "73630 is a valid, complete code pair with no documentation deficiencies."
        ),
        "policy_cited": "POL_001",
        "rule_cited": "RULE_001",
    },
    "CLAIM002": {
        "recommendation": "Approve",
        "confidence": 92,
        "explanation": (
            "All USPSTF eligibility criteria for LDCT lung cancer screening are explicitly "
            "confirmed in the note: patient age 62 (within the 50–80 range), 40 pack-year "
            "smoking history exceeding the ≥20 pack-year threshold, active smoker status, "
            "absence of lung cancer symptoms, and documented shared decision-making counseling. "
            "Policy POL_002 is fully satisfied. Coding Rule RULE_002 confirms Z87.891 + G0297 "
            "as the correct code pair for this encounter."
        ),
        "policy_cited": "POL_002",
        "rule_cited": "RULE_002",
    },
    "CLAIM003": {
        "recommendation": "Escalate",
        "confidence": 38,
        "explanation": (
            "Multiple critical documentation gaps prevent confident adjudication. Per Policy "
            "POL_003, hip MRI (CPT 73721) requires documented failure of conservative therapy "
            "(min. 4–6 weeks of PT, NSAIDs, or activity modification), which is entirely absent "
            "from this note. The only clinical finding is 'mild discomfort to palpation' — "
            "insufficient objective evidence per POL_006. Coding Rule RULE_003 additionally "
            "flags a laterality discrepancy: bilateral complaint submitted with left-specific "
            "ICD-10 code M25.552. Human clinical review is necessary before any reimbursement "
            "decision."
        ),
        "policy_cited": "POL_003",
        "rule_cited": "RULE_003",
    },
    "CLAIM004": {
        "recommendation": "Deny",
        "confidence": 91,
        "explanation": (
            "The claim submits a diagnostic knee X-ray (CPT 73560) under a routine preventive "
            "exam diagnosis (ICD-10 Z00.00). Per Policy POL_004, diagnostic imaging is not "
            "covered as an incidental order during preventive visits without a separately "
            "documented clinical indication and supporting diagnosis code. The physician note "
            "explicitly states no musculoskeletal complaints and confirms no knee examination "
            "was performed. Coding Rule RULE_004 confirms Z00.xx does not support any standalone "
            "imaging procedure."
        ),
        "policy_cited": "POL_004",
        "rule_cited": "RULE_004",
    },
    "CLAIM005": {
        "recommendation": "Deny",
        "confidence": 82,
        "explanation": (
            "A critical billing mismatch exists: PE diagnosis (I26.99) requires CT pulmonary "
            "angiography with contrast (CPT 71275) per Policy POL_005, but the claim was billed "
            "as non-contrast CT chest (CPT 71250). The radiology technician note acknowledges "
            "the procedure was performed without contrast due to an undocumented shellfish "
            "allergy — no pre-medication protocol was completed. Per Coding Rule RULE_005 and "
            "Policy POL_007, this represents a direct code pair mismatch requiring corrected "
            "billing and clinical review."
        ),
        "policy_cited": "POL_005",
        "rule_cited": "RULE_005",
    },
    "CUSTOM": {
        "recommendation": "Escalate",
        "confidence": 50,
        "explanation": (
            "Mock mode is active — full synthesis is unavailable for custom notes. "
            "Please configure your ANTHROPIC_API_KEY environment variable to enable live "
            "LLM-powered analysis of custom physician notes. This is a placeholder response."
        ),
        "policy_cited": "POL_006",
        "rule_cited": "RULE_001",
    },
}
