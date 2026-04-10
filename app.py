import json
import os
from io import BytesIO
from datetime import datetime, date

import pandas as pd
import PyPDF2
import streamlit as st
from docx import Document
from openai import OpenAI

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Procurement Constraint Intelligence",
    page_icon="📄",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to bottom, #f8fbff, #eef4f9);
    }
    .card {
        background: white;
        padding: 1rem 1.2rem;
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Session state
# -----------------------------
if "doc_outputs" not in st.session_state:
    st.session_state.doc_outputs = []

# -----------------------------
# OpenAI client
# -----------------------------
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            api_key = None

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Set it in Command Prompt before running the app "
            "or add it to Streamlit secrets after deployment."
        )

    return OpenAI(api_key=api_key)

# -----------------------------
# Structured output schema
# -----------------------------
DOCUMENT_SCHEMA = {
    "name": "procurement_document_extraction",
    "schema": {
        "type": "object",
        "properties": {
            "supplier_name": {"type": ["string", "null"]},
            "document_type": {"type": ["string", "null"]},
            "document_name": {"type": ["string", "null"]},
            "product_scope": {"type": ["string", "null"]},

            "moq": {"type": ["string", "null"]},
            "order_multiple": {"type": ["string", "null"]},
            "lead_time": {"type": ["string", "null"]},
            "payment_terms": {"type": ["string", "null"]},
            "penalties": {"type": ["string", "null"]},
            "delivery_restrictions": {"type": ["string", "null"]},
            "cancellation_conditions": {"type": ["string", "null"]},

            "conditions": {
                "type": "array",
                "items": {"type": "string"}
            },
            "order_deadline": {"type": ["string", "null"]},
            "conflicts_or_ambiguities": {
                "type": "array",
                "items": {"type": "string"}
            },
            "missing_critical_fields": {
                "type": "array",
                "items": {"type": "string"}
            },
            "overall_confidence": {
                "type": "string",
                "enum": ["High", "Medium", "Low"]
            },
            "evidence": {
                "type": "object",
                "properties": {
                    "supplier_name": {"type": ["string", "null"]},
                    "product_scope": {"type": ["string", "null"]},
                    "moq": {"type": ["string", "null"]},
                    "order_multiple": {"type": ["string", "null"]},
                    "lead_time": {"type": ["string", "null"]},
                    "payment_terms": {"type": ["string", "null"]},
                    "penalties": {"type": ["string", "null"]},
                    "delivery_restrictions": {"type": ["string", "null"]},
                    "cancellation_conditions": {"type": ["string", "null"]},
                    "order_deadline": {"type": ["string", "null"]}
                },
                "required": [
                    "supplier_name",
                    "product_scope",
                    "moq",
                    "order_multiple",
                    "lead_time",
                    "payment_terms",
                    "penalties",
                    "delivery_restrictions",
                    "cancellation_conditions",
                    "order_deadline"
                ],
                "additionalProperties": False
            }
        },
        "required": [
            "supplier_name",
            "document_type",
            "document_name",
            "product_scope",
            "moq",
            "order_multiple",
            "lead_time",
            "payment_terms",
            "penalties",
            "delivery_restrictions",
            "cancellation_conditions",
            "conditions",
            "order_deadline",
            "conflicts_or_ambiguities",
            "missing_critical_fields",
            "overall_confidence",
            "evidence"
        ],
        "additionalProperties": False
    },
    "strict": True
}

# -----------------------------
# File parsing helpers
# -----------------------------
def detect_document_type(filename: str) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        return "PDF"
    if name.endswith(".docx"):
        return "DOCX"
    if name.endswith(".txt"):
        return "TXT"
    if name.endswith(".xlsx"):
        return "XLSX"
    if name.endswith(".csv"):
        return "CSV"
    return "Unknown"

def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)
    return "\n".join(pages).strip()

def extract_docx_text(file_bytes: bytes) -> str:
    doc = Document(BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs).strip()

def dataframe_to_text(df: pd.DataFrame, label: str) -> str:
    lines = [f"Source table: {label}"]
    headers = [str(col).strip() for col in df.columns.tolist()]
    for _, row in df.iterrows():
        row_parts = []
        for h, v in zip(headers, row.tolist()):
            value = "" if pd.isna(v) else str(v).strip()
            row_parts.append(f"{h}: {value}")
        lines.append(" | ".join(row_parts))
    return "\n".join(lines)

def extract_excel_text(file_bytes: bytes) -> str:
    excel = pd.ExcelFile(BytesIO(file_bytes))
    sheet_texts = []
    for sheet_name in excel.sheet_names:
        df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name)
        sheet_texts.append(dataframe_to_text(df, sheet_name))
    return "\n\n".join(sheet_texts).strip()

def extract_csv_text(file_bytes: bytes) -> str:
    df = pd.read_csv(BytesIO(file_bytes))
    return dataframe_to_text(df, "CSV").strip()

def extract_text(uploaded_file) -> str:
    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if file_name.endswith(".pdf"):
        return extract_pdf_text(file_bytes)
    if file_name.endswith(".docx"):
        return extract_docx_text(file_bytes)
    if file_name.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore").strip()
    if file_name.endswith(".xlsx"):
        return extract_excel_text(file_bytes)
    if file_name.endswith(".csv"):
        return extract_csv_text(file_bytes)

    return ""

# -----------------------------
# Tracker helpers
# -----------------------------
def parse_deadline(deadline_text):
    if not deadline_text:
        return None

    formats = [
        "%Y-%m-%d",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(deadline_text.strip(), fmt).date()
        except Exception:
            continue

    return None

def get_status(deadline_text):
    parsed = parse_deadline(deadline_text)

    if not parsed:
        return "No deadline"

    days = (parsed - date.today()).days

    if days < 0 or days <= 7:
        return "Urgent"
    elif days <= 21:
        return "Upcoming"
    else:
        return "Planned"

# -----------------------------
# LLM extraction
# -----------------------------
def extract_constraints_llm(text: str, document_name: str, document_type: str) -> dict:
    client = get_openai_client()

    prompt = f"""
You are an expert procurement contract analyst.

Your task is to extract procurement and operational constraints from supplier documents and convert them into structured JSON.

The document may be a contract, agreement, supplier policy, ordering guideline, email, spreadsheet-derived text, or mixed-format commercial content.

Your role is not just to find keywords. Your role is to interpret procurement rules and convert them into planner-ready structured outputs.

--------------------------------------------------
STEP 1: INTERNAL ANALYSIS (DO NOT OUTPUT)
--------------------------------------------------

First, internally analyze the document and identify:
- procurement constraints
- conditional rules
- category-based rules
- multiple values for the same field
- conflicting clauses
- missing or ambiguous fields

Then convert the result into structured JSON.

Do not output your reasoning.

--------------------------------------------------
STEP 2: GENERAL EXTRACTION RULES
--------------------------------------------------

1. Extract values even if they are complex, conditional, or multi-valued.
2. Do not return null if information exists in any form.
3. Only return null if the field is completely missing.
4. If multiple values exist, return a structured summary string.
5. If conditions exist, preserve them explicitly.
6. If conflicts exist, populate both the field and conflicts_or_ambiguities.
7. Do not oversimplify business rules into a single value when the document contains multiple applicable rules.
8. Prefer structured summarization over omission.
9. Prefer outputs that help a procurement planner compare suppliers and make decisions.

--------------------------------------------------
STEP 3: FIELD-SPECIFIC RULES
--------------------------------------------------

supplier_name:
- extract exact supplier name

product_scope:
- summarize all relevant product categories

moq:
- preserve category-specific, conditional, and conflicting MOQ rules
- if multiple MOQ rules exist, return a structured summary string
- example:
  "Cotton: 600; Polyester: 800; Blended: 700; Long-term partners: 500 (conditional); Internal policy conflict: 650"

order_multiple:
- preserve standard and special order multiple rules
- example:
  "50 standard; 100 for bulk agreements"

lead_time:
- preserve all scenario-based lead times
- example:
  "Off-season: 10–12 days; Peak season: 18–25 days; Repeat orders: 8–10 days"

payment_terms:
- preserve standard and conditional payment structures
- example:
  "Net 45 standard; New clients: 50% advance + balance before shipment"

penalties:
- summarize all penalties clearly

delivery_restrictions:
- summarize all delivery limitations or conditions

cancellation_conditions:
- summarize cancellation restrictions and fees

conditions:
- list additional operational or commercial conditions not already captured in the core fields

order_deadline:
- extract the exact deadline date or deadline statement

--------------------------------------------------
STEP 4: CONFLICT HANDLING
--------------------------------------------------

If conflicting information exists:
- still populate the field using a structured summary
- also record the conflict in "conflicts_or_ambiguities"

--------------------------------------------------
STEP 5: MISSING FIELDS
--------------------------------------------------

If a critical field is missing, add it to "missing_critical_fields".

Critical fields:
- supplier_name
- product_scope
- moq
- lead_time
- payment_terms
- order_deadline

--------------------------------------------------
STEP 6: CONFIDENCE SCORING
--------------------------------------------------

Assign overall_confidence:
- High = explicit, consistent, and complete
- Medium = conditional, multi-valued, or partially conflicting
- Low = ambiguous or incomplete

--------------------------------------------------
STEP 7: EVIDENCE
--------------------------------------------------

For each field, provide supporting text copied or closely paraphrased from the document.

--------------------------------------------------
FINAL OUTPUT RULES
--------------------------------------------------

- Return strict JSON only
- Do not include explanations
- Do not include markdown
- Do not omit fields when evidence exists
- Ensure output matches the schema exactly

--------------------------------------------------
DOCUMENT INFO
--------------------------------------------------

Document name: {document_name}
Document type: {document_type}

--------------------------------------------------
DOCUMENT TEXT
--------------------------------------------------

{text}
"""

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": DOCUMENT_SCHEMA["name"],
                "schema": DOCUMENT_SCHEMA["schema"],
                "strict": True,
            }
        },
    )

    return json.loads(response.output_text)

# -----------------------------
# UI
# -----------------------------
with st.sidebar:
    st.header("Workflow")
    st.markdown("""
1. Upload multiple supplier documents  
2. Parse PDF / DOCX / TXT / XLSX / CSV  
3. Extract constraints with LLM  
4. Review tracker + JSON outputs
""")
    st.info("This PoC directly validates LLM use and prompt engineering.")
    st.warning("Use text-readable files for best results.")

st.title("Procurement Constraint Intelligence")
st.caption("LLM-based document extraction proof of concept")

st.markdown(
    """
    <div class="card">
        <strong>Purpose of this PoC</strong><br>
        Demonstrate that a strong prompt plus an LLM can extract structured procurement
        constraints from supplier documents and convert them into a planner-facing tracker.
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    "Upload supplier documents",
    type=["pdf", "docx", "txt", "xlsx", "csv"],
    accept_multiple_files=True
)

col1, col2, col3 = st.columns(3)
col1.metric("Input Types", "PDF / DOCX / TXT / XLSX / CSV")
col2.metric("Extraction Engine", "LLM")
col3.metric("Output", "JSON + Tracker")

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded.")

    if st.button("Analyze Documents"):
        outputs = []

        with st.spinner("Reading files and extracting procurement constraints..."):
            for file in uploaded_files:
                doc_type = detect_document_type(file.name)
                extracted_text = extract_text(file)

                if not extracted_text:
                    outputs.append(
                        {
                            "supplier_name": None,
                            "document_type": doc_type,
                            "document_name": file.name,
                            "product_scope": None,
                            "moq": None,
                            "order_multiple": None,
                            "lead_time": None,
                            "payment_terms": None,
                            "penalties": None,
                            "delivery_restrictions": None,
                            "cancellation_conditions": None,
                            "conditions": [],
                            "order_deadline": None,
                            "conflicts_or_ambiguities": ["No readable text could be extracted from this file."],
                            "missing_critical_fields": [
                                "supplier_name",
                                "product_scope",
                                "moq",
                                "lead_time",
                                "payment_terms",
                                "order_deadline",
                            ],
                            "overall_confidence": "Low",
                            "evidence": {
                                "supplier_name": None,
                                "product_scope": None,
                                "moq": None,
                                "order_multiple": None,
                                "lead_time": None,
                                "payment_terms": None,
                                "penalties": None,
                                "delivery_restrictions": None,
                                "cancellation_conditions": None,
                                "order_deadline": None,
                            },
                        }
                    )
                    continue

                output = extract_constraints_llm(
                    text=extracted_text,
                    document_name=file.name,
                    document_type=doc_type,
                )
                outputs.append(output)

        st.session_state.doc_outputs = outputs

# -----------------------------
# Display outputs + tracker
# -----------------------------
if st.session_state.doc_outputs:
    st.markdown("---")
    st.subheader("📊 Active Supplier Tracker")

    tracker_rows = []

    for doc in st.session_state.doc_outputs:
        tracker_rows.append({
            "Supplier": doc.get("supplier_name"),
            "Document": doc.get("document_name"),
            "Product Scope": doc.get("product_scope"),
            "MOQ": doc.get("moq"),
            "Order Multiple": doc.get("order_multiple"),
            "Lead Time": doc.get("lead_time"),
            "Payment Terms": doc.get("payment_terms"),
            "Deadline": doc.get("order_deadline"),
            "Status": get_status(doc.get("order_deadline")),
            "Confidence": doc.get("overall_confidence"),
        })

    tracker_df = pd.DataFrame(tracker_rows)

    if not tracker_df.empty:
        st.dataframe(tracker_df, use_container_width=True)

        tracker_csv = tracker_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Tracker CSV",
            tracker_csv,
            "supplier_tracker.csv",
            "text/csv"
        )
    else:
        st.info("No tracker data available.")

    st.markdown("---")
    st.subheader("Per-Document JSON Outputs")

    for i, output in enumerate(st.session_state.doc_outputs, start=1):
        title = output.get("document_name") or f"Document {i}"
        with st.expander(f"{i}. {title}", expanded=(i == 1)):
            st.json(output)

st.markdown("---")
if st.button("Reset"):
    st.session_state.doc_outputs = []
    st.rerun()
