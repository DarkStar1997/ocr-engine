# app.py
import io
import os
import re
import json
import base64
import mimetypes
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status
from fastapi.responses import PlainTextResponse
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from openai import OpenAI

app = FastAPI(title="OCR Intake Parser", version="1.0.0")

# =========================
# Config / Model selection
# =========================
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
MODEL_FALLBACK = os.getenv("OPENAI_MODEL_FALLBACK", "gpt-4.1-mini")

# OpenAI client with generous timeout + retries
CLIENT = OpenAI(timeout=600.0, max_retries=5)

# =========================
# Auth (SECRET_API_KEY)
# =========================
SECRET_API_KEY = os.getenv("SECRET_API_KEY")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_http_bearer = HTTPBearer(auto_error=False)

def require_api_key(
    api_key_header: str | None = Depends(_api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Depends(_http_bearer),
):
    if not SECRET_API_KEY:
        # Fail closed if you forgot to set the key on the server
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server not configured with SECRET_API_KEY",
        )
    supplied = None
    if api_key_header:
        supplied = api_key_header.strip()
    elif bearer and bearer.scheme.lower() == "bearer":
        supplied = (bearer.credentials or "").strip()
    if not supplied or supplied != SECRET_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return True

# =========================
# Intake fields schema
# =========================
INTAKE_FIELDS = [
  { "key": "applicant_name", "label": "Applicant Name" },
  { "key": "co_applicant_name", "label": "Co-Applicant Name" },
  { "key": "property_owner_name", "label": "Name of the Property Owner" },
  { "key": "loan_type", "label": "Loan Type / Product Type" },
  { "key": "application_number", "label": "Application Number" },
  { "key": "developer_name", "label": "Developer’s Name" },
  { "key": "flat", "label": "Flat" },
  { "key": "property_address_doc", "label": "Property Address (as per Document)" },
  { "key": "pincode", "label": "PIN Code (as per Document)" },
  { "key": "usage_approved_as_per_plan", "label": "Usage approved as per Plan" },
  { "key": "usage_as_per_cdp", "label": "Usage as per CDP/Master plan" },
  { "key": "address_as_per_plan", "label": "Address as per Plan" },
  { "key": "legal_documents", "label": "Legal Documents" },
  { "key": "list_of_documents", "label": "List of Documents provided" },
  { "key": "land_freehold_or_leasehold", "label": "Land Freehold / Leasehold, term of lease" },
  { "key": "period_expired_balance_lease_rent", "label": "Period expired, balance and lease rent" },
  { "key": "boundaries_deed_east", "label": "Boundaries EAST (As per Deed)" },
  { "key": "boundaries_deed_west", "label": "Boundaries WEST (As per Deed)" },
  { "key": "boundaries_deed_north", "label": "Boundaries NORTH (As per Deed)" },
  { "key": "boundaries_deed_south", "label": "Boundaries SOUTH (As per Deed)" },
  { "key": "dimensions_deed_east", "label": "Dimension EAST (As per Deed)" },
  { "key": "dimensions_deed_west", "label": "Dimension WEST (As per Deed)" },
  { "key": "dimensions_deed_north", "label": "Dimension NORTH (As per Deed)" },
  { "key": "dimensions_deed_south", "label": "Dimension SOUTH (As per Deed)" },
  { "key": "land_area_docs_sqyd", "label": "Land Area (Sqyds) as per Documents" },
  { "key": "land_area_docs_sqmt", "label": "Land Area (Sqmt) as per Documents" },
  { "key": "land_area_docs_sqft", "label": "Land Area (Sqft) as per Documents" },
  { "key": "land_area_plan", "label": "Land/Plot Area as per plan" },
  { "key": "type_of_plot", "label": "Type of plot" },
  { "key": "final_land_area_uds", "label": "Final Land area / UDS considered" },
  { "key": "document_area_sqft", "label": "Document Area (title deed)" },
  { "key": "approved_area_plan_sqft", "label": "Approved Area (as per plan)" },
  { "key": "sanction_approval_no", "label": "Sanctioned plans / approval no" },
  { "key": "sanction_number_date", "label": "Number and Date" },
  { "key": "property_documents", "label": "Property documents" },
  { "key": "ownership_type", "label": "Ownership type (Leasehold/Freehold)" },
  { "key": "amenities_idc", "label": "IDC" },
  { "key": "amenities_edc", "label": "EDC" },
  { "key": "amenities_power_backup", "label": "Power Backup" },
  { "key": "amenities_plc", "label": "PLC" },
  { "key": "amenities_car_parking", "label": "Car Parking" },
  { "key": "amenities_others", "label": "Others" },
  { "key": "floors_sanctioned", "label": "No. Of Floors Sanctioned" },
  { "key": "floors_proposed", "label": "No. of floors Proposed" },
]

def intake_schema() -> Dict:
    props = {f["key"]: {"type": "string"} for f in INTAKE_FIELDS}
    return {
        "name": "IntakeExtraction",
        "schema": {
            "type": "object",
            "properties": props,
            "additionalProperties": False,
        },
        "strict": True,
    }

# =========================
# Utils
# =========================
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

def first_json_object(text: str) -> Dict:
    if not text:
        return {}
    m = JSON_OBJ_RE.search(text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}

def merge_keep_longer(dst: Dict, src: Dict) -> Dict:
    for k, v in (src or {}).items():
        if not v:
            continue
        if k not in dst or len(str(v)) > len(str(dst.get(k, ""))):
            dst[k] = v
    return dst

def img_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"

def bytes_to_data_url(blob: bytes, mime: str = "image/png") -> str:
    b64 = base64.b64encode(blob).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def file_to_data_url(filename: str, blob: bytes) -> str:
    mime, _ = mimetypes.guess_type(filename)
    if not mime:
        mime = "image/png"
    b64 = base64.b64encode(blob).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def extract_text_from_response(resp) -> str:
    txt = getattr(resp, "output_text", None)
    if txt:
        return txt.strip()
    out_parts: List[str] = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) == "message":
            for part in getattr(item, "content", []) or []:
                if getattr(part, "type", None) in ("output_text", "text"):
                    piece = getattr(part, "text", None)
                    if piece:
                        out_parts.append(piece)
    return "\n".join(out_parts).strip()

def call_responses(messages: list, model: str = MODEL) -> str:
    resp = CLIENT.responses.create(model=model, input=messages)
    return extract_text_from_response(resp)

def translate_to_english(text: str) -> str:
    if not text.strip():
        return text
    messages = [
        {"role": "system", "content": [{"type": "input_text", "text":
            "You are a professional translator. Translate to natural, fluent English. Preserve line breaks; no commentary."}]},
        {"role": "user", "content": [{"type": "input_text", "text": text}]},
    ]
    out = call_responses(messages, MODEL)
    if not out or out.lower().startswith(("sorry", "i can't", "i cannot", "i am sorry")):
        out = call_responses(messages, MODEL_FALLBACK)
    return out.strip()

def structured_from_messages(messages: list, schema: dict) -> Dict:
    """
    Try JSON Schema mode. If the SDK is older (no response_format), fall back
    to a strong "return only JSON" prompt and parse.
    """
    try:
        resp = CLIENT.responses.create(
            model=MODEL,
            input=messages,
            response_format={"type": "json_schema", "json_schema": schema},
        )
        out = getattr(resp, "output_parsed", None)
        if out is not None:
            return out
        return first_json_object(extract_text_from_response(resp))
    except TypeError:
        # Old SDK: enforce JSON via prompt
        strong = list(messages)
        for blk in strong[-1]["content"]:
            if blk.get("type") == "input_text":
                blk["text"] += "\n\nReturn ONLY a minified JSON object (no prose, no code fences)."
                break
        raw = call_responses(strong, MODEL)
        out = first_json_object(raw)
        if out:
            return out
        raw2 = call_responses(strong, MODEL_FALLBACK)
        return first_json_object(raw2) or {}

# =========================
# OCR → Structured extract
# =========================
def ocr_image_dataurl_structured(data_url: str, translate_always: bool) -> Dict:
    labels_hint = "; ".join([f["label"] for f in INTAKE_FIELDS])
    messages = [
        {"role": "system", "content": [{"type": "input_text", "text":
            "You are an information extraction engine. Extract ONLY requested fields; return strict JSON; omit missing; no fabrication."}]},
        {"role": "user", "content": [
            {"type": "input_text", "text":
                ("Extract the following labeled fields from this page. "
                 "Return only the JSON object. "
                 + ("If a value is in Hindi, translate to English. " if translate_always else "")
                 + "Do not include fields you cannot find. "
                 f"Fields: {labels_hint}"
                )},
            {"type": "input_image", "image_url": data_url, "detail": "high"},
        ]},
    ]
    out = structured_from_messages(messages, intake_schema())
    if translate_always:
        for k, v in list(out.items()):
            if isinstance(v, str) and DEVANAGARI_RE.search(v):
                out[k] = translate_to_english(v)
    return out

def extract_from_pdf(blob: bytes, translate_always: bool, zoom: float = 2.5) -> Dict:
    merged: Dict = {}
    with fitz.open(stream=blob, filetype="pdf") as doc:
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            data_url = img_to_data_url(img, fmt="PNG")
            d = ocr_image_dataurl_structured(data_url, translate_always)
            merged = merge_keep_longer(merged, d)
    return merged

def extract_from_image(filename: str, blob: bytes, translate_always: bool) -> Dict:
    data_url = file_to_data_url(filename, blob)
    return ocr_image_dataurl_structured(data_url, translate_always)

def extract_docx_text(doc: Document) -> str:
    parts: List[str] = []
    for p in doc.paragraphs:
        if p.text.strip():
            parts.append(p.text)
    for tbl in doc.tables:
        for row in tbl.rows:
            row_text = [cell.text.strip() for cell in row.cells]
            if any(row_text):
                parts.append(" | ".join(row_text))
    return "\n".join(parts).strip()

def extract_from_docx(blob: bytes, translate_always: bool) -> Dict:
    doc = Document(io.BytesIO(blob))
    merged: Dict = {}

    # 1) selectable text -> structured
    selectable = extract_docx_text(doc)
    if selectable:
        if translate_always and DEVANAGARI_RE.search(selectable):
            selectable = translate_to_english(selectable)
        messages = [
            {"role": "system", "content": [{"type": "input_text", "text":
                "You are an information extraction engine. Extract ONLY requested fields; return strict JSON; omit missing; no fabrication."}]},
            {"role": "user", "content": [{"type": "input_text", "text":
                "Extract ONLY the requested fields from this text. "
                "Return only the JSON object. If a field is not found, omit it.\n\nText:\n" + selectable}]},
        ]
        d1 = structured_from_messages(messages, intake_schema())
        merged = merge_keep_longer(merged, d1)

    # 2) embedded images -> OCR -> structured
    rels = list(doc.part.rels.values())
    for rel in rels:
        if rel.reltype == RT.IMAGE:
            part = rel.target_part
            raw = part.blob
            ext = (part.partname.ext or "").lower().lstrip(".")
            if ext in ("jpg", "jpeg"):
                mime = "image/jpeg"
            elif ext == "png":
                mime = "image/png"
            elif ext == "gif":
                mime = "image/gif"
            elif ext == "bmp":
                mime = "image/bmp"
            elif ext in ("tiff", "tif"):
                mime = "image/tiff"
            else:
                mime = "image/png"
            data_url = bytes_to_data_url(raw, mime=mime)
            d2 = ocr_image_dataurl_structured(data_url, translate_always)
            merged = merge_keep_longer(merged, d2)

    return merged

def detect_type(filename: str, blob: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext in {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".docx"}:
        if ext == ".pdf":
            return "pdf"
        if ext == ".docx":
            return "docx"
        return "image"
    mime, _ = mimetypes.guess_type(filename)
    if mime == "application/pdf":
        return "pdf"
    if mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return "docx"
    if mime and mime.startswith("image/"):
        return "image"
    if blob[:5] == b"%PDF-":
        return "pdf"
    return "image"

def parse_file(filename: str, blob: bytes, lang: str) -> Dict:
    translate_always = (lang or "en").lower() == "hi"
    kind = detect_type(filename, blob)
    if kind == "pdf":
        return extract_from_pdf(blob, translate_always)
    elif kind == "docx":
        return extract_from_docx(blob, translate_always)
    else:
        return extract_from_image(filename, blob, translate_always)

# =========================
# API: /parse
# =========================
@app.post("/parse", response_class=PlainTextResponse)
async def parse(
    files: List[UploadFile] = File(..., description="List of files (PDF/PNG/JPG/WEBP/DOCX)"),
    langs: List[str] = Form(..., description="List of language tags (en/hi) in the same order as files"),
    _auth_ok: bool = Depends(require_api_key),
) -> PlainTextResponse:
    # Robust array handling for Swagger/curl
    files = [f for f in files if getattr(f, "filename", None)]
    if len(langs) == 1:
        langs = [s.strip() for s in re.split(r"[,\s]+", langs[0]) if s.strip()]
    langs = [l.strip().lower() for l in langs if l and l.strip()]
    if len(langs) < len(files):
        langs += ["en"] * (len(files) - len(langs))
    elif len(langs) > len(files):
        langs = langs[:len(files)]
    bad = [l for l in langs if l not in {"en", "hi"}]
    if bad:
        raise HTTPException(status_code=400, detail=f"Invalid language(s): {bad}. Use 'en' or 'hi'.")

    result: Dict[str, str] = {}
    for idx, uf in enumerate(files):
        blob = await uf.read()
        lang = langs[idx]
        parsed_obj = parse_file(uf.filename, blob, lang)
        result[uf.filename] = json.dumps(parsed_obj, ensure_ascii=False)

    # Return a JSON string mapping filename -> JSON string of extracted fields
    return PlainTextResponse(content=json.dumps(result, ensure_ascii=False), media_type="application/json")

# Optional health for liveness checks
@app.get("/health")
def health():
    if not SECRET_API_KEY:
        return {"ok": False, "error": "SECRET_API_KEY not set"}
    return {"ok": True}
