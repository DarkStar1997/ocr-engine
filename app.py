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
from fastapi.middleware.cors import CORSMiddleware
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from openai import OpenAI

app = FastAPI(title="OCR Intake Parser", version="1.2.1")

# ---- CORS (as requested) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        # add prod origins/domains here as needed
        # "https://your-frontend.example.com",
    ],
    allow_credentials=False,             # you aren't using cookies
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "X-API-Key",
        "Content-Type",
        "Accept",
        # optionally: "X-Requested-With"
    ],
    expose_headers=[],                   # add if you need to read custom response headers
    max_age=86400,                       # cache preflight 1 day
)

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

def intake_schema_rich() -> Dict:
    field_obj = {
        "type": "object",
        "properties": {
            "value":  {"type": "string"},
            "conf":   {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "source": {"type": "string"},
        },
        "required": ["value", "conf", "source"],
        "additionalProperties": False,
    }
    props = {f["key"]: field_obj for f in INTAKE_FIELDS}
    return {
        "name": "IntakeExtractionWithConfidence",
        "schema": {"type": "object", "properties": props, "additionalProperties": False},
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

def clamp_conf(x):
    try:
        v = float(x)
    except Exception:
        return 0.5
    if v < 0: v = 0.0
    if v > 1: v = 1.0
    return v

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

def structured_from_messages(messages: list, schema: dict) -> Dict:
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
        strong = list(messages)
        for blk in strong[-1]["content"]:
            if blk.get("type") == "input_text":
                blk["text"] += (
                    "\n\nReturn ONLY a minified JSON object (no prose, no code fences). "
                    "For each present field include: {\"value\": string, \"conf\": number 0-1, \"source\": string}."
                )
                break
        raw = call_responses(strong, MODEL)
        out = first_json_object(raw)
        if out:
            return out
        raw2 = call_responses(strong, MODEL_FALLBACK)
        return first_json_object(raw2) or {}

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

# =========================
# Single-request document extraction
# =========================
DOC_INSTR_BASE = (
    "You are an information extraction engine. Extract ONLY the requested fields from the provided content. "
    "For each field you find, return an object with keys: value (string), conf (0-1, 1=very confident), "
    "and source (use the format 'FILENAME#page N'). Omit any field that is not confidently present. "
    "Do not fabricate values. Prioritize printed text on the images; you may also use the provided raw text."
)

def ocr_document_structured(
    filename: str,
    page_dataurls: List[Tuple[int, str]],  # (page_number starting at 1, data_url)
    selectable_text: str | None,
    translate_always: bool,
) -> Dict:
    labels_hint = "; ".join([f["label"] for f in INTAKE_FIELDS])
    sys_msg = DOC_INSTR_BASE
    if translate_always:
        sys_msg += " If any content is in Hindi (Devanagari), output the English translation in the field 'value'."
    user_intro = (
        f"Document name: {filename}. Extract only these fields: {labels_hint}. "
        "Always include 'source' as 'FILENAME#page N' where N is the image page you used."
    )

    content = [{"type": "input_text", "text": user_intro}]
    for (pno, url) in page_dataurls:
        content.append({"type": "input_text", "text": f"Page {pno}:"})
        content.append({"type": "input_image", "image_url": url, "detail": "high"})
    if selectable_text:
        content.append({"type": "input_text", "text": "Raw selectable text from the document follows:"})
        content.append({"type": "input_text", "text": selectable_text})

    messages = [
        {"role": "system", "content": [{"type": "input_text", "text": sys_msg}]},
        {"role": "user", "content": content},
    ]

    out = structured_from_messages(messages, intake_schema_rich())

    for k, obj in list(out.items()):
        if not isinstance(obj, dict):
            out[k] = {"value": str(obj), "conf": 0.5, "source": f"{filename}#page 1"}
            obj = out[k]
        obj["conf"] = clamp_conf(obj.get("conf", 0.5))
        if not isinstance(obj.get("source"), str):
            obj["source"] = f"{filename}#page 1"
        val = obj.get("value", "")
        if translate_always and isinstance(val, str) and re.search(r"[\u0900-\u097F]", val):
            obj["value"] = translate_to_english(val)

    return out

# =========================
# PDF compression (default ON)
# =========================
def compress_pdf_bytes(blob: bytes) -> bytes:
    """
    Rebuilds the PDF as image-only pages (JPEG) to shrink size and speed up OCR.
    Defaults: 144 DPI, JPEG quality ~60.
    """
    target_dpi = 144
    jpeg_quality = 60

    src = fitz.open(stream=blob, filetype="pdf")
    try:
        out = fitz.open()
        scale = max(1.0, target_dpi / 72.0)
        mat = fitz.Matrix(scale, scale)
        for page in src:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
            img_bytes = buf.getvalue()

            w_pt = (pix.width * 72.0) / float(target_dpi)
            h_pt = (pix.height * 72.0) / float(target_dpi)
            newp = out.new_page(width=w_pt, height=h_pt)
            rect = fitz.Rect(0, 0, w_pt, h_pt)
            newp.insert_image(rect, stream=img_bytes)

        return out.tobytes(deflate=True, garbage=4, clean=True, linear=True)
    finally:
        src.close()

# =========================
# Renderers for file types
# =========================
def extract_from_pdf(
    blob: bytes,
    filename: str,
    translate_always: bool,
    zoom: float = 2.5  # unused now; kept for signature compatibility
) -> Dict:
    """
    Low-memory path: render each page to a capped-size JPEG and send
    those images to the model in a single request.
    - Caps longer side to ~1600 px
    - JPEG quality ~55 (great OCR tradeoff)
    - Frees per-page buffers immediately
    """
    max_side_px = 1600
    jpeg_quality = 55

    page_urls: List[Tuple[int, str]] = []
    with fitz.open(stream=blob, filetype="pdf") as doc:
        for i, page in enumerate(doc, start=1):
            w_pt = float(page.rect.width)
            h_pt = float(page.rect.height)

            # Compute a safe scale so the longer side ~= max_side_px
            cap_scale = max_side_px / max(w_pt, h_pt) if max(w_pt, h_pt) > 0 else 1.0
            scale = max(1.0, min(2.0, cap_scale))  # keep it reasonable

            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)

            # Convert to PIL and hard-cap size; then JPEG-encode
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img.thumbnail((max_side_px, max_side_px))

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True, progressive=True)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            data_url = f"data:image/jpeg;base64,{b64}"
            page_urls.append((i, data_url))

            # Free memory ASAP
            img.close()
            del pix, buf

    return ocr_document_structured(
        filename,
        page_urls,
        selectable_text=None,
        translate_always=translate_always,
    )

def extract_from_image(filename: str, blob: bytes, translate_always: bool) -> Dict:
    page_urls = [(1, file_to_data_url(filename, blob))]
    return ocr_document_structured(filename, page_urls, selectable_text=None, translate_always=translate_always)

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

def extract_from_docx(blob: bytes, filename: str, translate_always: bool) -> Dict:
    doc = Document(io.BytesIO(blob))
    text_block = extract_docx_text(doc)
    if text_block and translate_always and re.search(r"[\u0900-\u097F]", text_block):
        text_block = translate_to_english(text_block)
    page_urls: List[Tuple[int, str]] = []
    pno = 0
    for rel in list(doc.part.rels.values()):
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
            pno += 1
            page_urls.append((pno, bytes_to_data_url(raw, mime=mime)))

    return ocr_document_structured(filename, page_urls or [(1, file_to_data_url(filename, blob))], text_block, translate_always)

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
        return extract_from_pdf(blob, filename, translate_always)
    elif kind == "docx":
        return extract_from_docx(blob, filename, translate_always)
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

    return PlainTextResponse(content=json.dumps(result, ensure_ascii=False), media_type="application/json")

# Optional health for liveness checks
@app.get("/health")
def health():
    if not SECRET_API_KEY:
        return {"ok": False, "error": "SECRET_API_KEY not set"}
    return {"ok": True}
