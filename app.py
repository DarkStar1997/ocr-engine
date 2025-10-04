# app.py
import io
import os
import re
import json
import base64
import mimetypes
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import fitz  # PyMuPDF
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status
from fastapi.responses import PlainTextResponse
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from openai import OpenAI

app = FastAPI(title="OCR Intake Parser", version="1.7.0")

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        # add prod origins/domains here as needed
        # "https://your-frontend.example.com",
    ],
    allow_credentials=False,             # no cookies
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "X-API-Key",
        "Content-Type",
        "Accept",
        # optionally: "X-Requested-With"
    ],
    expose_headers=[],
    max_age=86400,
)

# =========================
# Config / Model selection
# =========================
MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
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
    api_key_header: Optional[str] = Depends(_api_key_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Depends(_http_bearer),
):
    if not SECRET_API_KEY:
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
# Dynamic field helpers (labels only -> keys)
# =========================
NON_ALNUM = re.compile(r"[^a-z0-9]+")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

def label_to_key(label: str) -> str:
    s = (label or "").strip().lower()
    s = NON_ALNUM.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "field"
    return s[:80]  # safe cap

def build_all_fields_from_labels(labels: List[str]) -> List[Dict[str, str]]:
    """
    From user-provided labels only, produce [{key,label}, ...] with dedup by key.
    """
    combined: List[Dict[str, str]] = []
    seen: set = set()
    for lbl in labels or []:
        if not lbl:
            continue
        k = label_to_key(lbl)
        if k not in seen:
            combined.append({"key": k, "label": lbl})
            seen.add(k)
    return combined

def intake_schema_rich(all_fields: List[Dict[str, str]]) -> Dict:
    """
    JSON schema where each key maps to {value, conf, source}.
    """
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
    props = {f["key"]: field_obj for f in all_fields}
    return {
        "name": "IntakeExtractionWithConfidence",
        "schema": {"type": "object", "properties": props, "additionalProperties": False},
        "strict": True,
    }

# =========================
# Utils
# =========================
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
    """
    Prefer JSON schema mode; if SDK lacks response_format, fall back to strict JSON prompting.
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
    selectable_text: Optional[str],
    translate_always: bool,
    all_fields: List[Dict[str, str]],
) -> Dict:
    if not all_fields:
        return {}

    labels_hint = "; ".join([f["label"] for f in all_fields])
    mapping_lines = "\n".join([f"- {f['label']} -> {f['key']}" for f in all_fields])

    sys_msg = DOC_INSTR_BASE
    if translate_always:
        sys_msg += " If any content is in Hindi (Devanagari), output the English translation in the field 'value'."

    user_intro = (
        f"Document name: {filename}.\n"
        f"Extract only these fields (labels): {labels_hint}\n"
        "Use the following key names in the JSON output (label -> key):\n"
        f"{mapping_lines}\n"
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

    out = structured_from_messages(messages, intake_schema_rich(all_fields))

    # Normalize conf/source and translate if needed
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
# Low-memory PDF rendering (default behaviour)
# =========================
def extract_from_pdf(
    blob: bytes,
    filename: str,
    translate_always: bool,
    all_fields: List[Dict[str, str]],
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
        all_fields=all_fields,
    )

def extract_from_image(filename: str, blob: bytes, translate_always: bool, all_fields: List[Dict[str, str]]) -> Dict:
    page_urls = [(1, file_to_data_url(filename, blob))]
    return ocr_document_structured(filename, page_urls, selectable_text=None, translate_always=translate_always, all_fields=all_fields)

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

def extract_from_docx(blob: bytes, filename: str, translate_always: bool, all_fields: List[Dict[str, str]]) -> Dict:
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

    return ocr_document_structured(filename, page_urls or [(1, file_to_data_url(filename, blob))], text_block, translate_always, all_fields=all_fields)

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

def parse_file(filename: str, blob: bytes, lang: str, all_fields: List[Dict[str, str]]) -> Dict:
    translate_always = (lang or "en").lower() == "hi"
    kind = detect_type(filename, blob)

    if kind == "pdf":
        return extract_from_pdf(blob, filename, translate_always, all_fields=all_fields)
    elif kind == "docx":
        return extract_from_docx(blob, filename, translate_always, all_fields=all_fields)
    else:
        return extract_from_image(filename, blob, translate_always, all_fields=all_fields)

# =========================
# API: /parse
# =========================
@app.post("/parse", response_class=PlainTextResponse)
async def parse(
    files: List[UploadFile] = File(..., description="List of files (PDF/PNG/JPG/WEBP/DOCX)"),
    langs: List[str] = Form(..., description="List of language tags (en/hi) in the same order as files"),
    fields_to_extract: List[str] = Form(
        ...,
        description="Mandatory: field labels to extract (repeat this field OR provide a comma/newline/semicolon-separated list)",
    ),
    _auth_ok: bool = Depends(require_api_key),
) -> PlainTextResponse:
    # Normalize files
    files = [f for f in files if getattr(f, "filename", None)]

    # Normalize langs (supports 'hi,en' or repeated fields)
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

    # Mandatory fields_to_extract: supports multiple rows or comma/newline/semicolon separated
    if not fields_to_extract:
        raise HTTPException(status_code=400, detail="fields_to_extract is required and cannot be empty.")
    if len(fields_to_extract) == 1:
        labels = [s.strip() for s in re.split(r"[,\n;]+", fields_to_extract[0]) if s.strip()]
    else:
        labels = []
        for item in fields_to_extract:
            labels.extend([s.strip() for s in re.split(r"[,\n;]+", item) if s.strip()])
    labels = [lbl for lbl in labels if lbl]

    if not labels:
        raise HTTPException(status_code=400, detail="fields_to_extract must contain at least one non-empty label.")

    # Build fields list solely from provided labels
    all_fields = build_all_fields_from_labels(labels)

    # Process files
    result: Dict[str, str] = {}
    for idx, uf in enumerate(files):
        blob = await uf.read()
        lang = langs[idx]
        parsed_obj = parse_file(uf.filename, blob, lang, all_fields=all_fields)
        # Contract: filename -> JSON string (minified) of extracted fields
        result[uf.filename] = json.dumps(parsed_obj, ensure_ascii=False)

    return PlainTextResponse(content=json.dumps(result, ensure_ascii=False), media_type="application/json")

# Optional health for liveness checks
@app.get("/health")
def health():
    if not SECRET_API_KEY:
        return {"ok": False, "error": "SECRET_API_KEY not set"}
    return {"ok": True}
