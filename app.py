# app.py
import io
import os
import re
import json
import time
import base64
import logging
import mimetypes
import uuid
import contextvars
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import fitz  # PyMuPDF
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status, Request
from fastapi.responses import PlainTextResponse
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from openai import OpenAI

# =========================
# Logging setup
# =========================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("ocr_engine")

# Request correlation id (per-request)
REQ_ID: contextvars.ContextVar[str] = contextvars.ContextVar("req_id", default="-")
def rid() -> str:
    return REQ_ID.get()

def log(level: int, msg: str, **kv):
    kv_str = " ".join(f"{k}={v}" for k, v in kv.items())
    logger.log(level, f"[rid={rid()}] {msg}" + (f" | {kv_str}" if kv_str else ""))

app = FastAPI(title="OCR Intake Parser", version="2.0.0")

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        # add prod origins/domains here as needed
        "https://ocr.finotoai.com",
    ],
    allow_credentials=False,
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
# Vision (for OCR fallback on scanned pages)
MODEL = os.getenv("OPENAI_MODEL", "gpt-5")  # vision-capable
# Text-only, cheaper model for interpretation
TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL") or os.getenv("OPENAI_MODEL_FALLBACK", "gpt-4.1-mini")

# OpenAI client with generous timeout + retries
CLIENT = OpenAI(timeout=600.0, max_retries=5)
log(logging.INFO, "Server startup", openai_timeout=600, openai_retries=5, ocr_model=MODEL, text_model=TEXT_MODEL)

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
    supplied = None
    if api_key_header:
        supplied = api_key_header.strip()
    elif bearer and bearer.scheme.lower() == "bearer":
        supplied = (bearer.credentials or "").strip()
    if not SECRET_API_KEY:
        log(logging.ERROR, "Auth failed: SECRET_API_KEY not configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server not configured with SECRET_API_KEY",
        )
    if not supplied or supplied != SECRET_API_KEY:
        log(logging.WARNING, "Auth failed: invalid or missing API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return True

# =========================
# Helpers
# =========================
NON_ALNUM = re.compile(r"[^a-z0-9]+")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

def label_to_key(label: str) -> str:
    s = (label or "").strip().lower()
    s = NON_ALNUM.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return (s or "field")[:80]

def intake_schema_rich(fields: List[str]) -> Dict:
    entries = [{"key": label_to_key(lbl), "label": lbl} for lbl in fields]
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
    props = {e["key"]: field_obj for e in entries}
    schema = {
        "name": "IntakeExtractionWithConfidence",
        "schema": {"type": "object", "properties": props, "additionalProperties": False},
        "strict": True,
    }
    log(logging.DEBUG, "Built JSON schema", fields=len(fields), keys=list(props.keys()))
    return schema

def first_json_object(text: str) -> Dict:
    if not text:
        return {}
    m = JSON_OBJ_RE.search(text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception as e:
        log(logging.WARNING, "JSON parse (regex) failed", error=str(e))
        return {}

def clamp_conf(x):
    try:
        v = float(x)
    except Exception:
        return 0.5
    return 0.0 if v < 0 else 1.0 if v > 1 else v

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

def call_responses(messages: list, *, model: str) -> str:
    t0 = time.time()
    try:
        log(logging.INFO, "OpenAI call:start", model=model, stage="responses.create", msg_roles=[m["role"] for m in messages])
        resp = CLIENT.responses.create(model=model, input=messages)
        out = extract_text_from_response(resp)
        log(logging.INFO, "OpenAI call:done", model=model, ms=int((time.time()-t0)*1000), out_len=len(out))
        return out
    except Exception as e:
        log(logging.ERROR, "OpenAI call:error", model=model, ms=int((time.time()-t0)*1000), error=str(e))
        raise

def translate_to_english(text: str) -> str:
    if not text.strip():
        return text
    messages = [
        {"role": "system", "content": [{"type": "input_text", "text":
            "You are a professional translator. Translate to natural, fluent English. Preserve line breaks; no commentary."}]},
        {"role": "user", "content": [{"type": "input_text", "text": text}]},
    ]
    log(logging.INFO, "Translate:start", model=TEXT_MODEL, detected_script="devanagari")
    out = call_responses(messages, model=TEXT_MODEL)
    if not out or out.lower().startswith(("sorry", "i can't", "i cannot", "i am sorry")):
        log(logging.WARNING, "Translate:fallback_to_vision_model", fallback_model=MODEL)
        out = call_responses(messages, model=MODEL)  # fallback
    log(logging.INFO, "Translate:done", out_len=len(out))
    return out.strip()

def text_maybe_translate(text: str, translate: bool) -> str:
    if translate and DEVANAGARI_RE.search(text or ""):
        return translate_to_english(text)
    return text

# ---------- Data URL helpers (for OCR fallback) ----------
def img_to_jpeg_data_url(img: Image.Image, quality: int = 52) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
    b = buf.getvalue()
    log(logging.DEBUG, "Rendered JPEG", bytes=len(b), quality=quality, mode=img.mode, size=f"{img.width}x{img.height}")
    return "data:image/jpeg;base64," + base64.b64encode(b).decode("utf-8")

# =========================
# STAGE 1: OCR (returns page texts)
# =========================
def ocr_pdf_pages(blob: bytes, lang: str) -> List[Tuple[int, str]]:
    """
    For each page:
      1) Try selectable text via PyMuPDF.
      2) If empty, render low-memory grayscale JPEG and OCR via vision model (per page).
    Returns: [(page_number, text), ...]
    """
    translate = (lang or "en").lower() == "hi"
    results: List[Tuple[int, str]] = []
    max_side_px = 1400
    jpeg_quality = 52

    with fitz.open(stream=blob, filetype="pdf") as doc:
        log(logging.INFO, "PDF opened", pages=doc.page_count, translate=translate, max_side_px=max_side_px, jpeg_quality=jpeg_quality)
        for i, page in enumerate(doc, start=1):
            t_page0 = time.time()
            text = (page.get_text("text") or "").strip()
            if text:
                log(logging.DEBUG, "PDF page selectable text", page=i, text_len=len(text))
            if not text:
                w_pt, h_pt = float(page.rect.width), float(page.rect.height)
                cap_scale = max_side_px / max(w_pt, h_pt) if max(w_pt, h_pt) > 0 else 1.0
                scale = max(1.0, min(2.0, cap_scale))
                log(logging.DEBUG, "PDF page render->OCR", page=i, width_pt=w_pt, height_pt=h_pt, scale=round(scale,3))
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csGRAY, alpha=False)
                img = Image.frombytes("L", [pix.width, pix.height], pix.samples)
                img.thumbnail((max_side_px, max_side_px))
                data_url = img_to_jpeg_data_url(img, quality=jpeg_quality)
                img.close(); del pix

                messages = [
                    {"role": "system", "content": [{"type": "input_text", "text":
                        "You perform OCR on a single scanned page. Return only the plain text. Keep original line breaks. No commentary."}]},
                    {"role": "user", "content": [
                        {"type": "input_image", "image_url": data_url, "detail": "high"},
                    ]},
                ]
                log(logging.INFO, "OCR page via OpenAI Vision:start", page=i, model=MODEL)
                text = call_responses(messages, model=MODEL).strip()
                log(logging.INFO, "OCR page via OpenAI Vision:done", page=i, model=MODEL, text_len=len(text))

            text = text_maybe_translate(text, translate)
            if translate:
                log(logging.DEBUG, "Page translated (if needed)", page=i, final_len=len(text))
            results.append((i, text))
            log(logging.DEBUG, "Page done", page=i, ms=int((time.time()-t_page0)*1000))

    return results

def ocr_image_pages(filename: str, blob: bytes, lang: str) -> List[Tuple[int, str]]:
    translate = (lang or "en").lower() == "hi"
    mime, _ = mimetypes.guess_type(filename)
    if not mime or not mime.startswith("image/"):
        mime = "image/jpeg"
    data_url = "data:%s;base64,%s" % (mime, base64.b64encode(blob).decode("utf-8"))
    log(logging.INFO, "Image OCR via OpenAI Vision:start", filename=filename, mime=mime, model=MODEL, bytes=len(blob))
    messages = [
        {"role": "system", "content": [{"type": "input_text", "text":
            "You perform OCR on a single image. Return only the plain text. Keep original line breaks. No commentary."}]},
        {"role": "user", "content": [{"type": "input_image", "image_url": data_url, "detail": "high"}]},
    ]
    text = call_responses(messages, model=MODEL).strip()
    log(logging.INFO, "Image OCR via OpenAI Vision:done", filename=filename, text_len=len(text))
    text = text_maybe_translate(text, translate)
    if translate:
        log(logging.DEBUG, "Image text translated (if needed)", filename=filename, final_len=len(text))
    return [(1, text)]

def ocr_docx_pages(blob: bytes, lang: str) -> List[Tuple[int, str]]:
    translate = (lang or "en").lower() == "hi"
    log(logging.INFO, "DOCX parse:start")
    doc = Document(io.BytesIO(blob))
    parts: List[str] = []
    for p in doc.paragraphs:
        if p.text.strip():
            parts.append(p.text)
    for tbl in doc.tables:
        for row in tbl.rows:
            row_text = [cell.text.strip() for cell in row.cells]
            if any(row_text):
                parts.append(" | ".join(row_text))
    text = "\n".join(parts).strip()
    log(logging.INFO, "DOCX parse:done", chars=len(text))
    text = text_maybe_translate(text, translate)
    if translate:
        log(logging.DEBUG, "DOCX text translated (if needed)", final_len=len(text))
    return [(1, text)]

def ocr_pages(filename: str, blob: bytes, lang: str) -> List[Tuple[int, str]]:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf" or blob[:5] == b"%PDF-":
        log(logging.INFO, "OCR stage: PDF detected")
        return ocr_pdf_pages(blob, lang)
    if ext == ".docx" or mimetypes.guess_type(filename)[0] == \
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        log(logging.INFO, "OCR stage: DOCX detected")
        return ocr_docx_pages(blob, lang)
    mime, _ = mimetypes.guess_type(filename)
    if mime and mime.startswith("image/"):
        log(logging.INFO, "OCR stage: IMAGE detected", mime=mime)
        return ocr_image_pages(filename, blob, lang)
    log(logging.INFO, "OCR stage: defaulting to IMAGE path", mime=mime)
    return ocr_image_pages(filename, blob, lang)

# =========================
# STAGE 2: INTERPRETATION (text-only)
# =========================
DOC_INSTR_BASE = (
    "You are an information extraction engine. Use ONLY the provided page texts. "
    "For each requested field, return an object with keys: value (string), conf (0-1), "
    "and source (format 'FILENAME#page N'). Omit fields not confidently present. Do not fabricate."
)

def build_interpret_messages(filename: str, pages: List[Tuple[int, str]], fields: List[str]) -> List[Dict]:
    entries = [{"key": label_to_key(lbl), "label": lbl} for lbl in fields]
    labels_hint = "; ".join([e["label"] for e in entries])
    mapping_lines = "\n".join([f"- {e['label']} -> {e['key']}" for e in entries])

    blocks = []
    for pno, txt in pages:
        if txt and txt.strip():
            blocks.append(f"Page {pno}:\n{txt.strip()}")

    user_intro = (
        f"Document name: {filename}.\n"
        f"Extract only these fields (labels): {labels_hint}\n"
        "Use the following key names in JSON (label -> key):\n"
        f"{mapping_lines}\n"
        "Always include 'source' as 'FILENAME#page N' where N is the page number from which you derived the value."
    )
    user_text = user_intro + "\n\n==== Page Texts ====\n" + ("\n\n---\n\n".join(blocks) if blocks else "Page 1:\n")

    log(logging.INFO, "Interpretation:build_messages", pages=len(pages), fields=len(fields))
    return [
        {"role": "system", "content": [{"type": "input_text", "text": DOC_INSTR_BASE}]},
        {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
    ]

def structured_from_messages(messages: list, schema: dict, *, model: str) -> Dict:
    # Prefer JSON schema; fall back to strict JSON instruction
    t0 = time.time()
    try:
        log(logging.INFO, "Interpretation:OpenAI JSON-schema:start", model=model)
        resp = CLIENT.responses.create(
            model=model,
            input=messages,
            response_format={"type": "json_schema", "json_schema": schema},
        )
        out = getattr(resp, "output_parsed", None)
        if out is not None:
            log(logging.INFO, "Interpretation:OpenAI JSON-schema:done", model=model, ms=int((time.time()-t0)*1000), keys=list(out.keys()))
            return out
        raw = extract_text_from_response(resp)
        out2 = first_json_object(raw)
        log(logging.INFO, "Interpretation:OpenAI JSON-schema:parsed_text", model=model, ms=int((time.time()-t0)*1000), parsed=bool(out2))
        return out2
    except TypeError:
        log(logging.WARNING, "Interpretation:JSON-schema not supported by SDK, falling back", model=model)
        strong = list(messages)
        for blk in strong[-1]["content"]:
            if blk.get("type") == "input_text":
                blk["text"] += (
                    "\n\nReturn ONLY a minified JSON object (no prose, no code fences). "
                    "For each present field include: {\"value\": string, \"conf\": number 0-1, \"source\": string}."
                )
                break
        raw = call_responses(strong, model=model)
        out = first_json_object(raw)
        log(logging.INFO, "Interpretation:fallback parsed", model=model, parsed=bool(out))
        if out:
            return out
        # final fallback to vision model if text model fails
        log(logging.WARNING, "Interpretation:fallback to vision model", model=MODEL)
        raw2 = call_responses(strong, model=MODEL)
        return first_json_object(raw2) or {}

def interpret_text_structured(filename: str, pages: List[Tuple[int, str]], fields: List[str]) -> Dict:
    if not fields:
        log(logging.WARNING, "Interpretation: no fields provided")
        return {}
    schema = intake_schema_rich(fields)
    messages = build_interpret_messages(filename, pages, fields)
    out = structured_from_messages(messages, schema, model=TEXT_MODEL)

    # Normalize conf/source
    for k, obj in list(out.items()):
        if not isinstance(obj, dict):
            out[k] = {"value": str(obj), "conf": 0.5, "source": f"{filename}#page 1"}
            obj = out[k]
        obj["conf"] = clamp_conf(obj.get("conf", 0.5))
        if not isinstance(obj.get("source"), str) or "#page" not in obj.get("source", ""):
            guess = 1
            val = (obj.get("value") or "").strip()
            if val:
                for pno, txt in pages:
                    if val and (val in (txt or "")):
                        guess = pno
                        break
            obj["source"] = f"{filename}#page {guess}"
    log(logging.INFO, "Interpretation:done", fields_found=len(out))
    return out

# =========================
# Pipeline
# =========================
def detect_type(filename: str, blob: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf" or blob[:5] == b"%PDF-":
        return "pdf"
    if ext == ".docx" or mimetypes.guess_type(filename)[0] == \
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return "docx"
    mime, _ = mimetypes.guess_type(filename)
    if mime and mime.startswith("image/"):
        return "image"
    return "image"

def parse_file(filename: str, blob: bytes, lang: str, fields: List[str]) -> Dict:
    t0 = time.time()
    ftype = detect_type(filename, blob)
    log(logging.INFO, "Parse file:start", filename=filename, size=len(blob), type=ftype, lang=lang, fields=len(fields))
    # 1) OCR pages → texts
    pages = ocr_pages(filename, blob, lang)
    log(logging.INFO, "Parse file:OCR done", filename=filename, pages=len(pages), ms=int((time.time()-t0)*1000))
    # 2) Interpretation on text only
    t1 = time.time()
    result = interpret_text_structured(filename, pages, fields)
    log(logging.INFO, "Parse file:Interpretation done", filename=filename, ms=int((time.time()-t1)*1000))
    return result

# =========================
# API: /parse
# =========================
@app.post("/parse", response_class=PlainTextResponse)
async def parse(
    request: Request,
    files: List[UploadFile] = File(..., description="List of files (PDF/PNG/JPG/WEBP/DOCX)"),
    langs: List[str] = Form(..., description="List of language tags (en/hi) in the same order as files"),
    fields_to_extract: List[str] = Form(
        ...,
        description="Mandatory: field labels to extract (repeat this field OR provide a comma/newline/semicolon-separated list)",
    ),
    _auth_ok: bool = Depends(require_api_key),
) -> PlainTextResponse:
    # Correlate logs for this request
    REQ_ID.set(str(uuid.uuid4())[:8])
    log(logging.INFO, "Request:start", client=str(request.client))

    # Normalize files
    files = [f for f in files if getattr(f, "filename", None)]
    log(logging.INFO, "Request:files received", count=len(files), names=[f.filename for f in files])

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
        log(logging.WARNING, "Request:bad langs", langs=langs)
        raise HTTPException(status_code=400, detail=f"Invalid language(s): {bad}. Use 'en' or 'hi'.")
    log(logging.INFO, "Request:langs normalized", langs=langs)

    # Mandatory fields_to_extract
    if not fields_to_extract:
        log(logging.WARNING, "Request:missing fields_to_extract")
        raise HTTPException(status_code=400, detail="fields_to_extract is required and cannot be empty.")
    if len(fields_to_extract) == 1:
        labels = [s.strip() for s in re.split(r"[,\n;]+", fields_to_extract[0]) if s.strip()]
    else:
        labels = []
        for item in fields_to_extract:
            labels.extend([s.strip() for s in re.split(r"[,\n;]+", item) if s.strip()])
    fields = [lbl for lbl in labels if lbl]
    if not fields:
        log(logging.WARNING, "Request:empty fields_to_extract after parsing")
        raise HTTPException(status_code=400, detail="fields_to_extract must contain at least one non-empty label.")
    log(logging.INFO, "Request:fields_to_extract parsed", count=len(fields))

    # Process files
    result: Dict[str, str] = {}
    for idx, uf in enumerate(files):
        t_file = time.time()
        blob = await uf.read()
        log(logging.INFO, "File read", filename=uf.filename, bytes=len(blob))
        lang = langs[idx]
        try:
            parsed_obj = parse_file(uf.filename, blob, lang, fields)
        except Exception as e:
            log(logging.ERROR, "Parse error", filename=uf.filename, error=str(e))
            raise
        # Contract: filename -> JSON string (minified) of extracted fields
        js = json.dumps(parsed_obj, ensure_ascii=False)
        result[uf.filename] = js
        log(logging.INFO, "File done", filename=uf.filename, out_chars=len(js), ms=int((time.time()-t_file)*1000))

    log(logging.INFO, "Request:done", files=len(files))
    return PlainTextResponse(content=json.dumps(result, ensure_ascii=False), media_type="application/json")

# Optional health for liveness checks
@app.get("/health")
def health():
    ok = bool(SECRET_API_KEY)
    log(logging.DEBUG, "Health check", ok=ok)
    if not ok:
        return {"ok": False, "error": "SECRET_API_KEY not set", "ocr_model": MODEL, "text_model": TEXT_MODEL}
    return {"ok": True, "ocr_model": MODEL, "text_model": TEXT_MODEL}
