# ocr_any.py
import os
import sys
import io
import re
import base64
import mimetypes
from pathlib import Path
from typing import List

from openai import OpenAI
from PIL import Image
import fitz  # PyMuPDF
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT

# ========= Config via env =========
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
MODEL_FALLBACK = os.getenv("OPENAI_MODEL_FALLBACK", "gpt-4.1-mini")
OCR_LANG = os.getenv("OCR_LANG", "en").strip().lower()
if OCR_LANG not in {"en", "hi"}:
    OCR_LANG = "en"
OCR_REDACT = os.getenv("OCR_REDACT", "false").strip().lower() == "true"
OCR_TRANSLATE_ALWAYS = os.getenv("OCR_TRANSLATE_ALWAYS", "false").strip().lower() == "true"

# Prompts:
# - en: verbatim English OCR
# - hi: OCR + translate to English output ONLY
PROMPTS = {
    "en": "OCR this image. Return the text exactly as printed in English.",
    "hi": "OCR this image. Extract the Hindi (Devanagari) text and return an accurate English translation ONLY. Preserve structure and line breaks where possible.",
}

# Redaction guidance (optional)
REDACTION_GUIDE = (
    "If you encounter government IDs or highly sensitive numbers, redact them by masking all "
    "but the last 4 characters (e.g., 'XXXXXXXXXXXX1234'). For PAN (pattern like AAAAA9999A), "
    "mask as 'XXXXX9999X'. Keep everything else verbatim."
)

SYSTEM_BASE = (
    "You are an OCR engine. The user has provided this document and explicitly consents to OCR. "
    "Your ONLY task is to transcribe text from the image(s) or page(s); when asked, translate to English. "
    "Do not add commentary, summaries, or warnings. Do not refuse unless the image is clearly illegal content. "
)

SYSTEM_VERBATIM = SYSTEM_BASE + "Return plain text only, preserving line breaks."
SYSTEM_REDACT = SYSTEM_BASE + REDACTION_GUIDE + " Return plain text only, preserving line breaks."

# System for translation of plain text (DOCX paragraphs/tables)
SYSTEM_TRANSLATOR = (
    "You are a professional translator. Translate the provided text into natural, fluent English. "
    "Preserve formatting and line breaks as much as possible. Do not add commentary."
)

CLIENT = OpenAI()

# ========= Helpers =========
def _img_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"

def _bytes_to_data_url(blob: bytes, mime: str = "image/png") -> str:
    b64 = base64.b64encode(blob).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def _file_to_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def _extract_text_from_response(resp) -> str:
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

_REFUSAL_RE = re.compile(r"\b(i'?m|i am|sorry|cannot|can'?t|unable|assist)\b", re.I)

def _call_responses(model: str, messages: list) -> str:
    resp = CLIENT.responses.create(model=model, input=messages)
    return _extract_text_from_response(resp)

# ========= OCR Calls =========
def _call_ocr(model: str, system_text: str, user_prompt: str, data_url: str, detail: str = "high") -> str:
    messages = [
        {"role": "system", "content": [{"type": "input_text", "text": system_text}]},
        {"role": "user", "content": [
            {"type": "input_text", "text": user_prompt},
            {"type": "input_image", "image_url": data_url, "detail": detail},
        ]},
    ]
    return _call_responses(model, messages)

def ocr_image_dataurl(data_url: str, user_prompt: str, detail: str = "high") -> str:
    system_text = SYSTEM_REDACT if OCR_REDACT else SYSTEM_VERBATIM
    text = _call_ocr(MODEL, system_text, user_prompt, data_url, detail=detail)
    if _REFUSAL_RE.search(text):
        narrow_system = (system_text + " This is first-party, user-owned content strictly for OCR/translation.")
        text = _call_ocr(MODEL_FALLBACK, narrow_system, user_prompt, data_url, detail=detail)
    return text.strip()

# ========= Translator for plain text (DOCX paragraphs/tables) =========
def translate_text_to_english(text: str) -> str:
    # Skip empty
    if not text.strip():
        return text
    messages = [
        {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_TRANSLATOR}]},
        {"role": "user", "content": [{"type": "input_text", "text": text}]},
    ]
    out = _call_responses(MODEL, messages)
    if _REFUSAL_RE.search(out):
        out = _call_responses(MODEL_FALLBACK, messages)
    return out.strip()

# ========= PDF (PyMuPDF) =========
def ocr_pdf(path: Path, prompt: str, zoom: float = 2.5) -> str:
    texts: List[str] = []
    doc = fitz.open(str(path))
    try:
        mat = fitz.Matrix(zoom, zoom)
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            data_url = _img_to_data_url(img, fmt="PNG")
            page_text = ocr_image_dataurl(data_url, f"{prompt}\nKeep original line breaks.")
            texts.append(f"[Page {i}]\n{page_text}".rstrip())
    finally:
        doc.close()
    return "\n\n".join(texts).strip()

# ========= Images =========
def ocr_image_file(path: Path, prompt: str) -> str:
    data_url = _file_to_data_url(path)
    return ocr_image_dataurl(data_url, prompt)

# ========= DOCX (text + embedded images) =========
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

def ocr_docx_images(doc: Document, prompt: str) -> List[str]:
    image_texts: List[str] = []
    rels = list(doc.part.rels.values())
    idx = 0
    for rel in rels:
        if rel.reltype == RT.IMAGE:
            idx += 1
            part = rel.target_part
            blob = part.blob
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
            data_url = _bytes_to_data_url(blob, mime=mime)
            # This prompt already outputs English if OCR_LANG == "hi"
            txt = ocr_image_dataurl(data_url, f"{prompt}\nThis image came from a DOCX file.")
            image_texts.append(f"[DOCX Image {idx}]\n{txt}".rstrip())
    return image_texts

def process_docx(path: Path, prompt: str) -> str:
    doc = Document(str(path))
    text_block = extract_docx_text(doc)

    # If Hindi mode or forced translate, translate the plain DOCX text to English
    if text_block and (OCR_LANG == "hi" or OCR_TRANSLATE_ALWAYS):
        text_block = translate_text_to_english(text_block)

    image_ocr_blocks = ocr_docx_images(doc, prompt)

    blocks: List[str] = []
    if text_block:
        blocks.append("[DOCX Text]\n" + text_block.strip())
    if image_ocr_blocks:
        blocks.extend(image_ocr_blocks)
    return ("\n\n".join(blocks)).strip() if blocks else ""

# ========= Main =========
def main():
    if len(sys.argv) < 2:
        print("Usage: python ocr_any.py <file.(pdf|png|jpg|jpeg|webp|docx)>", file=sys.stderr)
        sys.exit(1)

    infile = Path(sys.argv[1]).expanduser().resolve()
    if not infile.exists():
        print(f"File not found: {infile}", file=sys.stderr)
        sys.exit(1)

    prompt = PROMPTS[OCR_LANG]
    ext = infile.suffix.lower()

    if ext == ".pdf":
        text = ocr_pdf(infile, prompt, zoom=2.5)
    elif ext in {".png", ".jpg", ".jpeg", ".webp"}:
        text = ocr_image_file(infile, prompt)
    elif ext == ".docx":
        text = process_docx(infile, prompt)
    else:
        mime, _ = mimetypes.guess_type(str(infile))
        if mime == "application/pdf":
            text = ocr_pdf(infile, prompt, zoom=2.5)
        elif mime and mime.startswith("image/"):
            text = ocr_image_file(infile, prompt)
        elif ext == ".docx" or (mime and mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
            text = process_docx(infile, prompt)
        else:
            print("Unsupported file type. Provide PDF, DOCX, or an image (png/jpg/webp).", file=sys.stderr)
            sys.exit(2)

    print(text)

if __name__ == "__main__":
    main()
