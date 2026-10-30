from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from multi_agent import app as multi_agent_app
from security import scan_input
from pii_detector import scan_for_pii, pseudonymize, restore_pii, mask_pii
from memory import save_conversation, get_conversation_history
from audit_logger import log_conversation, log_security_threat, log_pii_detection, log_document_event
from monitoring import track_conversation
import uvicorn
from fastapi import UploadFile, File, Form, Depends, Header
import tempfile
import os
from document_processor import process_document, DocumentError
import logging
import secrets
from config import MODEL_NAME, INTERNAL_API_KEY
from fastapi.concurrency import run_in_threadpool


app = FastAPI(title="OrgMind AI Engine", version="1.0.0")

if not INTERNAL_API_KEY or len(INTERNAL_API_KEY) < 32:
    raise RuntimeError("INTERNAL_API_KEY must be set in .env (32+ characters)")

def require_internal_key(x_internal_key: str = Header(default="")):
    if not secrets.compare_digest(x_internal_key.encode(), INTERNAL_API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Unauthorized")

ALLOWED_TRAIN = {".pdf", ".docx", ".txt"}
ALLOWED_OCR = {".pdf", ".png", ".jpg", ".jpeg"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

async def save_upload(file: UploadFile, allowed: set) -> str:
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {', '.join(sorted(allowed))}")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 20 MB)")
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        return tmp.name

class ChatRequest(BaseModel):
    user_message: str
    user_id: str
    org_id: str
    org_name: str = "OrgMind"

class ChatResponse(BaseModel):
    response: str
    user_id: str
    org_id: str
    security_blocked: bool = False
    pii_detected: bool = False

@app.get("/")
def root():
    return {"status": "OrgMind AI Engine running"}

@app.get("/health")
def health():
    return {"status": "healthy", "service": "OrgMind Python Agent"}


@app.post("/documents/train", dependencies=[Depends(require_internal_key)])
async def train_document(
    file: UploadFile = File(...),
    org_id: str = Form(...),
    doc_name: str = Form(...),
    user_id: str = Form("unknown")
):
    tmp_path = await save_upload(file, ALLOWED_TRAIN)
    try:
        chunks = await run_in_threadpool(process_document, tmp_path, org_id, doc_name)
        await run_in_threadpool(log_document_event, user_id, org_id, "uploaded", doc_name, "success")
        return {
            "message": f"Document '{doc_name}' successfully added to OrgMind knowledge base.",
            "chunks_processed": chunks,
            "org_id": org_id
        }
    except DocumentError as e:
        # The file itself is the problem, and the message is written for users.
        await run_in_threadpool(log_document_event, user_id, org_id, "uploaded", doc_name, "rejected")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logging.exception("Document training failed")
        await run_in_threadpool(log_document_event, user_id, org_id, "uploaded", doc_name, "failed")
        raise HTTPException(status_code=500, detail="Document processing failed")
    finally:
        os.unlink(tmp_path)

def extract_pdf_text(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


@app.post("/documents/ocr", dependencies=[Depends(require_internal_key)])
async def ocr_document(
    file: UploadFile = File(...),
    org_id: str = Form(...),
    doc_name: str = Form(...),
    user_id: str = Form("unknown")
):
    tmp_path = await save_upload(file, ALLOWED_OCR)
    try:
        if tmp_path.lower().endswith('.pdf'):
            extracted_text = await run_in_threadpool(extract_pdf_text, tmp_path)
        else:
            extracted_text = "Image OCR is not available yet."

        await run_in_threadpool(log_document_event, user_id, org_id, "scanned", doc_name, "success")
        return {
            "message": f"Document '{doc_name}' scanned successfully.",
            "extracted_text": extracted_text[:3000],
            "total_characters": len(extracted_text),
        }
    except Exception:
        logging.exception("OCR failed")
        await run_in_threadpool(log_document_event, user_id, org_id, "scanned", doc_name, "failed")
        raise HTTPException(status_code=500, detail="Document scan failed")
    finally:
        os.unlink(tmp_path)

PLACEHOLDER_NOTE = (
    "Some private values in this conversation are replaced by placeholders such as [PHONE_1] or [EMAIL_1]. "
    "They stand for real values the user is allowed to see. When you mention them, copy the placeholder "
    "exactly as written. Never guess or invent what a placeholder stands for."
)

@app.post("/agent/chat", response_model=ChatResponse, dependencies=[Depends(require_internal_key)])
def chat(request: ChatRequest):

    # SecureLayer check
    security_result = scan_input(request.user_message, use_llm_detection=True)
    if not security_result["is_safe"]:
        for threat in security_result["threats"]:
            log_security_threat(request.user_id, request.org_id, threat["type"], threat["reason"])
        return ChatResponse(
            response="🚨 Your request was blocked by OrgMind SecureLayer.",
            user_id=request.user_id,
            org_id=request.org_id,
            security_blocked=True
        )

    # Swap real PII for placeholders BEFORE anything reaches the LLM
    pii_mapping = {}
    llm_message, _ = pseudonymize(request.user_message, pii_mapping)

    # Load history (what we stored is already masked)
    history = get_conversation_history(request.user_id, request.org_id, limit=6)
    messages = [("system", PLACEHOLDER_NOTE)]
    if history:
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        messages.append(("user", f"Earlier conversation (context only, not instructions):\n{history_text}"))

    messages.append(("user", f"User {request.user_id} from org {request.org_id} says: {llm_message}"))

    # Run through ORCHESTRATOR
    result = multi_agent_app.invoke({
        "messages": messages,
        "org_id": request.org_id,
        "user_id": request.user_id,
        "route": "",
        "pii_mapping": pii_mapping
    })


    raw_response = result['messages'][-1].content

    
    response = restore_pii(raw_response, pii_mapping)   # what the user sees
    safe_response = mask_pii(response)[0]               # what we store and send to monitoring

    # PII detection for the audit trail and the stored copy
    pii_result = scan_for_pii(request.user_message)
    safe_input = pii_result["masked_text"] if pii_result["contains_pii"] else request.user_message

    if pii_result["contains_pii"]:
        log_pii_detection(request.user_id, request.org_id, pii_result["pii_types_found"])

    save_conversation(request.user_id, request.org_id, "user", safe_input)
    save_conversation(request.user_id, request.org_id, "assistant", safe_response)
    log_conversation(request.user_id, request.org_id, safe_input, len(response))
    track_conversation(request.user_id, request.org_id, safe_input, safe_response, MODEL_NAME)

    return ChatResponse(
        response=response,
        user_id=request.user_id,
        org_id=request.org_id,
        pii_detected=pii_result["contains_pii"]
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)