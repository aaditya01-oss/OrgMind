from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from multi_agent import app as multi_agent_app
from security import scan_input
from pii_detector import scan_for_pii
from memory import save_conversation, get_conversation_history
from audit_logger import log_conversation, log_security_threat, log_pii_detection
from monitoring import track_conversation
from config import MODEL_NAME
import uvicorn
from fastapi import UploadFile, File, Form
import tempfile
import os
from document_processor import process_document, chunk_text, get_db_connection


app = FastAPI(title="OrgMind AI Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5147"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.post("/documents/train")
async def train_document(
    file: UploadFile = File(...),
    org_id: str = Form(...),
    doc_name: str = Form(...)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        process_document(tmp_path, org_id, doc_name)
        return {
            "message": f"Document '{doc_name}' successfully added to OrgMind knowledge base.",
            "chunks_processed": True,
            "org_id": org_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

@app.post("/documents/ocr")
async def ocr_document(
    file: UploadFile = File(...),
    org_id: str = Form(...),
    doc_name: str = Form(...)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        filename_lower = file.filename.lower()
        
        if filename_lower.endswith('.pdf'):
            from pypdf import PdfReader
            reader = PdfReader(tmp_path)
            extracted_text = ""
            for page in reader.pages:
                extracted_text += page.extract_text()
                
        elif filename_lower.endswith(('.png', '.jpg', '.jpeg')):
            # Use LLM vision for image OCR
            import base64
            with open(tmp_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            extracted_text = f"[Image file detected: {file.filename}]. For full OCR support on images, Tesseract OCR integration coming in next update. File received successfully."
        else:
            extracted_text = "Unsupported file format."
        
        return {
            "message": f"Document '{doc_name}' scanned successfully.",
            "extracted_text": extracted_text[:3000],
            "total_characters": len(extracted_text),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

@app.post("/agent/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    
    # SecureLayer check
    security_result = scan_input(request.user_message)
    if not security_result["is_safe"]:
        for threat in security_result["threats"]:
            log_security_threat(
                request.user_id,
                request.org_id,
                threat["type"],
                threat["reason"]
            )
        return ChatResponse(
            response="🚨 Your request was blocked by OrgMind SecureLayer.",
            user_id=request.user_id,
            org_id=request.org_id,
            security_blocked=True
        )
    
    # Load history
    history = get_conversation_history(request.user_id, request.org_id, limit=6)
    messages = []
    if history:
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        messages.append(("system", f"Previous conversation:\n{history_text}"))
    
    messages.append(("user", f"User {request.user_id} from org {request.org_id} says: {request.user_message}"))
    
    # Run through ORCHESTRATOR
    result = multi_agent_app.invoke({
        "messages": messages,
        "org_id": request.org_id,
        "user_id": request.user_id,
        "route": ""
    })
    
    response = result['messages'][-1].content
    
    # PII detection
    pii_result = scan_for_pii(request.user_message)
    safe_input = pii_result["masked_text"] if pii_result["contains_pii"] else request.user_message
    
    if pii_result["contains_pii"]:
        log_pii_detection(request.user_id, request.org_id, pii_result["pii_types_found"])
    
    save_conversation(request.user_id, request.org_id, "user", safe_input)
    save_conversation(request.user_id, request.org_id, "assistant", response)
    log_conversation(request.user_id, request.org_id, safe_input, len(response))
    track_conversation(request.user_id, request.org_id, safe_input, response, MODEL_NAME)
    
    return ChatResponse(
        response=response,
        user_id=request.user_id,
        org_id=request.org_id,
        pii_detected=pii_result["contains_pii"]
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)