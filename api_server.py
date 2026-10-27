from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import agent, tools
from security import scan_input
from pii_detector import scan_for_pii
from memory import save_conversation, get_conversation_history
from audit_logger import log_conversation, log_security_threat, log_pii_detection
from monitoring import track_conversation
from config import MODEL_NAME
import uvicorn

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
            response="🚨 Your request was blocked by OrgMind SecureLayer. Suspicious input detected.",
            user_id=request.user_id,
            org_id=request.org_id,
            security_blocked=True
        )
    
    # Load conversation history
    history = get_conversation_history(request.user_id, request.org_id, limit=6)
    messages = []
    if history:
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        messages.append(("system", f"Previous conversation:\n{history_text}"))
    
    messages.append(("user", f"User {request.user_id} from org {request.org_id} says: {request.user_message}"))
    
    # Run agent
    result = agent.invoke({"messages": messages})
    response = result['messages'][-1].content
    
    # PII detection
    pii_result = scan_for_pii(request.user_message)
    safe_input = pii_result["masked_text"] if pii_result["contains_pii"] else request.user_message
    
    if pii_result["contains_pii"]:
        log_pii_detection(request.user_id, request.org_id, pii_result["pii_types_found"])
    
    # Save to memory and audit
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