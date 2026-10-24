from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from memory import save_conversation, get_conversation_history, get_user_profile
from security import scan_input
from config import GROQ_API_KEY, MODEL_NAME
from pii_detector import scan_for_pii
from audit_logger import log_conversation, log_security_threat, log_pii_detection
from monitoring import track_conversation, track_security_event

llm = ChatGroq(api_key=GROQ_API_KEY, model_name=MODEL_NAME)

@tool
def create_support_ticket(issue: str, priority: str, user_id: str) -> str:
    """Creates a support ticket for a user issue. 
    Use when user reports a problem that needs IT or HR attention.
    Priority should be 'low', 'medium', or 'high'."""
    ticket_id = f"TKT-{user_id[:3].upper()}-001"
    return f"Support ticket {ticket_id} created. Issue: {issue}. Priority: {priority}. Department notified."

@tool
def search_knowledge_base(query: str, org_id: str) -> str:
    """Search the organization's knowledge base for information.
    Use when user asks about policies, procedures, or company information."""
    from search import semantic_search
    results = semantic_search(query, org_id, top_k=2)
    if not results:
        return "No relevant information found."
    context = ""
    for content, doc_name, similarity in results:
        if similarity > 0.3:
            context += f"From {doc_name}: {content[:300]}\n\n"
    return context if context else "No highly relevant information found."

@tool
def get_department_contact(department: str) -> str:
    """Get contact information for a specific department.
    Use when user needs to reach HR, IT, Finance, or Operations."""
    contacts = {
        "hr": "HR Department: hr@orgmind.com | Extension: 101",
        "it": "IT Support: it@orgmind.com | Extension: 102",
        "finance": "Finance: finance@orgmind.com | Extension: 103",
        "operations": "Operations: ops@orgmind.com | Extension: 104"
    }
    return contacts.get(department.lower(), f"Department '{department}' not found.")

@tool
def get_ticket_status(ticket_id: str) -> str:
    """Check the status of an existing support ticket."""
    mock_tickets = {
        "TKT-AAA-001": {"status": "In Progress", "assigned_to": "IT Team", "eta": "2 hours"},
        "TKT-91-001": {"status": "Open", "assigned_to": "IT Team", "eta": "4 hours"},
    }
    ticket = mock_tickets.get(ticket_id.upper())
    if not ticket:
        return f"Ticket {ticket_id} not found."
    return f"Ticket {ticket_id}: Status={ticket['status']}, Assigned={ticket['assigned_to']}, ETA={ticket['eta']}"

@tool
def translate_to_nepali(text: str) -> str:
    """Translate English text to Nepali language."""
    translations = {
        "hello": "नमस्ते",
        "how are you": "तपाईं कस्तो हुनुहुन्छ",
        "support ticket": "समर्थन टिकट",
        "thank you": "धन्यवाद",
    }
    text_lower = text.lower()
    if text_lower in translations:
        return f"Nepali: {translations[text_lower]}"
    return f"[NepaliMind] Translation for: '{text}' coming in Phase 5."

tools = [
    create_support_ticket,
    search_knowledge_base,
    get_department_contact,
    get_ticket_status,
    translate_to_nepali
]

if __name__ == "__main__":
    print("OrgMind Agent running...")
    agent = create_react_agent(llm, tools)
    
    org_id = input("Org ID: ")
    user_id = input("User ID: ")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        
        # SECURELAYER CHECK
        security_result = scan_input(user_input)
        if not security_result["is_safe"]:
            print("🚨 SecureLayer: Suspicious input blocked.")
            for threat in security_result["threats"]:
                print(f"   Reason: {threat['reason']}")
                log_security_threat(user_id, org_id, threat["type"], threat["reason"])
                track_security_event(user_id, org_id, threat["type"], threat["reason"])
            continue
        
        # Load history
        history = get_conversation_history(user_id, org_id, limit=6)
        messages = []
        if history:
            history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
            messages.append(("system", f"Previous conversation:\n{history_text}"))
        
        messages.append(("user", f"User {user_id} from org {org_id} says: {user_input}"))
        
        result = agent.invoke({"messages": messages})
        response = result['messages'][-1].content

        # Track in Langfuse
        track_conversation(user_id, org_id, user_input, response, MODEL_NAME)

        pii_result = scan_for_pii(user_input)
        safe_input = pii_result["masked_text"] if pii_result["contains_pii"] else user_input

        if pii_result["contains_pii"]:
            print(f"⚠️  PII detected and masked: {pii_result['pii_types_found']}")
            log_pii_detection(user_id, org_id, pii_result["pii_types_found"])

        save_conversation(user_id, org_id, "user", safe_input)
        save_conversation(user_id, org_id, "assistant", response)
        log_conversation(user_id, org_id, safe_input, len(response))

        print(f"Agent: {response}")