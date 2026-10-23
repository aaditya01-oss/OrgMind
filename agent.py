from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from config import GROQ_API_KEY, MODEL_NAME
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name=MODEL_NAME
)

@tool
def create_support_ticket(issue: str, priority: str, user_id: str) -> str:
    """Creates a support ticket for a user issue. 
    Use when user reports a problem that needs IT or HR attention.
    Priority should be 'low', 'medium', or 'high'."""
    
    ticket_id = f"TKT-{user_id[:3].upper()}-001"
    return f"Support ticket {ticket_id} created successfully. Issue: {issue}. Priority: {priority}. The relevant department has been notified."

@tool
def search_knowledge_base(query: str, org_id: str) -> str:
    """Search the organization's knowledge base for information.
    Use when user asks about policies, procedures, or company information."""
    
    from search import semantic_search
    results = semantic_search(query, org_id, top_k=2)
    
    if not results:
        return "No relevant information found in the knowledge base."
    
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
    """Check the status of an existing support ticket.
    Use when user asks about their ticket or wants an update."""
    
    mock_tickets = {
        "TKT-AAA-001": {"status": "In Progress", "assigned_to": "IT Team", "eta": "2 hours"},
        "TKT-91-001": {"status": "Open", "assigned_to": "IT Team", "eta": "4 hours"},
    }
    
    ticket = mock_tickets.get(ticket_id.upper())
    if not ticket:
        return f"Ticket {ticket_id} not found. Please check the ticket ID."
    
    return f"Ticket {ticket_id}: Status={ticket['status']}, Assigned to={ticket['assigned_to']}, ETA={ticket['eta']}"

@tool
def translate_to_nepali(text: str) -> str:
    """Translate English text to Nepali language.
    Use when user requests Nepali translation or communicates in Nepali."""
    
    translations = {
        "hello": "नमस्ते",
        "how are you": "तपाईं कस्तो हुनुहुन्छ",
        "support ticket": "समर्थन टिकट",
        "thank you": "धन्यवाद",
    }
    
    text_lower = text.lower()
    if text_lower in translations:
        return f"Nepali translation: {translations[text_lower]}"
    
    return f"[NepaliMind] Translation requested for: '{text}'. Full translation service coming in Phase 5."

tools = [create_support_ticket, search_knowledge_base, get_department_contact, get_ticket_status, translate_to_nepali]

from langgraph.prebuilt import create_react_agent

if __name__ == "__main__":
    print("OrgMind Agent running...")
    
    agent = create_react_agent(llm, tools)
    
    org_id = input("Org ID: ")
    user_id = input("User ID: ")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
            
        result = agent.invoke({
            "messages": [("user", f"User {user_id} from org {org_id} says: {user_input}")]
        })
        
        print(f"Agent: {result['messages'][-1].content}")