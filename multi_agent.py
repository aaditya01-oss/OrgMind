from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from config import GROQ_API_KEY, MODEL_NAME
from pii_detector import pseudonymize

llm = ChatGroq(api_key=GROQ_API_KEY, model_name=MODEL_NAME)

# SUPPORT AGENT TOOLS
@tool
def create_ticket(issue: str, priority: str) -> str:
    """Creates a support ticket for user issues."""
    return f"Ticket TKT-001 created. Issue: {issue}. Priority: {priority}. IT notified."

@tool
def get_ticket_status(ticket_id: str) -> str:
    """Gets the status of a support ticket."""
    return f"Ticket {ticket_id}: Status=In Progress, ETA=2 hours"

# DOCMIND AGENT TOOLS
# Built per request: org_id is fixed by the server, so the model cannot choose it.
def make_docmind_tools(org_id: str, pii_mapping: dict):

    @tool
    def search_documents(query: str) -> str:
        """Searches this organization's documents for information. Use specific keywords from the user's question."""
        from search import semantic_search
        results = semantic_search(query, org_id, top_k=5)
        if not results:
            return "No relevant documents found."
        context = ""
        for content, doc_name, similarity in results:
            if similarity > 0.3:
                safe_chunk, _ = pseudonymize(content[:1500], pii_mapping)
                context += f"[Source: {doc_name}]\n{safe_chunk}\n\n"
        return context if context else "No highly relevant information found."

    @tool
    def list_documents() -> str:
        """Lists all documents available in this organization's knowledge base."""
        import psycopg2
        from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT,
            dbname=DB_NAME, user=DB_USER,
            password=DB_PASSWORD
        )
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT DISTINCT doc_name FROM document_chunks WHERE org_id = %s ORDER BY doc_name",
                (org_id,)
            )
            rows = cur.fetchall()
            cur.close()
        finally:
            conn.close()

        if not rows:
            return "No documents found for this organization."
        return "Documents available: " + ", ".join(row[0] for row in rows)

    return [search_documents, list_documents]



# CREATE SPECIALIZED AGENTS
support_agent = create_react_agent(llm, [create_ticket, get_ticket_status])

# STATE
class OrgMindState(TypedDict):
    messages: Annotated[list, operator.add]
    org_id: str
    user_id: str
    route: str
    pii_mapping: dict

# ORCHESTRATOR
def orchestrator(state: OrgMindState):
    last_message = state["messages"][-1]
    
    routing_prompt = f"""
    Analyze this user message and decide which agent should handle it.
    Return ONLY one word: 'support' or 'docmind'
    
    - support: for issues, problems, tickets, IT help, broken things
    - docmind: for questions about documents, policies, information lookup
    
    User message: {last_message}
    """
    
    response = llm.invoke(routing_prompt)
    route = response.content.strip().lower()
    
    if "support" in route:
        return {"route": "support"}
    else:
        return {"route": "docmind"}

def run_support_agent(state: OrgMindState):
    result = support_agent.invoke({"messages": state["messages"]})
    return {"messages": result["messages"]}

def run_docmind_agent(state: OrgMindState):
    """Run the document mind agent."""
    docmind_agent = create_react_agent(llm, make_docmind_tools(state["org_id"], state["pii_mapping"]))
    result = docmind_agent.invoke({"messages": state["messages"]})
    return {"messages": result["messages"]}

def route_decision(state: OrgMindState):
    return state["route"]

# BUILD GRAPH
workflow = StateGraph(OrgMindState)

workflow.add_node("orchestrator", orchestrator)
workflow.add_node("support", run_support_agent)
workflow.add_node("docmind", run_docmind_agent)

workflow.set_entry_point("orchestrator")

workflow.add_conditional_edges(
    "orchestrator",
    route_decision,
    {
        "support": "support",
        "docmind": "docmind"
    }
)

workflow.add_edge("support", END)
workflow.add_edge("docmind", END)

app = workflow.compile()

if __name__ == "__main__":
    print("OrgMind Multi-Agent System running...")
    
    org_id = input("Org ID: ")
    user_id = input("User ID: ")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        
        result = app.invoke({
            "messages": [("user", user_input)],
            "org_id": org_id,
            "user_id": user_id,
            "route": "",
            "pii_mapping": {}
        })
        
        print(f"Agent: {result['messages'][-1].content}")