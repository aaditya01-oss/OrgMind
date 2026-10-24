import os
from langfuse import observe, get_client
from config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL

os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY
os.environ["LANGFUSE_HOST"] = LANGFUSE_BASE_URL

@observe()
def track_conversation(user_id: str, org_id: str, user_input: str, response: str, model: str):
    return response

def track_security_event(user_id: str, org_id: str, event_type: str, details: str):
    print(f"[Security] {event_type}: {details}")

if __name__ == "__main__":
    print("Testing Langfuse connection...")
    track_conversation(
        user_id="test_user",
        org_id="01",
        user_input="Hello OrgBot",
        response="Hello! How can I help?",
        model="openai/gpt-oss-20b"
    )
    client = get_client()
    client.flush()
    print("✅ Langfuse connected!")