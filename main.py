from groq import Groq
from config import GROQ_API_KEY, MODEL_NAME
import time
import json

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """
You are OrgBot, an intelligent organizational assistant 
for {organization_name}. You help employees find information, 
answer questions and assist with organizational tasks.
You only answer based on the organization's knowledge.
If you don't know something, say so honestly.
Never make up information.
"""

user_sessions = {}

def chat(user_message,user_id, org_name="OrgMind"):
    if user_id not in user_sessions:
        user_sessions[user_id] = []
    user_sessions[user_id].append({
        "role": "user",
        "content": user_message
    })
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(
            organization_name=org_name
        )}
    ] + user_sessions[user_id]
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages
    )
    
    assistant_message = response.choices[0].message.content
    
    user_sessions[user_id].append({
        "role": "assistant",
        "content": assistant_message
    })
    
    return assistant_message

def chat_stream(user_message, user_id, org_name="OrgMind"):
    if user_id not in user_sessions:
        user_sessions[user_id] = []
    
    user_sessions[user_id].append({
        "role": "user",
        "content": user_message
    })
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(
            organization_name=org_name
        )}
    ] + user_sessions[user_id]
    
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        stream=True
    )
    
    full_response = ""
    
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token is not None:
            print(token, end="", flush=True)
            full_response += token
    
    print()
    
    user_sessions[user_id].append({
        "role": "assistant",
        "content": full_response
    })
    time.sleep(0.15)
    
    return full_response

def get_structured_response(user_message, user_id, org_name="OrgMind"):
    if user_id not in user_sessions:
        user_sessions[user_id] = []

    structured_prompt = f"""
    Analyze the user message and return ONLY a JSON object with these fields:
    - intent: what the user wants to do
    - route_to: one of [OrgBot, DocMind, SupportDesk, InterLink, ComplianceBot]
    - confidence: float between 0 and 1
    - priority: one of [low, medium, high]
    - user_message: the original message
    
    Return ONLY valid JSON. No explanation. No markdown. No backticks.
    
    User message: {user_message}
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(
                organization_name=org_name
            )},
            {"role": "user", "content": structured_prompt}
        ]
    )

    raw = response.choices[0].message.content

    try:
        parsed = json.loads(raw)
        return parsed
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON", "raw": raw}

if __name__ == "__main__":
    print("OrgBot is running! Type 'quit' to exit.")
    org = input("Enter organization name: ")
    user_id = input("Enter your user ID: ")


    print("\n--- ROUTER TEST ---")
    user_input = input("You: ")
    structured = get_structured_response(user_input, user_id, org)
    print(json.dumps(structured, indent=2))
    print("-------------------\n")
    
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        print("OrgBot: ", end="", flush=True)
        response = chat_stream(user_input, user_id, org)



