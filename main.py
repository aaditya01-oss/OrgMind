from groq import Groq
from config import GROQ_API_KEY, MODEL_NAME

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

if __name__ == "__main__":
    print("OrgBot is running! Type 'quit' to exit.")
    org = input("Enter organization name: ")
    user_id = input("Enter your user ID: ")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        response = chat(user_input, user_id, org)
        print(f"OrgBot: {response}")