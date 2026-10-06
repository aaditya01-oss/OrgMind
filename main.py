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

def chat(user_message, org_name="OrgMind"):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(
                organization_name=org_name
            )},
            {"role": "user", "content": user_message}
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print("OrgBot is running...")
    user_input = input("You: ")
    response = chat(user_input)
    print(f"OrgBot: {response}")