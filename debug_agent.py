from multi_agent import app

QUESTIONS = [
    "How did the test users rate the system?",
    "What are the sick leave days, and how often must passwords be changed?",
]

for question in QUESTIONS:
    sources = []
    print("\n=====", question)
    try:
        r = app.invoke({"messages": [("user", question)], "org_id": "01",
                        "user_id": "test", "route": "", "pii_mapping": {}, "sources": sources})
    except Exception:
        import traceback
        traceback.print_exc()
        continue
    for m in r["messages"]:
        if isinstance(m, tuple):
            continue
        calls = getattr(m, "tool_calls", None)
        if calls:
            for c in calls:
                print("TOOL CALL:", c["name"], c["args"])
        elif type(m).__name__ == "AIMessage":
            print("ANSWER:", str(m.content)[:200].replace("\n", " "))
    print("sources:", sources)