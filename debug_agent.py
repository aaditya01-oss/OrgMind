from multi_agent import app

for question in ["Summarize the SRS", "What documents do we have?"]:
    r = app.invoke({"messages": [("user", question)], "org_id": "01",
                    "user_id": "test", "route": "", "pii_mapping": {}})
    print("\n=====", question, "| route:", r["route"])
    for m in r["messages"]:
        if isinstance(m, tuple):
            print("input tuple |", m[0], "|", str(m[1])[:150])
            continue
        calls = getattr(m, "tool_calls", None)
        print(type(m).__name__, "| tool calls:", [c["name"] for c in calls] if calls else "-",
              "|", str(m.content)[:150].replace("\n", " "))