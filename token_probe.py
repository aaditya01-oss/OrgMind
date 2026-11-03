import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.prebuilt import create_react_agent
from multi_agent import llm, make_docmind_tools, DOCMIND_RULES
import re

class Probe(BaseCallbackHandler):
    def __init__(self):
        self.n = 0
        self.total = 0

    def on_llm_end(self, response, **kwargs):
        self.n += 1
        try:
            u = response.generations[0][0].message.usage_metadata
            inp, out = u["input_tokens"], u["output_tokens"]
        except Exception:
            inp = out = 0
        self.total += inp + out
        print(f"  model call {self.n}: input {inp} tokens, output {out} tokens")

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"  tool: {serialized.get('name')}  {str(input_str)[:80]}")

    def on_tool_end(self, output, **kwargs):
        text = getattr(output, "content", str(output))
        docs = re.findall(r"\[Source: ([^\]]+)\]", text)
        print("  documents returned:", docs)
        print("  contains the helpdesk days sentence:", "आइतबारदेखि" in text)


question = sys.argv[1] if len(sys.argv) > 1 else "What is the project codename for the new office?"
print("QUESTION:", question)
probe = Probe()
tools = make_docmind_tools("01", {}, [], role="member", department=None, question=question)
agent = create_react_agent(llm, tools)
try:
    result = agent.invoke(
        {"messages": [("system", DOCMIND_RULES), ("user", question)]},
        {"recursion_limit": 5, "callbacks": [probe]},
    )
    print("ANSWER:", result["messages"][-1].content[:150])
except Exception as e:
    print("stopped:", type(e).__name__, str(e)[:200])
print("TOTAL tokens for the answering step:", probe.total)