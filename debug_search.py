import sys
sys.stdout.reconfigure(encoding="utf-8")
from search import semantic_search, MIN_SIMILARITY

QUESTIONS = [
    "हेल्पडेस्क कति बजे खुल्छ?",
    "हेल्पडेस्क कति बजे खुल्छ? मेरो प्यान नम्बर [PAN_1] हो",
    "helpdesk hours",
]

for q in QUESTIONS:
    print("\n=====", q)
    results = semantic_search(q, "01", top_k=8)
    for i, (content, doc, sim) in enumerate(results, 1):
        mark = "" if sim > MIN_SIMILARITY else "  (below threshold)"
        print(f"{i}. {sim:.3f}  {doc}{mark}\n     {content[:90]!r}")