"""
DocMind retrieval evaluation: does the RIGHT chunk come back for a question?

Run from the OrgMind folder:   python docmind_eval.py
Options:                       python docmind_eval.py --org 01 --k 5 --verbose

It only READS the database (no writes) and makes NO LLM calls, so it is free, fast
and safe to run after every change to chunking, embeddings or search.

HOW TO ADD A CASE
  (question, [snippets], document)
  - question: write it in YOUR OWN WORDS, not copied from the document. Paraphrased
    questions are the honest test; copied ones only test keyword matching.
  - snippets: any ONE of these appearing (case-insensitive) in a returned chunk counts as a hit.
    Check the real answer in the document first. Never trust a number you did not verify.
  - document: the stored document name to restrict the search to, or None for all documents.
  Questions whose answer is NOT in any document go in NEGATIVE: the right behaviour there is
  that nothing comes back above the similarity threshold.

KEEP A HOLDOUT: once you have tuned chunk size or settings on the main list, add new questions
under HOLDOUT and watch whether the score holds. Tuning on the same questions you score is cheating.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from search import semantic_search, MIN_SIMILARITY

THRESHOLD = float(sys.argv[sys.argv.index("--threshold") + 1]) if "--threshold" in sys.argv else MIN_SIMILARITY
# ---------------------------------------------------------------------------
# Questions whose answer IS in the documents
# ---------------------------------------------------------------------------
CASES = [
    # EnterpriseAI.docx: answers verified against Table 6.1 in the document
    ("How accurate is the retrieval of the knowledge agent?", ["91.7"], None),
    ("What accuracy did intent classification reach?", ["94.3"], None),
    ("What was the character error rate when reading scanned documents?", ["3.2%"], None),
    ("How accurately are support tickets sent to the right department?", ["92.0"], None),
    ("How long does a response take on average?", ["2.8 sec", "2.8 s"], None),
    ("How did the test users rate the system?", ["4.3/5", "4.3 / 5"], None),
    ("How well does LayoutLMv3 extract fields from forms?", ["88.5"], None),
    ("What accuracy does sentiment classification have?", ["89.6"], None),
    ("Which large language model does the assistant use?", ["gemini"], None),
    # SRS Document.docx: CHECK each snippet against the real document before trusting the score
    ("How long is a reserved slot held before it is released?", ["15 min", "15 minutes"], None),
    ("Which payment gateways does the parking system support?", ["esewa", "khalti"], None),
    ("What API uptime is required?", ["99.5"], None),
    ("Which protocol do the sensors use to send data?", ["mqtt"], None),
    ("What user roles does the parking system have?", ["operator", "reporter"], None),
]

# Add new questions here AFTER you have finished tuning on CASES
HOLDOUT = [
]

# ---------------------------------------------------------------------------
# Questions with NO answer in the documents: nothing should come back above THRESHOLD
# ---------------------------------------------------------------------------
NEGATIVE = [
    "What is the weather today?",
    "Who won the football world cup in 2018?",
    "What is the annual leave policy of the company?",
]


def norm(text):
    text = text.replace(" ", " ").replace(" ", " ")
    return " ".join(text.lower().split())


def first_hit_rank(results, snippets):
    """1-based rank of the first chunk (above the threshold) containing any snippet, else None."""
    wanted = [norm(s) for s in snippets]
    for rank, (content, _doc, similarity) in enumerate(results, start=1):
        if similarity <= THRESHOLD:
            continue
        text = norm(content)
        if any(w in text for w in wanted):
            return rank
    return None


def run_set(name, cases, org_id, k, verbose):
    if not cases:
        return None
    ranks, misses = [], []
    for question, snippets, doc in cases:
        results = semantic_search(question, org_id, top_k=k, doc_name=doc)
        rank = first_hit_rank(results, snippets)
        ranks.append(rank)
        if rank is None:
            best = max((r[2] for r in results), default=0.0)
            misses.append((question, [(d, round(s, 2)) for _c, d, s in results[:3]], round(best, 2)))
        elif verbose:
            print(f"  hit at rank {rank}: {question}")
    n = len(cases)
    hit1 = sum(1 for r in ranks if r == 1)
    hit3 = sum(1 for r in ranks if r is not None and r <= 3)
    hitk = sum(1 for r in ranks if r is not None)
    mrr = sum(1 / r for r in ranks if r) / n
    print(f"\n=== {name}  ({n} questions)")
    print(f"  right chunk is #1:        {hit1}/{n}  ({100*hit1//n}%)")
    print(f"  right chunk in top 3:     {hit3}/{n}  ({100*hit3//n}%)")
    print(f"  right chunk in top {k}:     {hitk}/{n}  ({100*hitk//n}%)")
    print(f"  mean reciprocal rank:     {mrr:.2f}   (1.00 = always first)")
    for question, top, best in misses:
        print(f"  MISSED: {question}")
        print(f"          best similarity {best}; top results: {top}")
    return hitk / n


def run_negative(org_id, k):
    wrong = []
    for question in NEGATIVE:
        results = semantic_search(question, org_id, top_k=k)
        best = max((r[2] for r in results), default=0.0)
        if best > THRESHOLD:
            wrong.append((question, round(best, 2), results[0][1]))
    n = len(NEGATIVE)
    print(f"\n=== QUESTIONS WITH NO ANSWER  ({n} questions; nothing should pass the {THRESHOLD} threshold)")
    print(f"  correctly found nothing:  {n - len(wrong)}/{n}")
    for question, best, doc in wrong:
        print(f"  FALSE MATCH: {question}  (similarity {best} with {doc})")
    return (n - len(wrong)) / n if n else None


def main():
    org_id = sys.argv[sys.argv.index("--org") + 1] if "--org" in sys.argv else "01"
    k = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else 5
    verbose = "--verbose" in sys.argv
    print(f"DocMind retrieval evaluation | org {org_id} | top {k} chunks | threshold {THRESHOLD}")
    main_score = run_set("MAIN SET", CASES, org_id, k, verbose)
    hold_score = run_set("HOLDOUT SET", HOLDOUT, org_id, k, verbose)
    neg_score = run_negative(org_id, k)
    print("\n=== SCORECARD")
    if main_score is not None:
        print(f"  Main set found in top {k}:    {100*main_score:.0f}%")
    if hold_score is not None:
        print(f"  Holdout found in top {k}:     {100*hold_score:.0f}%   (the honest number)")
    if neg_score is not None:
        print(f"  No-answer handled:          {100*neg_score:.0f}%")


if __name__ == "__main__":
    main()