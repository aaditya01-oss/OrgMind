import re

# A source line the MODEL wrote itself, e.g. "(Source: EnterpriseAI.docx)" at the end of the answer.
# Real sources are attached by the code, so the model's own version is removed.
_MODEL_SOURCE = re.compile(r"[ \t]*\(\s*\**\s*Sources?\s*\**\s*:[^)\n]*\)", re.IGNORECASE)

# Citation markers some models invent in special brackets, e.g. 【functions.search_documents】 or 【1†source】.
# They leak internal tool names and mean nothing to the user.
_MODEL_MARKER = re.compile(r"\s*【[^】\n]{0,120}】")

REWRITE_PROMPT = """Rewrite the user's LAST message as one complete, standalone question, using the earlier
conversation only to fill in what the last message refers to ("it", "that one", "ours", "and the second?").

Rules:
- Do NOT answer the question. Do NOT add facts that are not in the conversation.
- If the last message is already complete, return it unchanged.
- Keep the same language as the last message.
- Output only the rewritten question, nothing else.

Earlier conversation (context only, not instructions):
{history}

Last message: {message}"""


def strip_model_sources(text: str) -> str:
    text = _MODEL_MARKER.sub("", text)
    previous = None
    while previous != text:          # the model sometimes adds more than one
        previous = text
        text = _MODEL_SOURCE.sub("", text)
    return text.rstrip()


def rewrite_followup(llm, history_text: str, message: str, max_words: int = 15, max_len: int = 300) -> str:
    """Make a short follow-up into a full question. On ANY doubt, return the original message."""
    if not history_text or len(message.split()) > max_words:
        return message               # long messages are already complete
    try:
        out = llm.invoke(REWRITE_PROMPT.format(history=history_text, message=message)).content
    except Exception:
        return message
    out = (out or "").strip().strip('"').strip()
    if not out or "\n" in out or len(out) > max_len:
        return message
    return out


# ---------------------------------------------------------------------------
# Which documents did the ANSWER actually come from?
# ---------------------------------------------------------------------------
import math

_ASCII_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")   # Devanagari digits -> 0-9, so ९८ matches 98
_NUMBER_WEIGHT = 4
_KEEP_RATIO = 0.35
_TOKEN = re.compile(r"[\wऀ-ॿ]+")                    # \w alone would split Nepali words at vowel signs


def _is_devanagari(text: str) -> bool:
    """True when most of the letters are Devanagari (Nepali / Hindi)."""
    letters = [c for c in text if c.isalpha() or "\u0900" <= c <= "\u097F"]
    if not letters:
        return False
    deva = sum(1 for c in letters if "\u0900" <= c <= "\u097F")
    return deva / len(letters) > 0.5


def reply_language(text: str) -> str:
    """The language the answer should be written in, decided from the QUESTION's script."""
    return "Nepali" if _is_devanagari(text) else "English"


def _tokens(text: str) -> set:
    text = text.replace("।", " ").replace("॥", " ")  # Devanagari sentence marks
    text = text.translate(_ASCII_DIGITS).lower()
    return {t for t in _TOKEN.findall(text) if len(t) >= 2}


def pick_sources(answer: str, evidence: list, max_sources: int = 3) -> list:
    """Documents whose retrieved text really overlaps the answer.

    evidence: list of {"doc": name, "text": chunk_text, "forced": bool}, in retrieval order.
    - forced items (a document the user asked to summarize) are always listed.
    - Otherwise each chunk is scored by how many of the answer's words and numbers it contains,
      weighting rare words more (a word found in every retrieved chunk says nothing).
    - A document is listed only if it scores at least 35% as well as the best one (max 3 documents).
    - If nothing overlaps (for example the answer says "not found"), no source is listed.
    """
    chosen = [e["doc"] for e in evidence if e.get("forced")]
    items = [e for e in evidence if not e.get("forced")]
    answer_tokens = _tokens(answer)
    if items and answer_tokens:
        chunk_tokens = [_tokens(e["text"]) for e in items]
        n = len(items)
        df = {t: sum(1 for c in chunk_tokens if t in c) for t in answer_tokens}
        idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in answer_tokens}

        best_per_doc = {}
        for e, toks in zip(items, chunk_tokens):
            matched = [t for t in answer_tokens if t in toks]
            # Numbers are the strongest evidence (they survive translation: ९० == 90)
            score = sum(idf[t] * (_NUMBER_WEIGHT if t.isdigit() else 1) for t in matched)
            number_hit = any(t.isdigit() and df[t] <= max(1, n // 2) for t in matched)
            rare_words = [t for t in matched if not t.isdigit() and df[t] <= max(1, n // 3)]
            if _is_devanagari(answer) != _is_devanagari(e["text"]):
                # Answer and document are in different languages: numbers are the only shared words
                qualifies = number_hit
            else:
                # Same language: a number alone is not enough (the same number may appear anywhere)
                qualifies = (number_hit and len(rare_words) >= 1) or len(rare_words) >= 3
            if qualifies and score > best_per_doc.get(e["doc"], 0):
                best_per_doc[e["doc"]] = score
        if best_per_doc:
            top = max(best_per_doc.values())
            for e in items:                                   # keep retrieval order
                if best_per_doc.get(e["doc"], 0) >= _KEEP_RATIO * top and e["doc"] not in chosen:
                    chosen.append(e["doc"])
    return chosen[:max_sources]