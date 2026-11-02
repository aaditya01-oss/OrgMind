"""
Reads the important fields from the OCR text of a student ID card and CHECKS them.

OCR is good at words but bad at telling 0 from O (and 1 from I) inside codes.
So we never trust an ID blindly: we compare it with the known shape of the ID
(KIC + 3 digits + program letters + 3 digits), fix only the obvious look-alike
mistakes in the digit positions, and say so. Anything we cannot fix is flagged
for a human to check. No AI is used here, so the result is the same every time.
"""
import re
from datetime import date, datetime

# Characters OCR often confuses with a digit, and what digit they most likely are.
_TO_DIGIT = str.maketrans({"O": "0", "o": "0", "Q": "0", "D": "0",
                           "I": "1", "l": "1", "|": "1", "S": "5", "B": "8", "Z": "2"})
# And the reverse, for the letter positions (the program code, e.g. BCT).
_TO_LETTER = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z"})

_DIGIT_LIKE = r"[0-9OoQDIl|SBZ]"
# KIC, then 3 digit-like characters, then 2-4 letter-like characters, then 3 digit-like characters
_ID_SEARCH = re.compile(
    rf"\b(K[I1l]C)({_DIGIT_LIKE}{{3}})([A-Z0-9]{{2,4}})({_DIGIT_LIKE}{{3}})\b"
)
_ID_STRICT = re.compile(r"^KIC\d{3}[A-Z]{2,4}\d{3}$")


def check_student_id(raw: str) -> dict:
    """Return {'value', 'status', 'raw'}; status is 'ok', 'corrected' or 'needs_review'."""
    raw = raw.strip()
    if _ID_STRICT.match(raw):
        return {"value": raw, "status": "ok", "raw": raw}
    m = _ID_SEARCH.search(raw.upper() if raw.isupper() else raw)
    if not m:
        return {"value": None, "status": "needs_review", "raw": raw}
    prefix, num1, program, num2 = m.groups()
    fixed = "KIC" + num1.translate(_TO_DIGIT) + program.upper().translate(_TO_LETTER) + num2.translate(_TO_DIGIT)
    if _ID_STRICT.match(fixed):
        return {"value": fixed, "status": "corrected", "raw": raw}
    return {"value": None, "status": "needs_review", "raw": raw}


def _find_id(text: str) -> dict:
    m = _ID_SEARCH.search(text)
    if not m:
        return {"value": None, "status": "needs_review", "raw": None}
    return check_student_id(m.group(0))


def _find_after_label(text: str, label: str) -> str | None:
    m = re.search(rf"{label}\s*[:;.\-]\s*(.+)", text, re.IGNORECASE)
    return m.group(1).strip() if m else None


def _parse_date(value: str | None):
    if not value:
        return None
    value = re.sub(r"\s+", " ", value.strip())
    for fmt in ("%d %b %Y", "%d %B %Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.title(), fmt).date()
        except ValueError:
            continue
    return None


def extract_id_card_fields(text: str, today: date | None = None) -> dict | None:
    """Fields from OCR text, or None if this does not look like one of our ID cards."""
    id_info = _find_id(text)
    name = _find_after_label(text, r"Name")
    valid_text = _find_after_label(text, r"Valid\s*up\s*to")
    if id_info["status"] == "needs_review" and not name and not valid_text:
        return None

    valid_date = _parse_date(valid_text)
    today = today or date.today()
    review = []
    if id_info["status"] == "needs_review":
        review.append("student_id")
    if not name:
        review.append("name")
    if valid_text and not valid_date:
        review.append("valid_upto")

    return {
        "student_id": id_info["value"],
        "student_id_status": id_info["status"],   # ok / corrected / needs_review
        "name": name,
        "valid_upto": valid_date.isoformat() if valid_date else None,
        "expired": (valid_date < today) if valid_date else None,
        "needs_review": review,
    }