import re

# Digits: ASCII 0-9 plus Devanagari digits ०-९ (Nepali documents use both).
DIG = "0-9०-९"

# Each pattern may contain a named group "v". If it does, ONLY that group is the
# sensitive value (so "PAN 123456789" keeps the word PAN and hides the number).
PII_PATTERNS = {
    # Nepali mobile numbers, optional +977, ASCII or Devanagari digits
    "phone_np": rf"(?<![{DIG}])(?:\+?(?:977|९७७)[-\s]?)?[9९][6-9६-९][{DIG}]{{8}}(?![{DIG}])",
    "email": r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "pan_number": rf"(?<!\w)(?:PAN|प्यान)\s*(?:no\.?|number|नं\.?|नम्बर)?[-:\s]*(?P<v>[{DIG}]{{9}})(?![{DIG}])",
    "citizenship": rf"(?<![{DIG}])[{DIG}]{{2}}-[{DIG}]{{2}}-[{DIG}]{{2}}-[{DIG}]{{5}}(?![{DIG}])",
    "nid": rf"(?<!\w)NID[-:\s]*(?P<v>[{DIG}]{{10,12}})(?![{DIG}])",
    # Long digit strings only count as a bank account when a bank/account word is nearby.
    # This avoids flagging order numbers and ticket numbers.
    "bank_account": rf"(?:account|a/c|acct|bank|iban|खाता|बैंक)[^{DIG}\n]{{0,25}}(?P<v>[{DIG}]{{9,16}})(?![{DIG}])",
    # Money amounts in rupees: "NPR 85,000", "Rs. 5,000", "रू ८५,०००", "85,000 rupees"
    "npr_amount": (
        rf"(?<!\S)(?:NPR|NRS|Rs\.?|रू\.?|रु\.?)\s*[{DIG}][{DIG},]*(?:\.[{DIG}]+)?"
        rf"|(?<![{DIG},.])[{DIG}][{DIG},]*(?:\.[{DIG}]+)?\s*(?:rupees?|rs\b\.?|npr|nrs|रुपैयाँ|रुपियाँ|रुपैया)"
    ),
}

COMPILED = {t: re.compile(p, re.IGNORECASE) for t, p in PII_PATTERNS.items()}

PLACEHOLDER_LABEL = {
    "phone_np": "PHONE",
    "email": "EMAIL",
    "pan_number": "PAN",
    "citizenship": "CITIZENSHIP",
    "nid": "NID",
    "bank_account": "ACCOUNT",
    "npr_amount": "AMOUNT",
}


def _value(m: re.Match) -> str:
    """The sensitive part of a match."""
    if "v" in m.re.groupindex and m.group("v"):
        return m.group("v")
    return m.group()


def _sub_value(m: re.Match, fn) -> str:
    """Rebuild the match text with fn(sensitive_part) in place of the sensitive part."""
    if "v" in m.re.groupindex and m.group("v"):
        start, end = m.span("v")
        full, base = m.group(), m.start()
        return full[: start - base] + fn(m.group("v")) + full[end - base:]
    return fn(m.group())


def _mask(value: str) -> str:
    if len(value) <= 4:
        return "*" * len(value)
    return value[:2] + "*" * (len(value) - 4) + value[-2:]


def detect_pii(text: str) -> dict:
    found = {}
    for pii_type, rx in COMPILED.items():
        values = [_value(m) for m in rx.finditer(text)]
        if values:
            found[pii_type] = values
    return found


def mask_pii(text: str) -> tuple[str, dict]:
    """Irreversible partial masking (e.g. 98******67). Used for what we STORE."""
    found = detect_pii(text)
    masked = text
    for rx in COMPILED.values():
        masked = rx.sub(lambda m: _sub_value(m, _mask), masked)
    return masked, found


def scan_for_pii(text: str) -> dict:
    found = detect_pii(text)
    masked, _ = mask_pii(text)

    return {
        "contains_pii": len(found) > 0,
        "pii_types_found": list(found.keys()),
        "masked_text": masked,
        "original_length": len(text),
        "pii_count": sum(len(v) for v in found.values())
    }


def pseudonymize(text: str, mapping: dict | None = None) -> tuple[str, dict]:
    """
    Replace PII with placeholders such as [PHONE_1] so the text can be sent to an
    LLM without the real values. Returns (new_text, mapping) where mapping is
    {placeholder: original}. The same value always gets the same placeholder.
    Pass the same mapping dict back in to keep numbering consistent across calls.
    """
    mapping = {} if mapping is None else mapping
    reverse = {original: ph for ph, original in mapping.items()}

    for pii_type, rx in COMPILED.items():
        label = PLACEHOLDER_LABEL[pii_type]

        def repl(m, label=label):
            def to_placeholder(value):
                if value in reverse:
                    return reverse[value]
                n = sum(1 for p in mapping if p.startswith(f"[{label}_")) + 1
                placeholder = f"[{label}_{n}]"
                mapping[placeholder] = value
                reverse[value] = placeholder
                return placeholder
            return _sub_value(m, to_placeholder)

        text = rx.sub(repl, text)
    return text, mapping


def restore_pii(text: str, mapping: dict) -> str:
    """Put the real values back into an LLM answer."""
    for placeholder, original in mapping.items():
        text = text.replace(placeholder, original)
    return text


if __name__ == "__main__":
    for sample in [
        "My phone number is 9841234567 and email is aaditya@gmail.com",
        "Process salary NPR 85,000 for account 0123456789",
        "Can you help me with the leave policy?",
    ]:
        r = scan_for_pii(sample)
        print(("PII  " if r["contains_pii"] else "CLEAN"), r["pii_types_found"], r["masked_text"])