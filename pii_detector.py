import re

PII_PATTERNS = {
    "phone_np": r"\+?977[-\s]?[9][6-9]\d{8}",
    "phone_generic": r"\b[9][6-9]\d{8}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "pan_number": r"\bPAN[-:\s]?\d{9}\b",
    "bank_account": r"\b\d{9,16}\b",
    "citizenship": r"\b\d{2}-\d{2}-\d{2}-\d{5}\b",
    "salary": r"\b(NPR|Rs\.?|NRS)\s*[\d,]+\b",
    "nid": r"\bNID[-:\s]?\d{10,12}\b",
}

def detect_pii(text: str) -> dict:
    found_pii = {}
    
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            found_pii[pii_type] = matches
    
    return found_pii

def mask_pii(text: str) -> tuple[str, dict]:
    found_pii = detect_pii(text)
    masked_text = text
    
    for pii_type, pattern in PII_PATTERNS.items():
        def replace_match(match):
            original = match.group()
            masked = original[:2] + "*" * (len(original) - 4) + original[-2:]
            return masked
        
        masked_text = re.sub(pattern, replace_match, masked_text, flags=re.IGNORECASE)
    
    return masked_text, found_pii

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

if __name__ == "__main__":
    test_inputs = [
        "My phone number is 9841234567 and email is aaditya@gmail.com",
        "Process salary NPR 85,000 for account 0123456789",
        "My citizenship number is 12-34-56-78901",
        "Can you help me with the leave policy?",
        "Contact me at +977-9841234567 or aaditya@orgmind.com",
    ]
    
    print("OrgMind PII Detector\n")
    for test in test_inputs:
        result = scan_for_pii(test)
        status = "⚠️  PII FOUND" if result["contains_pii"] else "✅ CLEAN"
        print(f"{status}: {test[:50]}...")
        if result["contains_pii"]:
            print(f"   Types: {result['pii_types_found']}")
            print(f"   Masked: {result['masked_text']}")
        print()