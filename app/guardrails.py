import re, base64
from typing import Optional

INJECTION_PATTERNS = [
    r"ignore (all|previous|above) instructions",
    r"(reveal|show|display|print|give me) (the )?(system|developer|hidden) prompt",
    r"act as (system|developer|unrestricted)",
    r"override .* rules",
    r"you are now",
    r"jailbreak",
    r"follow the (next|below) instructions",
    r"api key",
    r"environment variable",
    r"env",
    "(system|developer) prompt",
]

DEFAULT_SYSTEM = "You are a concise, literal assistant. Be safe and stick to instructions."

RE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
RE_PHONE = re.compile(r"(?:\+?48)?\s?(?:\d[ -]?){9,}")
RE_PESEL = re.compile(r"\b\d{11}\b")
RE_CARD  = re.compile(r"\b(?:\d[ -]?){13,19}\b")
PROFANITY = {"damn", "hell", "crap", "bloody", "shit", "fuck", "screw", "bastard", "jerk", "wtf"}

ALLOWED_DOMAINS = {"example.com", "spacenews.com", "www.nasa.gov", "science.nasa.gov"}

RE_PATH_TRAVERSAL = re.compile(r"(\.\./|\.\.\\)+")

RE_BASE64 = re.compile(r"^[A-Za-z0-9+/=]{16,}$")

def _printable_ratio(s: str) -> float:
    if not s:
        return 0.0
    printable = sum(ch.isprintable() for ch in s)
    return printable / len(s)

def try_decode_base64(text: str) -> Optional[str]:
    t = (text or "").strip()
    if len(t) < 16 or (len(t) % 4 != 0):
        return None
    if not RE_BASE64.match(t):
        return None
    try:
        raw = base64.b64decode(t, validate=True)
    except Exception:
        return None
    decoded = raw.decode("utf-8", errors="ignore").strip()
    if len(decoded) < 8:
        return None
    if _printable_ratio(decoded) < 0.7:
        return None
    return decoded

def contains_path_traversal(text: str) -> bool:
    low = (text or "").lower()
    return bool(RE_PATH_TRAVERSAL.search(low)) or "system32" in low or "cmd.exe" in low

def contains_profanity(text: str) -> bool:
    text_lower = text.lower()
    words = set(re.findall(r"\b\w+\b", text_lower))
    return any(w in words for w in PROFANITY)

def detect_injection(text: str) -> bool:
    text_lower = (text or "").lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    decoded = try_decode_base64(text)
    if decoded:
        decoded_lower = decoded.lower()
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, decoded_lower):
                return True
        if any(x in decoded_lower for x in [
            "system prompt", "developer prompt", "api key", "environment variable", ".env", "secret"
        ]):
            return True
    return False

def scrub_user_input(text: str) -> str:
    out = re.sub(
        r"(?i)(ignore (all|previous|above) instructions|reveal (system|developer) prompt|jailbreak|act as developer)","", text)
    clean = re.sub(r"(?i)\[system\].*?\[\/system\]", "", out)
    clean = re.sub(r"<script.*?>.*?</script>", "", clean, flags=re.DOTALL)
    return clean.strip()

def contains_pii(text: str):
    return {"email": bool(RE_EMAIL.search(text)),
            "phone": bool(RE_PHONE.search(text)),
            "pesel": bool(RE_PESEL.search(text)),
            "card": bool(RE_CARD.search(text))}

def links_not_allowed(text: str):
    urls = re.findall(r"https?://([^/\s]+)", text)
    return any(u.lower() not in ALLOWED_DOMAINS for u in urls)

