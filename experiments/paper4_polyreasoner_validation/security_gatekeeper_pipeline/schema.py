"""
Neuralchemy Threat Matrix — Central Schema Definition
=====================================================
Every row in the dataset carries 5 independent labels:
  1. Intent     (7 classes)  — WHAT is the attacker trying to achieve?
  2. Technique  (8 classes)  — HOW is the attack constructed?
  3. Severity   (3 levels)   — How DANGEROUS / sophisticated is it?
  4. Surface    (4 classes)  — WHERE does the attack originate?
  5. Binary     (2 classes)  — Is it malicious at all? (0=benign, 1=malicious)
"""

# ─── DIMENSION 1: INTENT (7 classes) ─────────────────────────────────────────
INTENTS = [
    "benign",              # 0 — Safe, normal use
    "direct_injection",    # 1 — Directly ordering the AI to ignore rules
    "system_extraction",   # 2 — Trying to leak internal system prompt/data
    "role_hijack",         # 3 — Jailbreaks, DAN, forcing evil/unrestricted mode
    "obfuscation",         # 4 — Token smuggling, Base64, encoding tricks
    "tool_abuse",          # 5 — Manipulating plugins, code execution, agents
    "indirect_injection",  # 6 — Attacks hidden in documents, webpages, uploads
]
INTENT_TO_ID = {name: idx for idx, name in enumerate(INTENTS)}

# ─── DIMENSION 2: TECHNIQUE (8 classes) ──────────────────────────────────────
TECHNIQUES = [
    "none",                # 0 — No attack technique (benign)
    "keyword_override",    # 1 — "Ignore previous", "Disregard", "New instructions"
    "persona_play",        # 2 — "You are DAN", "Pretend you are evil"
    "encoding",            # 3 — Base64, hex, ROT13, unicode tricks
    "payload_splitting",   # 4 — Breaking the attack across multiple messages
    "context_overflow",    # 5 — Flooding context window to push out instructions
    "few_shot_poisoning",  # 6 — Embedding fake examples to steer output
    "multilingual",        # 7 — Using translation or foreign languages to bypass
]
TECHNIQUE_TO_ID = {name: idx for idx, name in enumerate(TECHNIQUES)}

# ─── DIMENSION 3: SEVERITY (3 levels) ────────────────────────────────────────
# 1 = Low/Trivial, 2 = Moderate, 3 = Advanced/Sophisticated
SEVERITY_LEVELS = [1, 2, 3]

# ─── DIMENSION 4: SURFACE (4 classes) ────────────────────────────────────────
SURFACES = [
    "user_input",    # 0 — Direct user-typed prompt
    "document",      # 1 — Hidden in uploaded PDF, DOCX, etc.
    "api",           # 2 — Injected via API call or tool response
    "tool_output",   # 3 — Embedded in the output of a connected tool
]
SURFACE_TO_ID = {name: idx for idx, name in enumerate(SURFACES)}

# ─── DIMENSION 5: BINARY (2 classes) ─────────────────────────────────────────
# The simplest usable label: is this prompt malicious at all?
# 0 = benign (safe, normal use)
# 1 = malicious (any attack intent)
BINARY_LABELS = {"benign": 0, "malicious": 1}

def get_binary_label(intent: str) -> int:
    """Return 0 if benign, 1 for any attack intent."""
    return 0 if intent == "benign" else 1

# ─── MAPPING: Original 31 Core Categories → 7 Intents ────────────────────────
CORE_CATEGORY_TO_INTENT = {
    # Benign
    "benign":                 "benign",

    # Direct Injection variants
    "direct_injection":       "direct_injection",
    "adversarial":            "direct_injection",
    "instruction_override":   "direct_injection",
    "goal_hijacking":         "direct_injection",
    "prompt_manipulation":    "direct_injection",
    "crescendo":              "direct_injection",
    "many_shot":              "direct_injection",
    "chain_of_thought":       "direct_injection",

    # System Extraction variants
    "system_extraction":      "system_extraction",
    "prompt_extraction":      "system_extraction",
    "prompt_leaking":         "system_extraction",
    "data_exfiltration":      "system_extraction",
    "context_extraction":     "system_extraction",

    # Role Hijack / Jailbreak variants
    "jailbreak":              "role_hijack",
    "role_hijack":            "role_hijack",
    "persona_replacement":    "role_hijack",
    "persona_adoption":       "role_hijack",
    "persona_manipulation":   "role_hijack",
    "role_play":              "role_hijack",
    "authority_manipulation": "role_hijack",

    # Obfuscation variants
    "obfuscation":            "obfuscation",
    "token_smuggling":        "obfuscation",
    "encoding_obfuscation":   "obfuscation",
    "payload_splitting":      "obfuscation",
    "language_switching":     "obfuscation",

    # Tool Abuse variants
    "tool_abuse":             "tool_abuse",
    "code_execution":         "tool_abuse",
    "agent_manipulation":     "tool_abuse",
    "plugin_exploitation":    "tool_abuse",

    # Indirect Injection variants
    "indirect_injection":     "indirect_injection",
    "contextual_injection":   "indirect_injection",
}

# ─── HEURISTIC: Detect Technique from Text ────────────────────────────────────
import re

def detect_technique(text: str) -> str:
    """Deterministically detect attack technique from prompt text."""
    text_lower = text.lower()

    # Check for encoding (Base64, hex, unicode)
    if re.search(r'[A-Za-z0-9+/]{20,}={0,2}', text):  # Base64-like
        return "encoding"
    if re.search(r'\\x[0-9a-f]{2}', text_lower) or re.search(r'\\u[0-9a-f]{4}', text_lower):
        return "encoding"
    # Zero-width characters
    if any(c in text for c in ['\u200b', '\u200c', '\u200d', '\ufeff']):
        return "encoding"

    # Check for persona play
    persona_keywords = ["you are now", "you are dan", "pretend you", "act as",
                        "roleplay as", "imagine you are", "from now on you",
                        "developer mode", "evil mode", "unrestricted"]
    if any(kw in text_lower for kw in persona_keywords):
        return "persona_play"

    # Check for keyword override
    override_keywords = ["ignore previous", "ignore all", "disregard",
                         "forget everything", "new instructions", "override",
                         "stop being", "do not follow", "ignore the above",
                         "ignore your"]
    if any(kw in text_lower for kw in override_keywords):
        return "keyword_override"

    # Check for multilingual
    non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / max(len(text), 1)
    if non_ascii_ratio > 0.3:
        return "multilingual"

    # Check for context overflow (very long prompts with repetition)
    if len(text) > 2000:
        return "context_overflow"

    # Check for few-shot poisoning
    few_shot_keywords = ["example:", "for example", "input:", "output:",
                         "q:", "a:", "sample:"]
    if sum(1 for kw in few_shot_keywords if kw in text_lower) >= 2:
        return "few_shot_poisoning"

    return "none"


def calculate_severity(text: str, technique: str, intent: str) -> int:
    """Calculate severity (1-3) using heuristic scoring."""
    if intent == "benign":
        return 1

    score = 0

    # Length complexity
    if len(text) > 500:
        score += 1
    if len(text) > 1500:
        score += 1

    # Technique sophistication
    advanced_techniques = ["encoding", "context_overflow", "few_shot_poisoning", "multilingual"]
    if technique in advanced_techniques:
        score += 2
    elif technique in ["persona_play", "payload_splitting"]:
        score += 1

    # Multi-line / structured attacks
    if text.count('\n') > 3:
        score += 1

    # Override keyword density
    override_count = sum(1 for kw in ["ignore", "disregard", "forget", "override",
                                       "bypass", "system", "admin"]
                         if kw in text.lower())
    if override_count >= 3:
        score += 1

    # Clamp to 1-3
    if score <= 1:
        return 1
    elif score <= 3:
        return 2
    else:
        return 3


def hackaprompt_level_to_severity(level: int) -> int:
    """Map HackAPrompt competition level (1-10) to severity (1-3)."""
    if level <= 3:
        return 1
    elif level <= 6:
        return 2
    else:
        return 3
