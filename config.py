"""
Polyreasoner v4 Configuration
Reads from .env via python-dotenv. No hardcoded paths.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── Backend ────────────────────────────────────────────────────────────────
BACKEND_TYPE = os.getenv("POLYREASONER_BACKEND", "ollama")  # ollama | api

# Ollama settings
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:14b")

# API settings (LiteLLM — supports OpenAI, Anthropic, Gemini, Groq, etc.)
LITELLM_MODEL = os.getenv("LITELLM_MODEL", "gpt-4o")

# ─── Agents ─────────────────────────────────────────────────────────────────
AVAILABLE_AGENTS = [
    "business",     # Market fit, revenue, competitive advantage
    "risk",         # Threats, downsides, failure modes
    "security",     # Vulnerabilities, privacy, attack vectors
    "feasibility",  # Technical complexity, resources, timeline
    "impact",       # Long-term consequences, scalability
    "ethical",      # Moral implications, fairness
    "contrarian",   # Devil's advocate, argues against
]

DEFAULT_AGENTS = ["business", "risk", "contrarian"]
MAX_AGENTS = 5  # Increased for deeper analysis

# ─── ASRT Judge ─────────────────────────────────────────────────────────────
# HuggingFace model for prompt injection detection
JUDGE_DEBERTA_MODEL = os.getenv("JUDGE_DEBERTA_MODEL", "neuralchemy/prompt-injection-deberta")

# Path to sklearn classifiers (pickled models)
JUDGE_SKLEARN_DIR = os.getenv("JUDGE_SKLEARN_DIR", "")

# ─── Security / Shield ─────────────────────────────────────────────────────
SHIELD_LEVEL = int(os.getenv("SHIELD_LEVEL", "5"))  # 3=light, 5=standard, 7=strict
