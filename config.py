"""
Polyreasoner v4 Configuration
Reads persistently from config/settings.json, falling back to .env via python-dotenv.
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CONFIG_DIR = Path(__file__).parent
SETTINGS_FILE = CONFIG_DIR / "settings.json"

class SettingsManager:
    """Manages loading, updating, and saving persistent settings for Polyreasoner."""
    
    @staticmethod
    def load() -> dict:
        defaults = {
            "POLYREASONER_BACKEND": os.getenv("POLYREASONER_BACKEND", "ollama"),
            "OLLAMA_HOST": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL", "qwen2.5-coder:14b"),
            "LITELLM_MODEL": os.getenv("LITELLM_MODEL", "gpt-4o"),
            "OLLAMA_FAST_MODEL": os.getenv("OLLAMA_FAST_MODEL", "llama3.1:8b"),
            "OLLAMA_SMART_MODEL": os.getenv("OLLAMA_SMART_MODEL", "qwen2.5-coder:14b"),
            "API_FAST_MODEL": os.getenv("API_FAST_MODEL", "gpt-4o-mini"),
            "API_SMART_MODEL": os.getenv("API_SMART_MODEL", "gpt-4o"),
            "SHIELD_LEVEL": int(os.getenv("SHIELD_LEVEL", "5")),
            "API_KEYS": {
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
                "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
                "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", "")
            }
        }
        
        if SETTINGS_FILE.exists():
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    # Merge saved settings into defaults to handle new keys gracefully
                    for k, v in saved.items():
                        if isinstance(v, dict) and k in defaults:
                            defaults[k].update(v)
                        else:
                            defaults[k] = v
            except Exception as e:
                print(f"[!] Warning: Failed to load settings.json: {e}. Using defaults.")
        return defaults

    @staticmethod
    def save(settings: dict) -> bool:
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2)
            return True
        except Exception as e:
            print(f"[!] Error: Failed to save settings to settings.json: {e}")
            return False

# Load active settings
_settings = SettingsManager.load()

# ─── Active Configurations (Exported Variables) ────────────────────────────
BACKEND_TYPE = _settings["POLYREASONER_BACKEND"]
OLLAMA_HOST = _settings["OLLAMA_HOST"]
OLLAMA_MODEL = _settings["OLLAMA_MODEL"]
LITELLM_MODEL = _settings["LITELLM_MODEL"]
SHIELD_LEVEL = _settings["SHIELD_LEVEL"]

# Apply saved API keys back to the environment dynamically
for key, value in _settings.get("API_KEYS", {}).items():
    if value:
        os.environ[key] = value

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
MAX_AGENTS = 5

# ─── ASRT Judge ─────────────────────────────────────────────────────────────
JUDGE_DEBERTA_MODEL = os.getenv("JUDGE_DEBERTA_MODEL", "neuralchemy/prompt-injection-deberta")
JUDGE_SKLEARN_DIR = os.getenv("JUDGE_SKLEARN_DIR", "")

