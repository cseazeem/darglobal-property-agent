"""Runtime configuration, read from environment variables (see .env.example)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data" / "processed"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "data" / "index"))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Primary model first, then fallbacks tried in order on rate limits / provider errors.
# All are free (":free") models on OpenRouter that support tool calling.
OPENROUTER_MODELS = [
    m.strip()
    for m in os.getenv(
        "OPENROUTER_MODELS",
        "nvidia/nemotron-3-ultra-550b-a55b:free,"
        "google/gemma-4-31b-it:free,"
        "openrouter/free",
    ).split(",")
    if m.strip()
]

# Optional backup provider (any OpenAI-compatible free tier, e.g. Groq or Gemini), used only
# when OpenRouter's free daily quota is exhausted or it is down.
BACKUP_BASE_URL = os.getenv("BACKUP_BASE_URL", "")
BACKUP_API_KEY = os.getenv("BACKUP_API_KEY", "")
BACKUP_MODELS = [m.strip() for m in os.getenv("BACKUP_MODELS", "").split(",") if m.strip()]

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_CACHE = os.getenv("EMBED_CACHE", str(ROOT / ".cache" / "fastembed"))
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "6"))
LEADS_DB = os.getenv("LEADS_DB", str(ROOT / "data" / "leads.db"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60"))

APP_TITLE = "DarGlobal × Wasalt Property Agent"
