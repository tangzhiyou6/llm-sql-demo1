import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# LLM Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

# Dual-model routing
FAST_MODEL = os.getenv("FAST_MODEL", "deepseek-chat")
REASONING_MODEL = os.getenv("REASONING_MODEL", "deepseek-reasoner")

# Database Sandbox Configuration
raw_db_path = os.getenv("DB_PATH", "ecommerce.db")
_db_p = Path(raw_db_path)
if not _db_p.is_absolute():
    DB_PATH = str((BASE_DIR / _db_p).resolve())
else:
    DB_PATH = str(_db_p.resolve())
STATEMENT_TIMEOUT_SECONDS = float(os.getenv("STATEMENT_TIMEOUT_SECONDS", "5.0"))
MAX_ROW_LIMIT = int(os.getenv("MAX_ROW_LIMIT", "100"))

# Pipeline Safeguards
MAX_SELF_HEAL_ROUNDS = int(os.getenv("MAX_SELF_HEAL_ROUNDS", "3"))
DIALECT = os.getenv("DB_DIALECT", "sqlite")
