"""
utils/__init__.py — shared helpers: logging, env loading, path resolution.
"""

import os
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

# ── project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

def load_env() -> None:
    """Load .env from project root (safe to call multiple times)."""
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.debug(f"Loaded env from {env_path}")
    else:
        logger.warning(".env not found — relying on shell environment variables.")

def get_env(key: str, default: str | None = None) -> str:
    """Return env var or raise if missing and no default given."""
    load_env()
    val = os.getenv(key, default)
    if val is None:
        raise EnvironmentError(f"Required env var '{key}' is not set.")
    return val

def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist, return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_env() -> None:
    env_path = ROOT / ".env"

    print("ROOT =", ROOT)
    print("ENV PATH =", env_path)
    print("EXISTS =", env_path.exists())

    if env_path.exists():
        load_dotenv(env_path)
        logger.debug(f"Loaded env from {env_path}")
    else:
        logger.warning(".env not found — relying on shell environment variables.")