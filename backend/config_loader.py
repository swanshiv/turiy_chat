import yaml
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv

_CONFIG_DIR = Path(__file__).resolve().parent
_ENV_PATH = _CONFIG_DIR / ".env"

if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)
else:  # pragma: no cover - fallback when running from repo root
    load_dotenv()


@lru_cache(maxsize=1)
def load_agent_config() -> dict:
    config_path = _CONFIG_DIR / "config" / "agent_config.yaml"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}

