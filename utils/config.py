from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Lit settings.yaml une seule fois pour tout le processus."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)