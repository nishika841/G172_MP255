import yaml
from pathlib import Path
from typing import Any, Dict


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    """Load configuration parameters safely using yaml.safe_load."""
    path = Path(params_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {params_path}")
    with open(path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params
