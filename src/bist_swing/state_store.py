from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

class StateStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, state: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def seen(self, state: Dict[str, Any], key: str) -> bool:
        return bool(state.get("seen", {}).get(key))

    def mark(self, state: Dict[str, Any], key: str) -> None:
        state.setdefault("seen", {})[key] = True
