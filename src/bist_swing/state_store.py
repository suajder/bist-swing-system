from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SeenGCConfig:
    # Keep dedup keys for the last N days to prevent unbounded growth of signals_state.json
    keep_days: int = 45


class StateStore:
    def __init__(self, path: Path, *, seen_gc: Optional[SeenGCConfig] = None):
        self.path = path
        self.seen_gc = seen_gc or SeenGCConfig()

    def load(self) -> Dict[str, Any]:
        """Load state from disk.

        If the JSON is corrupted (partial write, manual edit, etc.), the file is moved aside
        and an empty state is returned to keep the system running.
        """
        if not self.path.exists():
            return {}

        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Move the corrupted file aside for later inspection, then continue with empty state.
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            corrupt_path = self.path.with_suffix(self.path.suffix + f".corrupt.{ts}")
            try:
                os.replace(self.path, corrupt_path)
            except OSError:
                # If we can't move it, just continue with empty state.
                pass
            return {}

    def _gc_seen(self, state: Dict[str, Any]) -> None:
        seen = state.get("seen")
        if not isinstance(seen, dict) or not seen:
            return

        cutoff = date.today() - timedelta(days=int(self.seen_gc.keep_days))
        keep: Dict[str, Any] = {}

        # Convention in this repo: keys end with ::YYYY-MM-DD
        for k, v in seen.items():
            try:
                last = str(k).rsplit("::", 1)[-1]
                d = date.fromisoformat(last)
            except Exception:
                # Unknown format -> keep (safer than accidentally dropping dedup keys)
                keep[k] = v
                continue

            if d >= cutoff:
                keep[k] = v

        state["seen"] = keep

    def save(self, state: Dict[str, Any]) -> None:
        """Atomically persist state to disk (prevents partial writes)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Prevent unbounded growth of the dedup dictionary.
        self._gc_seen(state)

        payload = json.dumps(state, ensure_ascii=False, indent=2)

        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, self.path)

    def seen(self, state: Dict[str, Any], key: str) -> bool:
        return bool(state.get("seen", {}).get(key))

    def mark(self, state: Dict[str, Any], key: str) -> None:
        state.setdefault("seen", {})[key] = True

