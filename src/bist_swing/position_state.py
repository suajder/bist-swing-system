from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class Position:
    ticker: str
    entry_date: str
    entry_px: float
    stop_px: float
    r: float
    qty: float = 1.0
    tp1_done: bool = False
    tp2_done: bool = False
    is_open: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Position":
        return Position(**d)
