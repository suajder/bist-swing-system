import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import time

STATE_FILE = Path("out/live/live_portfolio.json")

@dataclass
class Position:
    ticker: str
    entry_price: float
    qty: float
    stop_price: float
    risk_amount: float
    entry_time: float
    tp1_done: bool = False
    tp2_done: bool = False

@dataclass
class PortfolioState:
    capital: float
    risk_pct: float
    max_open: int
    friction_pct: float # commission + slippage
    positions: List[Position]

class PortfolioManager:
    def __init__(self, path: Path = STATE_FILE):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self.load_state()

    def load_state(self) -> PortfolioState:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                positions = [Position(**p) for p in data.get("positions", [])]
                return PortfolioState(
                    capital=data.get("capital", 100000.0),
                    risk_pct=data.get("risk_pct", 1.5),
                    max_open=data.get("max_open", 5),
                    friction_pct=data.get("friction_pct", 0.0025),
                    positions=positions
                )
            except Exception as e:
                print(f"Error loading portfolio state: {e}")
        
        # Default state
        state = PortfolioState(capital=100000.0, risk_pct=1.5, max_open=5, friction_pct=0.0025, positions=[])
        self.save_state(state)
        return state

    def save_state(self, state: Optional[PortfolioState] = None):
        if state:
            self.state = state
        
        data = {
            "capital": self.state.capital,
            "risk_pct": self.state.risk_pct,
            "max_open": self.state.max_open,
            "friction_pct": self.state.friction_pct,
            "positions": [asdict(p) for p in self.state.positions]
        }
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def update_settings(self, capital: float, risk_pct: float, max_open: int, friction_pct: float):
        self.state.capital = capital
        self.state.risk_pct = risk_pct
        self.state.max_open = max_open
        self.state.friction_pct = friction_pct
        self.save_state()

    def add_position(self, ticker: str, entry_price: float, stop_price: float):
        if len(self.state.positions) >= self.state.max_open:
            raise ValueError("Maksimum açık pozisyon sınırına ulaşıldı.")
            
        for p in self.state.positions:
            if p.ticker == ticker:
                raise ValueError(f"{ticker} zaten portföyde mevcut.")

        risk_amount = self.state.capital * (self.state.risk_pct / 100.0)
        risk_per_share = entry_price - stop_price
        
        if risk_per_share <= 0:
            raise ValueError("Stop fiyatı giriş fiyatından küçük/eşit olmalı.")

        qty = max(1, int(risk_amount / risk_per_share))
        
        # Apply friction to entry capital deduction
        notional = qty * entry_price
        friction_cost = notional * self.state.friction_pct
        deduction = notional + friction_cost
        
        if deduction > self.state.capital:
            raise ValueError("Yetersiz sermaye.")
            
        self.state.capital -= deduction
        
        new_pos = Position(
            ticker=ticker,
            entry_price=entry_price,
            qty=qty,
            stop_price=stop_price,
            risk_amount=risk_amount,
            entry_time=time.time()
        )
        self.state.positions.append(new_pos)
        self.save_state()
        
    def close_position(self, ticker: str, current_price: float, fraction: float = 1.0, mark_tp1: bool = False, mark_tp2: bool = False) -> float:
        """Kapatılan PnL döner. fraction 0.5 ise %50'sini kapatır."""
        pos = next((p for p in self.state.positions if p.ticker == ticker), None)
        if not pos:
            return 0.0

        close_qty = int(pos.qty * fraction)
        if close_qty <= 0:
            return 0.0

        notional_value = close_qty * current_price
        friction_cost = notional_value * self.state.friction_pct
        proceeds = notional_value - friction_cost
        
        # PnL Calculation (excluding entry friction, only exit friction for simplicity of representation)
        pnl = (current_price - pos.entry_price) * close_qty - friction_cost
        
        self.state.capital += proceeds
        pos.qty -= close_qty
        
        if mark_tp1:
            pos.tp1_done = True
        if mark_tp2:
            pos.tp2_done = True

        if pos.qty <= 0.001:  # full close
            self.state.positions = [p for p in self.state.positions if p.ticker != ticker]
            
        self.save_state()
        return pnl

