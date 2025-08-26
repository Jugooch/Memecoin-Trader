"""
Lightweight P&L tracking using JSON file instead of SQLite
Provides persistent tracking of profits/losses across bot restarts
"""

import json
import os
import tempfile
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time


class PnLStore:
    """Simple JSON-based P&L tracking"""
    
    def __init__(self, path: str = "data/pnl_state.json", initial_capital: float = 100.0):
        self.path = path
        self.logger = logging.getLogger(__name__)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        # Default data structure
        self.data = {
            "starting_capital": initial_capital,
            "current_capital": initial_capital,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "daily_pnl": {},  # Date -> P&L mapping
            "trade_history": [],  # Recent trades for tracking
            "positions": {},  # Active positions
            "last_updated": datetime.utcnow().isoformat()
        }
        
        # Load existing data
        self._load()
    
    def _load(self):
        """Load data from JSON file"""
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    loaded_data = json.load(f)
                    # Merge with defaults to handle missing keys
                    self.data.update(loaded_data)
                    self.logger.info(f"Loaded P&L state: ${self.current_equity:.2f} equity")
            except Exception as e:
                self.logger.warning(f"Could not load P&L state: {e}, using defaults")
    
    def _atomic_write(self):
        """Atomically write data to file to prevent corruption"""
        self.data["last_updated"] = datetime.utcnow().isoformat()
        
        try:
            # Serialize data
            json_data = json.dumps(self.data, indent=2)
            
            # Write to temporary file first
            dir_name = os.path.dirname(self.path) or "."
            fd, tmp_path = tempfile.mkstemp(prefix=".pnl_", suffix=".tmp", dir=dir_name)
            
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(json_data)
                
                # Atomically replace the original file
                os.replace(tmp_path, self.path)
                
            finally:
                # Clean up temp file if it still exists
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                    
        except Exception as e:
            self.logger.error(f"Failed to save P&L state: {e}")
    
    def add_trade(self, 
                  action: str,
                  symbol: str,
                  mint_address: str,
                  amount: float,
                  price: float,
                  usd_value: float,
                  realized_pnl: Optional[float] = None,
                  paper_mode: bool = True):
        """Record a trade and update P&L"""
        
        trade_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "symbol": symbol,
            "mint": mint_address[:16],  # Shortened for storage
            "amount": round(amount, 8),
            "price": round(price, 8),
            "usd_value": round(usd_value, 2),
            "realized_pnl": round(realized_pnl, 2) if realized_pnl else None,
            "paper_mode": paper_mode
        }
        
        # Update totals
        self.data["total_trades"] += 1
        
        if action.upper() == "BUY":
            # Track position
            self.data["positions"][mint_address] = {
                "symbol": symbol,
                "amount": amount,
                "entry_price": price,
                "usd_invested": usd_value,
                "entry_time": datetime.utcnow().isoformat()
            }
            # Deduct from capital
            self.data["current_capital"] -= usd_value
            
        elif action.upper() == "SELL":
            # Update realized P&L
            if realized_pnl is not None:
                self.add_realized(realized_pnl)
                
                if realized_pnl > 0:
                    self.data["winning_trades"] += 1
                else:
                    self.data["losing_trades"] += 1
            
            # Add back to capital
            self.data["current_capital"] += usd_value
            
            # Remove or reduce position
            if mint_address in self.data["positions"]:
                position = self.data["positions"][mint_address]
                if amount >= position["amount"] * 0.99:  # Full sell
                    del self.data["positions"][mint_address]
                else:  # Partial sell
                    position["amount"] -= amount
        
        # Add to trade history (keep last 100 trades)
        self.data["trade_history"].append(trade_record)
        if len(self.data["trade_history"]) > 100:
            self.data["trade_history"] = self.data["trade_history"][-100:]
        
        # Save to disk
        self._atomic_write()
        
        return trade_record
    
    def add_realized(self, amount: float):
        """Add to realized P&L"""
        self.data["realized_pnl"] = round(self.data.get("realized_pnl", 0.0) + amount, 2)
        
        # Update daily P&L
        today = datetime.utcnow().date().isoformat()
        if today not in self.data["daily_pnl"]:
            self.data["daily_pnl"][today] = 0.0
        self.data["daily_pnl"][today] = round(self.data["daily_pnl"][today] + amount, 2)
        
        self._atomic_write()
    
    def update_unrealized(self, current_prices: Dict[str, float]):
        """Update unrealized P&L based on current prices"""
        total_unrealized = 0.0
        
        for mint, position in self.data["positions"].items():
            if mint in current_prices:
                current_value = position["amount"] * current_prices[mint]
                unrealized = current_value - position["usd_invested"]
                total_unrealized += unrealized
        
        self.data["unrealized_pnl"] = round(total_unrealized, 2)
        self._atomic_write()
    
    @property
    def current_equity(self) -> float:
        """Get current total equity (current capital + unrealized P&L from positions)"""
        return round(
            self.data["current_capital"] + 
            self.data.get("unrealized_pnl", 0.0),
            2
        )
    
    @property
    def total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)"""
        return round(
            self.data.get("realized_pnl", 0.0) + 
            self.data.get("unrealized_pnl", 0.0),
            2
        )
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate percentage"""
        total = self.data.get("winning_trades", 0) + self.data.get("losing_trades", 0)
        if total == 0:
            return 0.0
        return round((self.data.get("winning_trades", 0) / total) * 100, 1)
    
    def get_daily_pnl(self, date: Optional[str] = None) -> float:
        """Get P&L for a specific date (default: today)"""
        if date is None:
            date = datetime.utcnow().date().isoformat()
        return self.data["daily_pnl"].get(date, 0.0)
    
    def get_summary(self) -> Dict:
        """Get complete P&L summary"""
        return {
            "equity": self.current_equity,
            "starting_capital": self.data["starting_capital"],
            "current_capital": self.data["current_capital"],
            "realized_pnl": self.data.get("realized_pnl", 0.0),
            "unrealized_pnl": self.data.get("unrealized_pnl", 0.0),
            "total_pnl": self.total_pnl,
            "total_trades": self.data.get("total_trades", 0),
            "winning_trades": self.data.get("winning_trades", 0),
            "losing_trades": self.data.get("losing_trades", 0),
            "win_rate": self.win_rate,
            "active_positions": len(self.data.get("positions", {})),
            "daily_pnl": self.get_daily_pnl(),
            "last_updated": self.data.get("last_updated")
        }
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Get recent trade history"""
        return self.data.get("trade_history", [])[-limit:]
    
    def reset_daily_pnl(self):
        """Reset daily P&L (call at start of new day)"""
        # Clean up old daily P&L entries (keep last 30 days)
        if len(self.data["daily_pnl"]) > 30:
            dates = sorted(self.data["daily_pnl"].keys())
            for old_date in dates[:-30]:
                del self.data["daily_pnl"][old_date]
        
        self._atomic_write()
    
    def cleanup_old_data(self, days_to_keep: int = 7):
        """Clean up old trade history and daily P&L"""
        cutoff_date = (datetime.utcnow() - timedelta(days=days_to_keep)).isoformat()
        
        # Clean old trades
        self.data["trade_history"] = [
            trade for trade in self.data.get("trade_history", [])
            if trade.get("timestamp", "") > cutoff_date
        ]
        
        # Clean old daily P&L
        cutoff_date_str = (datetime.utcnow().date() - timedelta(days=days_to_keep)).isoformat()
        self.data["daily_pnl"] = {
            date: pnl for date, pnl in self.data.get("daily_pnl", {}).items()
            if date > cutoff_date_str
        }
        
        self._atomic_write()