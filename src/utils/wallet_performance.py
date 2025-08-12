"""
Wallet performance tracking with JSON storage
Tracks individual wallet metrics and performance over time
"""

import json
import os
import tempfile
import logging
import time
from typing import Dict, List, Optional
from datetime import datetime

class WalletPerformanceTracker:
    """JSON-based wallet performance tracking"""
    
    def __init__(self, path: str = "data/wallet_performance.json"):
        self.path = path
        self.logger = logging.getLogger(__name__)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        # Default data structure
        self.data = {
            "wallets": {},  # {wallet_address: wallet_metrics}
            "last_updated": datetime.utcnow().isoformat(),
            "last_rotation": None,
            "rotation_history": []
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
                    self.logger.info(f"Loaded wallet performance data for {len(self.data.get('wallets', {}))} wallets")
            except Exception as e:
                self.logger.warning(f"Could not load wallet performance data: {e}, using defaults")
    
    def _atomic_write(self):
        """Atomically write data to file to prevent corruption"""
        self.data["last_updated"] = datetime.utcnow().isoformat()
        
        try:
            # Serialize data
            json_data = json.dumps(self.data, indent=2)
            
            # Write to temporary file first
            dir_name = os.path.dirname(self.path) or "."
            fd, tmp_path = tempfile.mkstemp(prefix=".wallet_perf_", suffix=".tmp", dir=dir_name)
            
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
            self.logger.error(f"Failed to save wallet performance data: {e}")
    
    def initialize_wallet(self, wallet_address: str):
        """Initialize tracking for a new wallet"""
        if wallet_address not in self.data["wallets"]:
            self.data["wallets"][wallet_address] = {
                "address": wallet_address,
                "first_seen": time.time(),
                "last_activity": time.time(),
                "trades_detected": 0,
                "trades_followed": 0,
                "wins": 0,
                "losses": 0,
                "total_pnl_pct": 0.0,
                "avg_pnl_pct": 0.0,
                "current_tier": "Unknown",
                "evaluation_score": 0.0,
                "last_evaluated": time.time(),
                "trade_history": []  # Recent trades for analysis
            }
            self._atomic_write()
    
    def record_wallet_activity(self, wallet_address: str):
        """Record that a wallet was active (bought a token we detected)"""
        self.initialize_wallet(wallet_address)
        
        wallet_data = self.data["wallets"][wallet_address]
        wallet_data["last_activity"] = time.time()
        wallet_data["trades_detected"] += 1
        
        self._atomic_write()
    
    def record_trade_follow(self, wallet_address: str, mint_address: str, confidence_score: float):
        """Record that we followed this wallet's trade"""
        self.initialize_wallet(wallet_address)
        
        wallet_data = self.data["wallets"][wallet_address]
        wallet_data["trades_followed"] += 1
        
        # Add to trade history
        trade_record = {
            "timestamp": time.time(),
            "mint": mint_address[:16],  # Shortened for storage
            "confidence_score": confidence_score,
            "outcome": None,  # Will be updated when trade closes
            "pnl_pct": None
        }
        
        wallet_data["trade_history"].append(trade_record)
        
        # Keep only last 20 trades per wallet
        if len(wallet_data["trade_history"]) > 20:
            wallet_data["trade_history"] = wallet_data["trade_history"][-20:]
        
        self._atomic_write()
    
    def record_trade_outcome(self, mint_address: str, pnl_pct: float):
        """Record the outcome of a trade for all wallets that triggered it"""
        is_win = pnl_pct > 0
        
        for wallet_address, wallet_data in self.data["wallets"].items():
            # Find the trade in this wallet's history
            for trade in reversed(wallet_data["trade_history"]):
                if trade["mint"] == mint_address[:16] and trade["outcome"] is None:
                    # Update the trade outcome
                    trade["outcome"] = "win" if is_win else "loss"
                    trade["pnl_pct"] = pnl_pct
                    
                    # Update wallet stats
                    if is_win:
                        wallet_data["wins"] += 1
                    else:
                        wallet_data["losses"] += 1
                    
                    # Update PnL tracking
                    total_trades = wallet_data["wins"] + wallet_data["losses"]
                    wallet_data["total_pnl_pct"] += pnl_pct
                    wallet_data["avg_pnl_pct"] = wallet_data["total_pnl_pct"] / total_trades if total_trades > 0 else 0
                    
                    break
        
        self._atomic_write()
    
    def evaluate_wallet(self, wallet_address: str) -> float:
        """Calculate evaluation score for a wallet (0-100)"""
        if wallet_address not in self.data["wallets"]:
            return 0.0
        
        wallet_data = self.data["wallets"][wallet_address]
        current_time = time.time()
        
        # Activity Score (0-40 points)
        last_activity = wallet_data.get("last_activity", 0)
        time_since_activity = current_time - last_activity
        
        if time_since_activity < 2 * 3600:  # < 2 hours
            activity_score = 40
        elif time_since_activity < 6 * 3600:  # < 6 hours
            activity_score = 30
        elif time_since_activity < 12 * 3600:  # < 12 hours
            activity_score = 20
        elif time_since_activity < 24 * 3600:  # < 24 hours
            activity_score = 10
        else:
            activity_score = 0
        
        # Performance Score (0-40 points)
        total_trades = wallet_data["wins"] + wallet_data["losses"]
        if total_trades >= 3:  # Need at least 3 trades for reliable stats
            win_rate = wallet_data["wins"] / total_trades
            performance_score = win_rate * 40
        else:
            performance_score = 20  # Neutral score for new wallets
        
        # Hit Rate Score (0-20 points)
        trades_detected = wallet_data.get("trades_detected", 0)
        trades_followed = wallet_data.get("trades_followed", 0)
        
        if trades_detected > 0:
            hit_rate = trades_followed / trades_detected
            hit_rate_score = hit_rate * 20
        else:
            hit_rate_score = 10  # Neutral score
        
        total_score = activity_score + performance_score + hit_rate_score
        
        # Update the wallet's evaluation
        wallet_data["evaluation_score"] = total_score
        wallet_data["last_evaluated"] = current_time
        
        return total_score
    
    def get_wallets_for_rotation(self, min_score: float = 50.0) -> Dict[str, List[str]]:
        """Get wallets categorized for rotation decisions"""
        keep_wallets = []
        replace_candidates = []
        replace_wallets = []
        
        for wallet_address in self.data["wallets"]:
            score = self.evaluate_wallet(wallet_address)
            
            if score >= 70:
                keep_wallets.append(wallet_address)
            elif score >= min_score:
                # Check if active in last 6 hours
                wallet_data = self.data["wallets"][wallet_address]
                last_activity = wallet_data.get("last_activity", 0)
                if time.time() - last_activity < 6 * 3600:
                    keep_wallets.append(wallet_address)
                else:
                    replace_candidates.append(wallet_address)
            else:
                replace_wallets.append(wallet_address)
        
        return {
            "keep": keep_wallets,
            "candidates": replace_candidates,
            "replace": replace_wallets
        }
    
    def record_rotation(self, kept_wallets: List[str], new_wallets: List[str], replaced_wallets: List[str]):
        """Record a wallet rotation event"""
        rotation_record = {
            "timestamp": time.time(),
            "kept_count": len(kept_wallets),
            "new_count": len(new_wallets),
            "replaced_count": len(replaced_wallets),
            "kept_wallets": [w[:8] + "..." for w in kept_wallets[:5]],  # Sample for logging
            "new_wallets": [w[:8] + "..." for w in new_wallets[:5]],
            "replaced_wallets": [w[:8] + "..." for w in replaced_wallets[:5]]
        }
        
        self.data["rotation_history"].append(rotation_record)
        self.data["last_rotation"] = time.time()
        
        # Keep only last 10 rotation records
        if len(self.data["rotation_history"]) > 10:
            self.data["rotation_history"] = self.data["rotation_history"][-10:]
        
        self._atomic_write()
        
        self.logger.info(f"Wallet rotation recorded: kept {len(kept_wallets)}, "
                        f"added {len(new_wallets)}, replaced {len(replaced_wallets)}")
    
    def get_wallet_stats(self, wallet_address: str) -> Dict:
        """Get detailed stats for a specific wallet"""
        if wallet_address not in self.data["wallets"]:
            return {}
        
        wallet_data = self.data["wallets"][wallet_address].copy()
        
        # Add calculated metrics
        total_trades = wallet_data["wins"] + wallet_data["losses"]
        wallet_data["total_trades"] = total_trades
        wallet_data["win_rate"] = wallet_data["wins"] / total_trades if total_trades > 0 else 0
        
        # Add activity status
        last_activity = wallet_data.get("last_activity", 0)
        hours_since_activity = (time.time() - last_activity) / 3600
        wallet_data["hours_since_activity"] = hours_since_activity
        wallet_data["is_active"] = hours_since_activity < 6
        
        return wallet_data
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        total_wallets = len(self.data["wallets"])
        active_wallets = 0
        total_trades = 0
        total_wins = 0
        
        for wallet_data in self.data["wallets"].values():
            last_activity = wallet_data.get("last_activity", 0)
            if time.time() - last_activity < 6 * 3600:  # Active in last 6 hours
                active_wallets += 1
            
            total_trades += wallet_data["wins"] + wallet_data["losses"]
            total_wins += wallet_data["wins"]
        
        overall_win_rate = total_wins / total_trades if total_trades > 0 else 0
        
        return {
            "total_wallets": total_wallets,
            "active_wallets": active_wallets,
            "activity_rate": active_wallets / total_wallets if total_wallets > 0 else 0,
            "total_trades": total_trades,
            "overall_win_rate": overall_win_rate,
            "last_rotation": self.data.get("last_rotation"),
            "rotations_performed": len(self.data.get("rotation_history", []))
        }