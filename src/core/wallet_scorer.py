"""
Bayesian Wallet Scorer for intelligent wallet performance tracking
Uses Beta distribution with prior to handle uncertainty in wallet performance
"""

import time
import logging
from collections import deque
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class WalletStats:
    """Statistics for a single wallet"""
    wins: int = 0
    losses: int = 0
    recent_trades: deque = field(default_factory=lambda: deque(maxlen=100))
    last_activity: float = 0
    last_trade_size: float = 0
    total_volume: float = 0
    trades_count: int = 0
    

class WalletScorer:
    """
    Bayesian wallet scoring using Beta distribution
    Prior: Beta(3,5) represents ~37.5% baseline win rate
    """
    
    def __init__(self, prior_alpha: float = 3, prior_beta: float = 5):
        """
        Initialize Bayesian Wallet Scorer
        
        Args:
            prior_alpha: Alpha parameter for Beta prior (successes + 1)
            prior_beta: Beta parameter for Beta prior (failures + 1)
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.wallet_stats: Dict[str, WalletStats] = {}
        self.window_seconds = 3600  # 60-minute rolling window for recency
        self.logger = logging.getLogger(__name__)
        
    def update_wallet_outcome(self, wallet: str, win: bool, timestamp: Optional[float] = None, 
                             trade_size: Optional[float] = None) -> None:
        """
        Update wallet with a new trade outcome
        
        Args:
            wallet: Wallet address
            win: True if trade was profitable, False otherwise
            timestamp: Unix timestamp of the trade
            trade_size: Size of the trade in USD (optional)
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Initialize wallet if not seen before
        if wallet not in self.wallet_stats:
            self.wallet_stats[wallet] = WalletStats()
            
        stats = self.wallet_stats[wallet]
        
        # Update win/loss counts
        if win:
            stats.wins += 1
        else:
            stats.losses += 1
            
        # Update recent activity
        stats.last_activity = timestamp
        stats.trades_count += 1
        
        # Track trade size for size factor calculation
        if trade_size:
            stats.last_trade_size = trade_size
            stats.total_volume += trade_size
            
        # Add to recent trades (timestamp, outcome, size)
        stats.recent_trades.append({
            'timestamp': timestamp,
            'win': win,
            'size': trade_size or 0
        })
        
        # Prune old trades from the deque (automatic with maxlen)
        self.logger.debug(f"Updated wallet {wallet[:8]}...: wins={stats.wins}, losses={stats.losses}")
        
    def get_wallet_score(self, wallet: str) -> float:
        """
        Calculate Bayesian posterior mean for wallet win rate
        
        Returns:
            Posterior mean win rate (0.0 to 1.0)
        """
        if wallet not in self.wallet_stats:
            # Return prior mean for unknown wallets
            return self.prior_alpha / (self.prior_alpha + self.prior_beta)
            
        stats = self.wallet_stats[wallet]
        
        # Calculate posterior parameters
        posterior_alpha = self.prior_alpha + stats.wins
        posterior_beta = self.prior_beta + stats.losses
        
        # Posterior mean
        base_score = posterior_alpha / (posterior_alpha + posterior_beta)
        
        return base_score
        
    def get_recency_boost(self, wallet: str) -> float:
        """
        Calculate recency boost based on last activity
        
        Returns:
            Boost multiplier (1.0 to 1.3)
        """
        if wallet not in self.wallet_stats:
            return 1.0
            
        stats = self.wallet_stats[wallet]
        current_time = time.time()
        time_since_activity = current_time - stats.last_activity
        
        # Exponential decay recency boost
        if time_since_activity < 300:  # < 5 minutes
            return 1.3
        elif time_since_activity < 600:  # < 10 minutes
            return 1.25
        elif time_since_activity < 1200:  # < 20 minutes
            return 1.2
        elif time_since_activity < 1800:  # < 30 minutes
            return 1.15
        elif time_since_activity < 3600:  # < 60 minutes
            return 1.1
        elif time_since_activity < 7200:  # < 2 hours
            return 1.05
        else:
            return 1.0
            
    def get_size_factor(self, wallet: str) -> float:
        """
        Calculate size factor based on average trade size
        
        Returns:
            Size multiplier (1.0 to 1.5)
        """
        if wallet not in self.wallet_stats:
            return 1.0
            
        stats = self.wallet_stats[wallet]
        
        # No trades or no volume tracked
        if stats.trades_count == 0 or stats.total_volume == 0:
            return 1.0
            
        avg_trade_size = stats.total_volume / stats.trades_count
        
        # Size-based multiplier (assuming avg trade ~$100)
        if avg_trade_size >= 1000:  # Large trades
            return 1.5
        elif avg_trade_size >= 500:
            return 1.4
        elif avg_trade_size >= 250:
            return 1.3
        elif avg_trade_size >= 100:
            return 1.2
        elif avg_trade_size >= 50:
            return 1.1
        else:
            return 1.0
            
    def get_weighted_score(self, wallet: str) -> float:
        """
        Get final weighted score combining base score, recency, and size
        
        Returns:
            Weighted score (0.0 to ~2.0)
        """
        base_score = self.get_wallet_score(wallet)
        recency_boost = self.get_recency_boost(wallet)
        size_factor = self.get_size_factor(wallet)
        
        weighted_score = base_score * recency_boost * size_factor
        
        self.logger.debug(f"Wallet {wallet[:8]}... score: base={base_score:.3f}, "
                         f"recency={recency_boost:.2f}, size={size_factor:.2f}, "
                         f"weighted={weighted_score:.3f}")
        
        return weighted_score
        
    def get_smart_confidence_score(self, wallet: str) -> float:
        """
        Smart confidence scoring that handles fresh wallets appropriately
        
        Fresh wallets from discovery get benefit of the doubt since they
        passed rigorous discovery filters. Developing wallets get a floor
        to prevent cold start issues. Established wallets use full Bayesian.
        
        Returns:
            Confidence score (0.0 to 1.0)
        """
        if wallet not in self.wallet_stats:
            # Fresh wallet from discovery = trust the discovery process
            self.logger.debug(f"Fresh wallet {wallet[:8]}... gets default confidence: 0.65")
            return 0.65  # 65% confidence (above 55% threshold)
            
        stats = self.wallet_stats[wallet]
        trade_count = stats.wins + stats.losses
        
        if trade_count <= 3:
            # Developing wallet - use optimistic floor with Bayesian score
            bayesian_score = self.get_wallet_score(wallet)
            confidence_score = max(bayesian_score, 0.55)  # Floor at 55%
            self.logger.debug(f"Developing wallet {wallet[:8]}... (trades={trade_count}): "
                            f"bayesian={bayesian_score:.3f}, confidence={confidence_score:.3f}")
            return confidence_score
        else:
            # Established wallet - full Bayesian scoring
            confidence_score = self.get_wallet_score(wallet)
            self.logger.debug(f"Established wallet {wallet[:8]}... (trades={trade_count}): "
                            f"confidence={confidence_score:.3f}")
            return confidence_score
            
    def get_wallet_trade_count(self, wallet: str) -> int:
        """
        Get total number of trades for a wallet
        
        Returns:
            Total trade count
        """
        if wallet not in self.wallet_stats:
            return 0
        return self.wallet_stats[wallet].wins + self.wallet_stats[wallet].losses
        
    def get_wallet_confidence_interval(self, wallet: str, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for wallet win rate
        
        Args:
            wallet: Wallet address
            confidence: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if wallet not in self.wallet_stats:
            # Return prior confidence interval
            # For Beta(3,5), approximate 95% CI is roughly (0.15, 0.65)
            return (0.15, 0.65)
            
        stats = self.wallet_stats[wallet]
        
        # Calculate posterior parameters
        posterior_alpha = self.prior_alpha + stats.wins
        posterior_beta = self.prior_beta + stats.losses
        
        # For Beta distribution, we can use the quantile function
        # Approximation using Wilson score interval
        n = stats.wins + stats.losses
        if n == 0:
            return (0.15, 0.65)
            
        p_hat = stats.wins / n
        z = 1.96  # 95% confidence
        
        denominator = 1 + z**2 / n
        center = (p_hat + z**2 / (2*n)) / denominator
        margin = z * ((p_hat * (1 - p_hat) / n + z**2 / (4*n**2)) ** 0.5) / denominator
        
        lower = max(0, center - margin)
        upper = min(1, center + margin)
        
        return (lower, upper)
        
    def prune_old_trades(self, max_age_seconds: float = 86400) -> None:
        """
        Remove trades older than max_age from all wallets
        
        Args:
            max_age_seconds: Maximum age in seconds (default 24 hours)
        """
        current_time = time.time()
        cutoff_time = current_time - max_age_seconds
        
        for wallet, stats in self.wallet_stats.items():
            # Filter recent trades to keep only those within the window
            stats.recent_trades = deque(
                [t for t in stats.recent_trades if t['timestamp'] > cutoff_time],
                maxlen=100
            )
            
    def get_stats_summary(self) -> Dict:
        """
        Get summary statistics for all tracked wallets
        
        Returns:
            Dictionary with summary stats
        """
        if not self.wallet_stats:
            return {
                'total_wallets': 0,
                'avg_win_rate': 0,
                'total_trades': 0,
                'active_wallets': 0
            }
            
        total_wins = sum(s.wins for s in self.wallet_stats.values())
        total_losses = sum(s.losses for s in self.wallet_stats.values())
        total_trades = total_wins + total_losses
        
        current_time = time.time()
        active_wallets = sum(
            1 for s in self.wallet_stats.values() 
            if current_time - s.last_activity < 3600
        )
        
        avg_win_rate = total_wins / total_trades if total_trades > 0 else 0
        
        return {
            'total_wallets': len(self.wallet_stats),
            'avg_win_rate': avg_win_rate,
            'total_trades': total_trades,
            'active_wallets': active_wallets,
            'total_wins': total_wins,
            'total_losses': total_losses
        }
        
    def reset_wallet(self, wallet: str) -> None:
        """Reset statistics for a specific wallet"""
        if wallet in self.wallet_stats:
            del self.wallet_stats[wallet]
            self.logger.info(f"Reset statistics for wallet {wallet[:8]}...")
            
    def reset_all(self) -> None:
        """Reset all wallet statistics"""
        self.wallet_stats.clear()
        self.logger.info("Reset all wallet statistics")