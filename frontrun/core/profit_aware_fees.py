"""
Profit-Aware Fee Cap for Frontrun Bot
Calculates conservative fee caps based on expected value to prevent fee burn
"""

import time
import statistics
from typing import List, Optional, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta

from core.priority_fees import PriorityFeeCalculator, FeeUrgency
from core.logger import get_logger
from core.metrics import get_metrics


logger = get_logger(__name__)
metrics = get_metrics()


LAMPORTS_PER_SOL = 1_000_000_000


@dataclass
class Trade:
    """
    Trade result for EV calculation

    Attributes:
        won_race: Whether we won the frontrun race
        pnl_sol: Profit/loss in SOL (can be negative)
        fee_paid_lamports: Priority fee paid
        timestamp: When trade completed
    """
    won_race: bool
    pnl_sol: float
    fee_paid_lamports: int
    timestamp: datetime

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "won_race": self.won_race,
            "pnl_sol": self.pnl_sol,
            "fee_paid_lamports": self.fee_paid_lamports,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class BidderConfig:
    """
    Configuration for profit-aware fee bidding

    Fee strategy:
    - base_multiplier: Multiply average network fee by this (default: 10.0)
    - competition_multiplier: Multiply competing tx fee by this (default: 1.5)
    - max_fee_lamports: Absolute maximum fee cap (default: 10M = 0.01 SOL)
    - min_fee_lamports: Minimum fee floor (default: 100k = 0.0001 SOL)

    Profit-aware caps:
    - cold_start_max_lamports: Conservative cap before we have data (default: 50k)
    - profit_cap_pct: Never bid >X% of conservative profit (default: 30%)
    - min_samples_for_ev: Need N+ trades before using EV caps (default: 10)

    Trade history:
    - max_trade_history: Keep last N trades for EV calculation (default: 100)
    - trade_history_ttl_hours: Only consider trades from last N hours (default: 24)
    """
    # Base fee strategy
    base_multiplier: float = 10.0
    competition_multiplier: float = 1.5
    max_fee_lamports: int = 10_000_000  # 0.01 SOL
    min_fee_lamports: int = 100_000  # 0.0001 SOL

    # Profit-aware caps
    cold_start_max_lamports: int = 50_000  # ~$0.005
    profit_cap_pct: float = 0.3  # 30%
    min_samples_for_ev: int = 10

    # Trade history management
    max_trade_history: int = 100
    trade_history_ttl_hours: int = 24


class ProfitAwareFeeBidder:
    """
    Calculates aggressive fees with profit-aware caps

    Features:
    - Dynamic fee calculation (10x avg or 1.5x competition)
    - Conservative profit estimation (p25 of wins, not mean)
    - Cold-start protection (max 50k lamports until 10+ trades)
    - Per-compute-unit price limits
    - Pre-submit EV validation

    Usage:
        bidder = ProfitAwareFeeBidder(fee_calculator, config)

        # Calculate fee for trade
        fee = await bidder.calculate_aggressive_fee(
            competing_tx=None,
            compute_units=200_000
        )

        # Add trade result
        bidder.add_trade(Trade(
            won_race=True,
            pnl_sol=0.05,
            fee_paid_lamports=100_000,
            timestamp=datetime.now()
        ))
    """

    def __init__(
        self,
        fee_calculator: PriorityFeeCalculator,
        config: Optional[BidderConfig] = None
    ):
        """
        Initialize profit-aware fee bidder

        Args:
            fee_calculator: Base priority fee calculator
            config: Bidder configuration (optional)
        """
        self.fee_calculator = fee_calculator
        self.config = config or BidderConfig()

        # Trade history for EV calculation
        self._trade_history: List[Trade] = []

        logger.info(
            "profit_aware_bidder_initialized",
            base_multiplier=self.config.base_multiplier,
            competition_multiplier=self.config.competition_multiplier,
            profit_cap_pct=self.config.profit_cap_pct,
            cold_start_max=self.config.cold_start_max_lamports,
            min_samples=self.config.min_samples_for_ev
        )

    async def calculate_aggressive_fee(
        self,
        competing_tx: Optional[Dict] = None,
        compute_units: int = 200_000
    ) -> int:
        """
        Calculate aggressive fee with profit-aware cap

        Args:
            competing_tx: Competing transaction (if known) with 'fee' field
            compute_units: Compute units for transaction

        Returns:
            Priority fee in micro-lamports per compute unit

        Example:
            # No competition - use aggressive base
            fee = await bidder.calculate_aggressive_fee(
                competing_tx=None,
                compute_units=200_000
            )

            # With competition - outbid
            fee = await bidder.calculate_aggressive_fee(
                competing_tx={"fee": 500_000},
                compute_units=200_000
            )
        """
        # Get base network fee
        base_fee = await self.fee_calculator.calculate_priority_fee(
            compute_units=compute_units,
            urgency=FeeUrgency.NORMAL
        )

        # Calculate aggressive fee
        if competing_tx and "fee" in competing_tx:
            # Outbid competition
            competing_fee_lamports = competing_tx["fee"]
            # Convert to micro-lamports per CU
            competing_fee_per_cu = (competing_fee_lamports * 1_000_000) // compute_units
            aggressive_fee = int(competing_fee_per_cu * self.config.competition_multiplier)
        else:
            # Use aggressive base
            aggressive_fee = int(base_fee * self.config.base_multiplier)

        # Get profit-aware cap
        recent_trades = self._get_recent_trades()
        profit_cap = self._calculate_profit_cap(recent_trades, compute_units)

        # Apply all caps
        final_fee = max(
            self.config.min_fee_lamports,
            min(aggressive_fee, profit_cap, self.config.max_fee_lamports)
        )

        # Convert to total lamports for logging
        total_lamports = (final_fee * compute_units) // 1_000_000

        logger.info(
            "aggressive_fee_calculated",
            base_fee=base_fee,
            aggressive_fee=aggressive_fee,
            profit_cap=profit_cap,
            final_fee=final_fee,
            total_lamports=total_lamports,
            compute_units=compute_units,
            capped_by_profit=aggressive_fee > profit_cap
        )

        metrics.increment_counter(
            "profit_aware_fees_calculated",
            labels={
                "capped_by_profit": str(aggressive_fee > profit_cap)
            }
        )

        return final_fee

    def _calculate_profit_cap(
        self,
        recent_trades: List[Trade],
        compute_units: int
    ) -> int:
        """
        Calculate fee cap based on conservative profit estimate

        Args:
            recent_trades: Recent trade history
            compute_units: Compute units for this trade

        Returns:
            Fee cap in micro-lamports per compute unit

        Strategy:
        - If <10 trades: Use cold_start_max_lamports
        - If >=10 trades: Use 30% of p25 of winning trades
        """
        # Cold start: not enough data
        if len(recent_trades) < self.config.min_samples_for_ev:
            logger.debug(
                "using_cold_start_cap",
                trade_count=len(recent_trades),
                min_samples=self.config.min_samples_for_ev,
                cap_lamports=self.config.cold_start_max_lamports
            )
            # Convert to micro-lamports per CU
            return (self.config.cold_start_max_lamports * 1_000_000) // compute_units

        # Calculate conservative EV
        conservative_ev_sol = self._estimate_conservative_ev(recent_trades)

        if conservative_ev_sol <= 0:
            # No positive expected value - use minimum cap
            logger.warning(
                "negative_expected_value",
                conservative_ev_sol=conservative_ev_sol,
                using_cold_start_cap=True
            )
            return (self.config.cold_start_max_lamports * 1_000_000) // compute_units

        # Calculate profit cap (30% of conservative EV)
        profit_cap_lamports = int(
            conservative_ev_sol * LAMPORTS_PER_SOL * self.config.profit_cap_pct
        )

        logger.debug(
            "profit_cap_calculated",
            conservative_ev_sol=conservative_ev_sol,
            profit_cap_lamports=profit_cap_lamports,
            profit_cap_pct=self.config.profit_cap_pct
        )

        # Convert to micro-lamports per CU
        return (profit_cap_lamports * 1_000_000) // compute_units

    def _estimate_conservative_ev(self, trades: List[Trade]) -> float:
        """
        Estimate conservative expected value using p25 of wins

        Args:
            trades: Recent trade history

        Returns:
            Conservative EV in SOL (p25 of winning trades)

        Strategy:
        - Filter to winning trades only
        - Take 25th percentile (conservative, not mean)
        - This protects against over-optimistic EV estimates
        """
        if not trades:
            return 0.0

        # Get winning trades only
        winning_trades = [t for t in trades if t.won_race and t.pnl_sol > 0]

        if not winning_trades:
            logger.warning("no_winning_trades_in_history")
            return 0.0

        # Calculate p25 (25th percentile) of wins
        profits = [t.pnl_sol for t in winning_trades]
        profits.sort()

        # Calculate percentile
        p25_index = int(len(profits) * 0.25)
        p25_profit = profits[p25_index] if p25_index < len(profits) else profits[0]

        logger.debug(
            "conservative_ev_estimated",
            winning_trades=len(winning_trades),
            total_trades=len(trades),
            p25_profit_sol=p25_profit,
            win_rate=len(winning_trades) / len(trades)
        )

        return max(0.0, p25_profit)

    def _get_recent_trades(self) -> List[Trade]:
        """
        Get recent trades within TTL window

        Returns:
            List of trades from last trade_history_ttl_hours
        """
        cutoff = datetime.now() - timedelta(hours=self.config.trade_history_ttl_hours)
        recent = [t for t in self._trade_history if t.timestamp >= cutoff]

        # Update metrics
        if recent:
            wins = sum(1 for t in recent if t.won_race)
            win_rate = wins / len(recent) if recent else 0.0
            metrics.set_gauge("trade_win_rate", win_rate)
            metrics.set_gauge("trade_history_size", len(recent))

        return recent

    def add_trade(self, trade: Trade):
        """
        Add trade result to history

        Args:
            trade: Trade result

        Example:
            bidder.add_trade(Trade(
                won_race=True,
                pnl_sol=0.05,
                fee_paid_lamports=100_000,
                timestamp=datetime.now()
            ))
        """
        self._trade_history.append(trade)

        # Trim history to max size
        if len(self._trade_history) > self.config.max_trade_history:
            self._trade_history.pop(0)

        logger.debug(
            "trade_added_to_history",
            won_race=trade.won_race,
            pnl_sol=trade.pnl_sol,
            fee_paid_lamports=trade.fee_paid_lamports,
            history_size=len(self._trade_history)
        )

        metrics.increment_counter(
            "trades_recorded",
            labels={
                "won_race": str(trade.won_race),
                "profitable": str(trade.pnl_sol > 0)
            }
        )

    def validate_ev_before_submit(
        self,
        expected_profit_sol: float,
        fee_lamports: int
    ) -> bool:
        """
        Validate that EV > 0 before submitting trade

        Args:
            expected_profit_sol: Expected profit from this trade
            fee_lamports: Fee we're about to pay

        Returns:
            True if EV > 0 (safe to submit), False otherwise

        Example:
            # Before submitting
            if not bidder.validate_ev_before_submit(
                expected_profit_sol=0.03,
                fee_lamports=500_000
            ):
                logger.warning("Abort - negative EV after price movement")
                return
        """
        fee_sol = fee_lamports / LAMPORTS_PER_SOL
        net_ev = expected_profit_sol - fee_sol

        if net_ev <= 0:
            logger.warning(
                "negative_ev_detected",
                expected_profit_sol=expected_profit_sol,
                fee_sol=fee_sol,
                net_ev=net_ev
            )
            metrics.increment_counter("trades_aborted_negative_ev")
            return False

        logger.debug(
            "ev_validation_passed",
            expected_profit_sol=expected_profit_sol,
            fee_sol=fee_sol,
            net_ev=net_ev
        )

        return True

    def get_stats(self) -> Dict:
        """
        Get bidder statistics

        Returns:
            Dictionary with statistics

        Example:
            stats = bidder.get_stats()
            print(f"Win rate: {stats['win_rate']:.1%}")
            print(f"Conservative EV: {stats['conservative_ev_sol']} SOL")
        """
        recent_trades = self._get_recent_trades()

        if not recent_trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "conservative_ev_sol": 0.0,
                "avg_fee_paid_sol": 0.0,
                "avg_profit_sol": 0.0,
                "using_cold_start": True
            }

        wins = [t for t in recent_trades if t.won_race]
        win_rate = len(wins) / len(recent_trades)

        conservative_ev = self._estimate_conservative_ev(recent_trades)

        avg_fee_paid = statistics.mean(t.fee_paid_lamports for t in recent_trades) / LAMPORTS_PER_SOL
        avg_profit = statistics.mean(t.pnl_sol for t in recent_trades)

        return {
            "total_trades": len(recent_trades),
            "winning_trades": len(wins),
            "win_rate": win_rate,
            "conservative_ev_sol": conservative_ev,
            "avg_fee_paid_sol": avg_fee_paid,
            "avg_profit_sol": avg_profit,
            "using_cold_start": len(recent_trades) < self.config.min_samples_for_ev
        }


# Example usage
if __name__ == "__main__":
    from core.logger import setup_logging
    from core.rpc_manager import RPCManager
    from core.config import ConfigurationManager
    import asyncio

    async def main():
        # Setup logging
        setup_logging(level="INFO", format="console")

        # Load config
        config_mgr = ConfigurationManager("config/config.yml")
        bot_config = config_mgr.load_config()

        # Create RPC manager
        rpc_manager = RPCManager(bot_config.rpc_config)
        await rpc_manager.start()

        try:
            # Create fee calculator
            fee_calc = PriorityFeeCalculator(rpc_manager)

            # Create profit-aware bidder
            bidder_config = BidderConfig(
                cold_start_max_lamports=50_000,
                profit_cap_pct=0.3,
                min_samples_for_ev=10
            )
            bidder = ProfitAwareFeeBidder(fee_calc, bidder_config)

            # Simulate cold start (no trades)
            logger.info("=== Cold Start Test ===")
            fee = await bidder.calculate_aggressive_fee(
                competing_tx=None,
                compute_units=200_000
            )
            logger.info(f"Cold start fee: {fee} micro-lamports/CU")

            # Add some winning trades
            logger.info("\n=== Adding Trade History ===")
            for i in range(15):
                bidder.add_trade(Trade(
                    won_race=True,
                    pnl_sol=0.02 + (i * 0.01),  # Varying profits
                    fee_paid_lamports=100_000,
                    timestamp=datetime.now()
                ))

            # Now with trade history
            logger.info("\n=== With Trade History ===")
            fee = await bidder.calculate_aggressive_fee(
                competing_tx=None,
                compute_units=200_000
            )
            logger.info(f"Fee with history: {fee} micro-lamports/CU")

            # Get stats
            stats = bidder.get_stats()
            logger.info("bidder_stats", **stats)

            # Test EV validation
            logger.info("\n=== EV Validation Test ===")
            valid = bidder.validate_ev_before_submit(
                expected_profit_sol=0.03,
                fee_lamports=500_000
            )
            logger.info(f"EV validation (0.03 SOL profit, 500k fee): {valid}")

            valid = bidder.validate_ev_before_submit(
                expected_profit_sol=0.0001,
                fee_lamports=500_000
            )
            logger.info(f"EV validation (0.0001 SOL profit, 500k fee): {valid}")

        finally:
            await rpc_manager.stop()

    asyncio.run(main())
