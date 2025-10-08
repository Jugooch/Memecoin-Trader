"""
PnL (Profit and Loss) Calculator for Frontrun Bot
Calculates realized and unrealized PnL with comprehensive fee accounting
"""

from typing import Optional
from dataclasses import dataclass
from datetime import datetime, timezone

from core.logger import get_logger
from core.metrics import get_metrics


logger = get_logger(__name__)
metrics = get_metrics()


# Lamports per SOL for conversions
LAMPORTS_PER_SOL = 1_000_000_000


@dataclass
class TradeFees:
    """All fees associated with a trade"""
    platform_fee_lamports: int  # Pump.fun 1% fee
    priority_fee_lamports: int  # Priority/compute fee
    network_fee_lamports: int = 5000  # Base network fee (signature)

    @property
    def total_lamports(self) -> int:
        """Total fees in lamports"""
        return self.platform_fee_lamports + self.priority_fee_lamports + self.network_fee_lamports

    @property
    def total_sol(self) -> float:
        """Total fees in SOL"""
        return self.total_lamports / LAMPORTS_PER_SOL


@dataclass
class RealizedPnL:
    """Realized profit/loss from a closed position"""
    entry_price_lamports: int  # Entry price in lamports
    exit_price_lamports: int  # Exit price in lamports
    position_size_tokens: int  # Number of tokens
    entry_fees: TradeFees  # Entry trade fees
    exit_fees: TradeFees  # Exit trade fees
    holding_time_seconds: int  # Time held in seconds

    @property
    def gross_pnl_lamports(self) -> int:
        """Gross PnL before fees (can be negative)"""
        return self.exit_price_lamports - self.entry_price_lamports

    @property
    def total_fees_lamports(self) -> int:
        """Total fees paid"""
        return self.entry_fees.total_lamports + self.exit_fees.total_lamports

    @property
    def net_pnl_lamports(self) -> int:
        """Net PnL after all fees (can be negative)"""
        return self.gross_pnl_lamports - self.total_fees_lamports

    @property
    def net_pnl_sol(self) -> float:
        """Net PnL in SOL"""
        return self.net_pnl_lamports / LAMPORTS_PER_SOL

    @property
    def pnl_percentage(self) -> float:
        """PnL as percentage of entry price"""
        if self.entry_price_lamports == 0:
            return 0.0
        return (self.net_pnl_lamports / self.entry_price_lamports) * 100

    @property
    def is_profitable(self) -> bool:
        """Whether position was profitable"""
        return self.net_pnl_lamports > 0

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "entry_price_sol": self.entry_price_lamports / LAMPORTS_PER_SOL,
            "exit_price_sol": self.exit_price_lamports / LAMPORTS_PER_SOL,
            "position_size_tokens": self.position_size_tokens,
            "gross_pnl_sol": self.gross_pnl_lamports / LAMPORTS_PER_SOL,
            "total_fees_sol": self.total_fees_lamports / LAMPORTS_PER_SOL,
            "net_pnl_sol": self.net_pnl_sol,
            "pnl_percentage": self.pnl_percentage,
            "is_profitable": self.is_profitable,
            "holding_time_seconds": self.holding_time_seconds
        }


@dataclass
class UnrealizedPnL:
    """Unrealized profit/loss for an open position"""
    entry_price_lamports: int  # Entry price in lamports
    current_price_lamports: int  # Current market price in lamports
    position_size_tokens: int  # Number of tokens
    entry_fees: TradeFees  # Entry trade fees
    holding_time_seconds: int  # Time held so far

    @property
    def current_value_lamports(self) -> int:
        """Current value of position"""
        return self.current_price_lamports

    @property
    def unrealized_pnl_lamports(self) -> int:
        """Unrealized PnL before exit (including entry fees)"""
        # Current value - entry value - entry fees
        return self.current_price_lamports - self.entry_price_lamports - self.entry_fees.total_lamports

    @property
    def unrealized_pnl_sol(self) -> float:
        """Unrealized PnL in SOL"""
        return self.unrealized_pnl_lamports / LAMPORTS_PER_SOL

    @property
    def pnl_percentage(self) -> float:
        """Unrealized PnL as percentage of entry"""
        if self.entry_price_lamports == 0:
            return 0.0
        return (self.unrealized_pnl_lamports / self.entry_price_lamports) * 100

    @property
    def is_profitable(self) -> bool:
        """Whether position is currently profitable"""
        return self.unrealized_pnl_lamports > 0

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "entry_price_sol": self.entry_price_lamports / LAMPORTS_PER_SOL,
            "current_price_sol": self.current_price_lamports / LAMPORTS_PER_SOL,
            "position_size_tokens": self.position_size_tokens,
            "current_value_sol": self.current_value_lamports / LAMPORTS_PER_SOL,
            "unrealized_pnl_sol": self.unrealized_pnl_sol,
            "pnl_percentage": self.pnl_percentage,
            "is_profitable": self.is_profitable,
            "holding_time_seconds": self.holding_time_seconds
        }


class PnLCalculator:
    """
    Calculates profit and loss for trading positions

    Features:
    - Realized PnL (closed positions)
    - Unrealized PnL (open positions)
    - Comprehensive fee accounting
    - Percentage returns
    - Holding time tracking

    Usage:
        calculator = PnLCalculator()

        # Calculate realized PnL
        pnl = calculator.calculate_realized_pnl(
            entry_price_lamports=1_000_000_000,  # 1 SOL
            exit_price_lamports=1_500_000_000,   # 1.5 SOL
            position_size_tokens=1_000_000,
            entry_fees=entry_fees,
            exit_fees=exit_fees,
            entry_time=entry_time,
            exit_time=exit_time
        )
        print(f"Net PnL: {pnl.net_pnl_sol:.4f} SOL ({pnl.pnl_percentage:.2f}%)")
    """

    def __init__(self):
        """Initialize PnL calculator"""
        logger.info("pnl_calculator_initialized")

    def calculate_realized_pnl(
        self,
        entry_price_lamports: int,
        exit_price_lamports: int,
        position_size_tokens: int,
        entry_fees: TradeFees,
        exit_fees: TradeFees,
        entry_time: datetime,
        exit_time: datetime
    ) -> RealizedPnL:
        """
        Calculate realized PnL for a closed position

        Args:
            entry_price_lamports: Entry price in lamports
            exit_price_lamports: Exit price in lamports
            position_size_tokens: Position size in tokens
            entry_fees: Entry trade fees
            exit_fees: Exit trade fees
            entry_time: Entry timestamp
            exit_time: Exit timestamp

        Returns:
            RealizedPnL with complete PnL breakdown

        Example:
            pnl = calculator.calculate_realized_pnl(
                entry_price_lamports=1_000_000_000,
                exit_price_lamports=1_500_000_000,
                position_size_tokens=1_000_000,
                entry_fees=TradeFees(10_000_000, 500_000),
                exit_fees=TradeFees(15_000_000, 500_000),
                entry_time=datetime.now(),
                exit_time=datetime.now()
            )
        """
        # Validate times
        if exit_time <= entry_time:
            raise ValueError("Exit time must be after entry time")

        # Calculate holding time
        holding_time_seconds = int((exit_time - entry_time).total_seconds())

        pnl = RealizedPnL(
            entry_price_lamports=entry_price_lamports,
            exit_price_lamports=exit_price_lamports,
            position_size_tokens=position_size_tokens,
            entry_fees=entry_fees,
            exit_fees=exit_fees,
            holding_time_seconds=holding_time_seconds
        )

        logger.debug(
            "realized_pnl_calculated",
            net_pnl_sol=pnl.net_pnl_sol,
            pnl_percentage=pnl.pnl_percentage,
            is_profitable=pnl.is_profitable,
            holding_time_seconds=holding_time_seconds
        )

        # Update metrics
        metrics.increment_counter("pnl_calculations", labels={"type": "realized"})
        if pnl.is_profitable:
            metrics.increment_counter("profitable_trades")
        else:
            metrics.increment_counter("unprofitable_trades")

        return pnl

    def calculate_unrealized_pnl(
        self,
        entry_price_lamports: int,
        current_price_lamports: int,
        position_size_tokens: int,
        entry_fees: TradeFees,
        entry_time: datetime
    ) -> UnrealizedPnL:
        """
        Calculate unrealized PnL for an open position

        Args:
            entry_price_lamports: Entry price in lamports
            current_price_lamports: Current market price in lamports
            position_size_tokens: Position size in tokens
            entry_fees: Entry trade fees
            entry_time: Entry timestamp

        Returns:
            UnrealizedPnL with current position status

        Example:
            pnl = calculator.calculate_unrealized_pnl(
                entry_price_lamports=1_000_000_000,
                current_price_lamports=1_200_000_000,
                position_size_tokens=1_000_000,
                entry_fees=TradeFees(10_000_000, 500_000),
                entry_time=datetime.now()
            )
        """
        # Calculate holding time
        holding_time_seconds = int((datetime.now(timezone.utc) - entry_time).total_seconds())

        pnl = UnrealizedPnL(
            entry_price_lamports=entry_price_lamports,
            current_price_lamports=current_price_lamports,
            position_size_tokens=position_size_tokens,
            entry_fees=entry_fees,
            holding_time_seconds=holding_time_seconds
        )

        logger.debug(
            "unrealized_pnl_calculated",
            unrealized_pnl_sol=pnl.unrealized_pnl_sol,
            pnl_percentage=pnl.pnl_percentage,
            is_profitable=pnl.is_profitable,
            holding_time_seconds=holding_time_seconds
        )

        metrics.increment_counter("pnl_calculations", labels={"type": "unrealized"})

        return pnl

    def calculate_breakeven_price(
        self,
        entry_price_lamports: int,
        entry_fees: TradeFees,
        estimated_exit_fees: TradeFees
    ) -> int:
        """
        Calculate breakeven exit price (where net PnL = 0)

        Args:
            entry_price_lamports: Entry price in lamports
            entry_fees: Entry trade fees
            estimated_exit_fees: Estimated exit trade fees

        Returns:
            Breakeven price in lamports

        Example:
            breakeven = calculator.calculate_breakeven_price(
                entry_price_lamports=1_000_000_000,
                entry_fees=TradeFees(10_000_000, 500_000),
                estimated_exit_fees=TradeFees(10_000_000, 500_000)
            )
            print(f"Breakeven: {breakeven / LAMPORTS_PER_SOL:.4f} SOL")
        """
        # Breakeven = entry + total fees
        total_fees = entry_fees.total_lamports + estimated_exit_fees.total_lamports
        breakeven = entry_price_lamports + total_fees

        logger.debug(
            "breakeven_calculated",
            entry_price_sol=entry_price_lamports / LAMPORTS_PER_SOL,
            breakeven_price_sol=breakeven / LAMPORTS_PER_SOL,
            total_fees_sol=total_fees / LAMPORTS_PER_SOL
        )

        return breakeven

    def calculate_required_profit_percentage(
        self,
        entry_fees: TradeFees,
        estimated_exit_fees: TradeFees,
        entry_price_lamports: int
    ) -> float:
        """
        Calculate required profit percentage to break even

        Args:
            entry_fees: Entry trade fees
            estimated_exit_fees: Estimated exit trade fees
            entry_price_lamports: Entry price in lamports

        Returns:
            Required profit percentage to break even

        Example:
            required = calculator.calculate_required_profit_percentage(
                entry_fees=TradeFees(10_000_000, 500_000),
                estimated_exit_fees=TradeFees(10_000_000, 500_000),
                entry_price_lamports=1_000_000_000
            )
            print(f"Need {required:.2f}% profit to break even")
        """
        if entry_price_lamports == 0:
            return 0.0

        total_fees = entry_fees.total_lamports + estimated_exit_fees.total_lamports
        required_pct = (total_fees / entry_price_lamports) * 100

        logger.debug(
            "required_profit_calculated",
            required_percentage=required_pct,
            total_fees_sol=total_fees / LAMPORTS_PER_SOL
        )

        return required_pct

    def calculate_roi(
        self,
        net_pnl_lamports: int,
        entry_price_lamports: int
    ) -> float:
        """
        Calculate Return on Investment (ROI)

        Args:
            net_pnl_lamports: Net profit/loss in lamports
            entry_price_lamports: Initial investment in lamports

        Returns:
            ROI as percentage

        Example:
            roi = calculator.calculate_roi(
                net_pnl_lamports=500_000_000,  # 0.5 SOL profit
                entry_price_lamports=1_000_000_000  # 1 SOL investment
            )
            # Returns 50.0 (50% ROI)
        """
        if entry_price_lamports == 0:
            return 0.0

        roi = (net_pnl_lamports / entry_price_lamports) * 100

        return roi

    def estimate_exit_fees(
        self,
        exit_price_lamports: int,
        platform_fee_bps: int = 100,
        estimated_priority_fee_lamports: int = 500_000
    ) -> TradeFees:
        """
        Estimate exit trade fees

        Args:
            exit_price_lamports: Expected exit price in lamports
            platform_fee_bps: Platform fee in basis points (default 100 = 1%)
            estimated_priority_fee_lamports: Estimated priority fee

        Returns:
            TradeFees estimate for exit

        Example:
            exit_fees = calculator.estimate_exit_fees(
                exit_price_lamports=1_500_000_000,
                platform_fee_bps=100,
                estimated_priority_fee_lamports=500_000
            )
        """
        # Calculate platform fee
        platform_fee = (exit_price_lamports * platform_fee_bps) // 10_000

        fees = TradeFees(
            platform_fee_lamports=platform_fee,
            priority_fee_lamports=estimated_priority_fee_lamports
        )

        return fees


# Example usage
if __name__ == "__main__":
    from core.logger import setup_logging

    # Setup logging
    setup_logging(level="DEBUG", format="console")

    # Create calculator
    calculator = PnLCalculator()

    # Example: Profitable trade
    print("=== PROFITABLE TRADE ===")
    entry_fees = TradeFees(
        platform_fee_lamports=10_000_000,  # 0.01 SOL
        priority_fee_lamports=500_000  # 0.0005 SOL
    )
    exit_fees = TradeFees(
        platform_fee_lamports=15_000_000,  # 0.015 SOL
        priority_fee_lamports=500_000  # 0.0005 SOL
    )

    pnl = calculator.calculate_realized_pnl(
        entry_price_lamports=1_000_000_000,  # 1 SOL
        exit_price_lamports=1_500_000_000,  # 1.5 SOL
        position_size_tokens=1_000_000,
        entry_fees=entry_fees,
        exit_fees=exit_fees,
        entry_time=datetime.now(timezone.utc),
        exit_time=datetime.now(timezone.utc)
    )

    print(f"Entry: {pnl.entry_price_lamports / LAMPORTS_PER_SOL:.4f} SOL")
    print(f"Exit: {pnl.exit_price_lamports / LAMPORTS_PER_SOL:.4f} SOL")
    print(f"Gross PnL: {pnl.gross_pnl_lamports / LAMPORTS_PER_SOL:.4f} SOL")
    print(f"Total Fees: {pnl.total_fees_lamports / LAMPORTS_PER_SOL:.4f} SOL")
    print(f"Net PnL: {pnl.net_pnl_sol:.4f} SOL ({pnl.pnl_percentage:.2f}%)")
    print(f"Profitable: {pnl.is_profitable}")

    # Example: Losing trade
    print("\n=== LOSING TRADE ===")
    pnl2 = calculator.calculate_realized_pnl(
        entry_price_lamports=1_000_000_000,  # 1 SOL
        exit_price_lamports=800_000_000,  # 0.8 SOL
        position_size_tokens=1_000_000,
        entry_fees=entry_fees,
        exit_fees=TradeFees(8_000_000, 500_000),
        entry_time=datetime.now(timezone.utc),
        exit_time=datetime.now(timezone.utc)
    )

    print(f"Net PnL: {pnl2.net_pnl_sol:.4f} SOL ({pnl2.pnl_percentage:.2f}%)")
    print(f"Profitable: {pnl2.is_profitable}")

    # Breakeven calculation
    print("\n=== BREAKEVEN ===")
    breakeven = calculator.calculate_breakeven_price(
        entry_price_lamports=1_000_000_000,
        entry_fees=entry_fees,
        estimated_exit_fees=exit_fees
    )
    print(f"Entry: 1.0000 SOL")
    print(f"Breakeven: {breakeven / LAMPORTS_PER_SOL:.4f} SOL")

    required_profit = calculator.calculate_required_profit_percentage(
        entry_fees=entry_fees,
        estimated_exit_fees=exit_fees,
        entry_price_lamports=1_000_000_000
    )
    print(f"Required profit: {required_profit:.2f}%")
