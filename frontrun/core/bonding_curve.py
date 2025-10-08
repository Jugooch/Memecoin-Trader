"""
Bonding Curve Calculator for Pump.fun
Calculates prices, slippage, and impact from bonding curve state using exact on-chain math
"""

from typing import Optional
from dataclasses import dataclass

from core.logger import get_logger
from core.metrics import get_metrics


logger = get_logger(__name__)
metrics = get_metrics()


# Pump.fun constants
FEE_BPS = 100  # 1% fee (100 basis points)
BPS_DENOMINATOR = 10_000


@dataclass
class BondingCurveState:
    """Bonding curve state from on-chain account

    All values in base units:
    - Token values: base token units (typically 6 decimals for pump.fun)
    - SOL values: lamports (9 decimals)
    """
    virtual_token_reserves: int  # Virtual token reserves (base units)
    virtual_sol_reserves: int  # Virtual SOL reserves (lamports)
    real_token_reserves: int  # Real token reserves (base units)
    real_sol_reserves: int  # Real SOL reserves (lamports)
    token_total_supply: int  # Total token supply (base units)
    complete: bool  # Whether curve is complete (migrated to Raydium)


@dataclass
class BuyQuote:
    """Quote for buying tokens

    All values in base units:
    - tokens_out: base token units
    - sol_in: lamports
    - price_per_token_sol: lamports per base token unit
    """
    tokens_out: int  # Tokens received (base units, after fees)
    sol_in: int  # SOL to spend (lamports)
    price_per_token_sol: float  # Effective price (lamports per base token unit)
    price_impact_pct: float  # Price impact percentage
    fee_lamports: int  # Fee amount in lamports


@dataclass
class SellQuote:
    """Quote for selling tokens

    All values in base units:
    - sol_out: lamports
    - tokens_in: base token units
    - price_per_token_sol: lamports per base token unit
    """
    sol_out: int  # SOL received (lamports, after fees)
    tokens_in: int  # Tokens to sell (base units)
    price_per_token_sol: float  # Effective price (lamports per base token unit)
    price_impact_pct: float  # Price impact percentage
    fee_lamports: int  # Fee amount in lamports


class BondingCurveCalculator:
    """
    Calculates prices and slippage from pump.fun bonding curve state

    Uses constant product AMM formula (x * y = k) with exact integer math
    to match Solana on-chain calculations.

    Formula:
        tokens_out = (token_reserves * sol_in) / (sol_reserves + sol_in)
        sol_out = (sol_reserves * tokens_in) / (token_reserves + tokens_in)

    Fees:
        - 1% fee applied to input amount before calculation
        - Fee goes to pump.fun platform

    Usage:
        calculator = BondingCurveCalculator()
        quote = calculator.calculate_buy_price(curve_state, 1_000_000_000)  # 1 SOL
        print(f"Tokens out: {quote.tokens_out}")
        print(f"Price impact: {quote.price_impact_pct:.2f}%")
    """

    def __init__(self):
        """Initialize bonding curve calculator"""
        logger.info(
            "bonding_curve_calculator_initialized",
            fee_bps=FEE_BPS
        )

    def calculate_buy_price(
        self,
        curve_state: BondingCurveState,
        amount_sol_lamports: int
    ) -> BuyQuote:
        """
        Calculate tokens received for buying with SOL

        Args:
            curve_state: Current bonding curve state
            amount_sol_lamports: Amount of SOL to spend (in lamports)

        Returns:
            BuyQuote with tokens out and price info

        Raises:
            ValueError: If reserves are zero or invalid

        Example:
            quote = calculator.calculate_buy_price(curve_state, 1_000_000_000)  # 1 SOL
        """
        if curve_state.virtual_sol_reserves <= 0 or curve_state.virtual_token_reserves <= 0:
            raise ValueError("Invalid bonding curve reserves (must be > 0)")

        if amount_sol_lamports <= 0:
            raise ValueError("Amount must be positive")

        # Calculate fee (1%)
        fee_lamports = (amount_sol_lamports * FEE_BPS) // BPS_DENOMINATOR
        amount_after_fee = amount_sol_lamports - fee_lamports

        # Constant product formula: tokens_out = (token_reserves * sol_in) / (sol_reserves + sol_in)
        # Using integer math to match on-chain exactly
        tokens_out = (
            curve_state.virtual_token_reserves * amount_after_fee
        ) // (
            curve_state.virtual_sol_reserves + amount_after_fee
        )

        # Calculate effective price per token (lamports per base token unit)
        if tokens_out > 0:
            price_per_token = amount_sol_lamports / tokens_out
        else:
            price_per_token = 0.0

        # Calculate price impact
        price_impact_pct = self._calculate_price_impact(
            curve_state.virtual_sol_reserves,
            amount_after_fee
        )

        logger.debug(
            "buy_quote_calculated",
            sol_in=amount_sol_lamports,
            tokens_out=tokens_out,
            price_per_token=price_per_token,
            price_impact_pct=price_impact_pct,
            fee_lamports=fee_lamports
        )

        metrics.increment_counter("bonding_curve_buy_quotes")

        return BuyQuote(
            tokens_out=tokens_out,
            sol_in=amount_sol_lamports,
            price_per_token_sol=price_per_token,
            price_impact_pct=price_impact_pct,
            fee_lamports=fee_lamports
        )

    def calculate_sell_price(
        self,
        curve_state: BondingCurveState,
        amount_tokens: int
    ) -> SellQuote:
        """
        Calculate SOL received for selling tokens

        Args:
            curve_state: Current bonding curve state
            amount_tokens: Amount of tokens to sell

        Returns:
            SellQuote with SOL out and price info

        Raises:
            ValueError: If reserves are zero or invalid

        Example:
            quote = calculator.calculate_sell_price(curve_state, 1_000_000)  # 1M tokens
        """
        if curve_state.virtual_sol_reserves <= 0 or curve_state.virtual_token_reserves <= 0:
            raise ValueError("Invalid bonding curve reserves (must be > 0)")

        if amount_tokens <= 0:
            raise ValueError("Amount must be positive")

        # Calculate fee (1%) - taken from token amount BEFORE swap
        fee_tokens = (amount_tokens * FEE_BPS) // BPS_DENOMINATOR
        amount_after_fee = amount_tokens - fee_tokens

        # Constant product formula: sol_out = (sol_reserves * tokens_in) / (token_reserves + tokens_in)
        # Using integer math to match on-chain exactly
        sol_out = (
            curve_state.virtual_sol_reserves * amount_after_fee
        ) // (
            curve_state.virtual_token_reserves + amount_after_fee
        )

        # Calculate fee in SOL terms (what we would have gotten without fee - what we got)
        # This is the SOL value of the tokens taken as fee
        sol_without_fee = (
            curve_state.virtual_sol_reserves * amount_tokens
        ) // (
            curve_state.virtual_token_reserves + amount_tokens
        )
        fee_lamports = sol_without_fee - sol_out

        # Calculate effective price per token (lamports per base token unit)
        if amount_tokens > 0:
            price_per_token = sol_out / amount_tokens
        else:
            price_per_token = 0.0

        # Calculate price impact (use amount_after_fee since that's what actually gets swapped)
        price_impact_pct = self._calculate_price_impact(
            curve_state.virtual_token_reserves,
            amount_after_fee
        )

        logger.debug(
            "sell_quote_calculated",
            tokens_in=amount_tokens,
            sol_out=sol_out,
            price_per_token=price_per_token,
            price_impact_pct=price_impact_pct,
            fee_lamports=fee_lamports
        )

        metrics.increment_counter("bonding_curve_sell_quotes")

        return SellQuote(
            sol_out=sol_out,
            tokens_in=amount_tokens,
            price_per_token_sol=price_per_token,
            price_impact_pct=price_impact_pct,
            fee_lamports=fee_lamports
        )

    def get_current_price(self, curve_state: BondingCurveState) -> float:
        """
        Get current spot price (lamports per base token unit)

        Args:
            curve_state: Current bonding curve state

        Returns:
            Current price in lamports per base token unit

        Note:
            This is the instantaneous price at current reserves.
            Actual trade price will differ due to slippage.
            Returns lamports per base token unit (both in base units).
        """
        if curve_state.virtual_token_reserves <= 0:
            return 0.0

        # Price = sol_reserves / token_reserves (both in base units)
        # Returns lamports per base token unit
        price = curve_state.virtual_sol_reserves / curve_state.virtual_token_reserves

        return price

    def calculate_price_impact(
        self,
        curve_state: BondingCurveState,
        amount_sol_lamports: int,
        is_buy: bool = True
    ) -> float:
        """
        Calculate price impact percentage for a trade

        Args:
            curve_state: Current bonding curve state
            amount_sol_lamports: Trade amount in lamports (SOL for buy, token value for sell)
            is_buy: Whether this is a buy (True) or sell (False)

        Returns:
            Price impact as percentage (e.g., 2.5 for 2.5%)
        """
        if is_buy:
            reserves = curve_state.virtual_sol_reserves
            amount = amount_sol_lamports
        else:
            reserves = curve_state.virtual_token_reserves
            amount = amount_sol_lamports

        return self._calculate_price_impact(reserves, amount)

    def _calculate_price_impact(self, reserves: int, amount: int) -> float:
        """
        Calculate price impact using formula: impact = amount / (reserves + amount)

        Args:
            reserves: Current reserves
            amount: Trade amount

        Returns:
            Price impact as percentage
        """
        if reserves <= 0:
            return 0.0

        # Price impact = amount / (reserves + amount) * 100
        impact_pct = (amount / (reserves + amount)) * 100

        return impact_pct

    def estimate_tokens_for_sol(
        self,
        curve_state: BondingCurveState,
        target_sol_out: int
    ) -> int:
        """
        Estimate how many tokens to sell to receive target SOL amount

        This uses an iterative approximation since the exact formula is complex.

        Args:
            curve_state: Current bonding curve state
            target_sol_out: Desired SOL output in lamports

        Returns:
            Estimated tokens needed (in base token units, approximate)
        """
        if target_sol_out <= 0:
            return 0

        # Start with linear approximation
        current_price_lamports_per_token = self.get_current_price(curve_state)
        if current_price_lamports_per_token <= 0:
            return 0

        # Initial estimate: tokens = target_lamports / price_lamports_per_token
        estimate = int(target_sol_out / current_price_lamports_per_token)

        # Refine with one iteration
        quote = self.calculate_sell_price(curve_state, estimate)
        if quote.sol_out < target_sol_out:
            # Need more tokens
            shortfall_lamports = target_sol_out - quote.sol_out
            adjustment = int(shortfall_lamports / current_price_lamports_per_token)
            estimate += adjustment

        return estimate

    def validate_curve_state(self, curve_state: BondingCurveState) -> bool:
        """
        Validate bonding curve state is valid for trading

        Args:
            curve_state: Curve state to validate

        Returns:
            True if valid for trading, False otherwise
        """
        # Complete curves cannot be traded (migrated to Raydium)
        if curve_state.complete:
            logger.warning("invalid_curve_state", reason="bonding_curve_complete")
            return False

        if curve_state.virtual_sol_reserves <= 0:
            logger.warning("invalid_curve_state", reason="virtual_sol_reserves <= 0")
            return False

        if curve_state.virtual_token_reserves <= 0:
            logger.warning("invalid_curve_state", reason="virtual_token_reserves <= 0")
            return False

        if curve_state.real_sol_reserves < 0:
            logger.warning("invalid_curve_state", reason="real_sol_reserves < 0")
            return False

        if curve_state.real_token_reserves < 0:
            logger.warning("invalid_curve_state", reason="real_token_reserves < 0")
            return False

        if curve_state.token_total_supply <= 0:
            logger.warning("invalid_curve_state", reason="token_total_supply <= 0")
            return False

        return True

    def get_stats(self) -> dict:
        """
        Get calculator statistics

        Returns:
            Dictionary with stats
        """
        return {
            "fee_bps": FEE_BPS,
            "fee_percentage": FEE_BPS / BPS_DENOMINATOR * 100
        }


# Example usage
if __name__ == "__main__":
    from core.logger import setup_logging

    # Setup logging
    setup_logging(level="DEBUG", format="console")

    # Create calculator
    calculator = BondingCurveCalculator()

    # Example curve state (realistic values)
    curve_state = BondingCurveState(
        virtual_token_reserves=1_000_000_000_000,  # 1T tokens
        virtual_sol_reserves=30_000_000_000,  # 30 SOL
        real_token_reserves=800_000_000_000,  # 800B tokens
        real_sol_reserves=20_000_000_000,  # 20 SOL
        token_total_supply=1_000_000_000_000,  # 1T total supply
        complete=False
    )

    # Calculate buy quote for 1 SOL
    buy_quote = calculator.calculate_buy_price(curve_state, 1_000_000_000)
    print(f"\nBuy 1 SOL:")
    print(f"  Tokens out: {buy_quote.tokens_out:,}")
    print(f"  Price per token: {buy_quote.price_per_token_sol:.10f} SOL")
    print(f"  Price impact: {buy_quote.price_impact_pct:.2f}%")
    print(f"  Fee: {buy_quote.fee_lamports:,} lamports")

    # Calculate sell quote
    sell_quote = calculator.calculate_sell_price(curve_state, buy_quote.tokens_out)
    print(f"\nSell {buy_quote.tokens_out:,} tokens:")
    print(f"  SOL out: {sell_quote.sol_out:,} lamports ({sell_quote.sol_out / 1e9:.4f} SOL)")
    print(f"  Price per token: {sell_quote.price_per_token_sol:.10f} SOL")
    print(f"  Price impact: {sell_quote.price_impact_pct:.2f}%")
    print(f"  Fee: {sell_quote.fee_lamports:,} lamports")

    # Show current price
    current_price = calculator.get_current_price(curve_state)
    print(f"\nCurrent spot price: {current_price:.10f} SOL per token")
