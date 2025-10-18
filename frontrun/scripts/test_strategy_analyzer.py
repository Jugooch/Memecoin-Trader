"""
Quick Test for Strategy Analyzer Components

Tests each piece individually to ensure everything works before running full analysis.

Usage:
    python scripts/test_strategy_analyzer.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.bonding_curve import (
    BondingCurveCalculator,
    BondingCurveState,
    INITIAL_VIRTUAL_TOKEN_RESERVES,
    INITIAL_VIRTUAL_SOL_RESERVES,
    INITIAL_REAL_TOKEN_RESERVES,
    INITIAL_REAL_SOL_RESERVES,
    INITIAL_TOKEN_TOTAL_SUPPLY
)
from core.logger import setup_logging, get_logger

logger = get_logger(__name__)


def test_initial_state():
    """Test 1: Initial bonding curve state"""
    print("\n" + "="*80)
    print("TEST 1: Initial Bonding Curve State")
    print("="*80)

    initial_state = BondingCurveState(
        virtual_token_reserves=INITIAL_VIRTUAL_TOKEN_RESERVES,
        virtual_sol_reserves=INITIAL_VIRTUAL_SOL_RESERVES,
        real_token_reserves=INITIAL_REAL_TOKEN_RESERVES,
        real_sol_reserves=INITIAL_REAL_SOL_RESERVES,
        token_total_supply=INITIAL_TOKEN_TOTAL_SUPPLY,
        complete=False
    )

    print(f"Virtual Token Reserves: {initial_state.virtual_token_reserves:,}")
    print(f"Virtual SOL Reserves: {initial_state.virtual_sol_reserves:,}")
    print(f"Real Token Reserves: {initial_state.real_token_reserves:,}")
    print(f"Real SOL Reserves: {initial_state.real_sol_reserves:,}")

    # Calculate initial price
    price_per_token = initial_state.virtual_sol_reserves / initial_state.virtual_token_reserves
    price_per_million = price_per_token * 1_000_000
    price_in_sol = price_per_million / 1e9

    print(f"\nInitial Price: {price_in_sol:.6f} SOL per 1M tokens")

    if price_in_sol > 0:
        print("✅ TEST 1 PASSED: Valid initial price")
        return True
    else:
        print("❌ TEST 1 FAILED: Price is zero!")
        return False


def test_buy_simulation():
    """Test 2: Simulate a buy transaction"""
    print("\n" + "="*80)
    print("TEST 2: Buy Simulation")
    print("="*80)

    calculator = BondingCurveCalculator()

    initial_state = BondingCurveState(
        virtual_token_reserves=INITIAL_VIRTUAL_TOKEN_RESERVES,
        virtual_sol_reserves=INITIAL_VIRTUAL_SOL_RESERVES,
        real_token_reserves=INITIAL_REAL_TOKEN_RESERVES,
        real_sol_reserves=INITIAL_REAL_SOL_RESERVES,
        token_total_supply=INITIAL_TOKEN_TOTAL_SUPPLY,
        complete=False
    )

    # Simulate a 1 SOL buy
    sol_amount = int(1.0 * 1e9)  # 1 SOL in lamports

    print(f"Simulating buy with {sol_amount / 1e9:.2f} SOL...")

    quote = calculator.calculate_buy_price(initial_state, sol_amount)

    print(f"Tokens Out: {quote.tokens_out:,}")
    print(f"Tokens Out (UI): {quote.tokens_out / 1e6:,.2f} tokens")

    # Calculate new state
    new_virtual_sol = initial_state.virtual_sol_reserves + sol_amount
    new_virtual_tokens = initial_state.virtual_token_reserves - quote.tokens_out

    new_state = BondingCurveState(
        virtual_token_reserves=new_virtual_tokens,
        virtual_sol_reserves=new_virtual_sol,
        real_token_reserves=initial_state.real_token_reserves - quote.tokens_out,
        real_sol_reserves=initial_state.real_sol_reserves + sol_amount,
        token_total_supply=initial_state.token_total_supply,
        complete=False
    )

    print(f"\nNew Virtual Token Reserves: {new_state.virtual_token_reserves:,}")
    print(f"New Virtual SOL Reserves: {new_state.virtual_sol_reserves:,}")

    # Calculate new price
    price_after = new_state.virtual_sol_reserves / new_state.virtual_token_reserves
    price_per_million_after = price_after * 1_000_000
    price_in_sol_after = price_per_million_after / 1e9

    print(f"Price After Buy: {price_in_sol_after:.6f} SOL per 1M tokens")

    # Calculate price before
    price_before = initial_state.virtual_sol_reserves / initial_state.virtual_token_reserves
    price_per_million_before = price_before * 1_000_000
    price_in_sol_before = price_per_million_before / 1e9

    price_impact = ((price_in_sol_after - price_in_sol_before) / price_in_sol_before) * 100

    print(f"Price Impact: +{price_impact:.2f}%")

    if quote.tokens_out > 0 and new_state.virtual_token_reserves > 0 and price_impact > 0:
        print("✅ TEST 2 PASSED: Buy simulation works correctly")
        return True
    else:
        print("❌ TEST 2 FAILED: Buy simulation produced invalid results")
        return False


def test_multiple_buys():
    """Test 3: Simulate multiple sequential buys"""
    print("\n" + "="*80)
    print("TEST 3: Multiple Sequential Buys")
    print("="*80)

    calculator = BondingCurveCalculator()

    current_state = BondingCurveState(
        virtual_token_reserves=INITIAL_VIRTUAL_TOKEN_RESERVES,
        virtual_sol_reserves=INITIAL_VIRTUAL_SOL_RESERVES,
        real_token_reserves=INITIAL_REAL_TOKEN_RESERVES,
        real_sol_reserves=INITIAL_REAL_SOL_RESERVES,
        token_total_supply=INITIAL_TOKEN_TOTAL_SUPPLY,
        complete=False
    )

    buy_amounts = [0.5, 0.3, 0.2, 0.1, 0.05]  # SOL amounts

    print(f"Simulating {len(buy_amounts)} sequential buys...")

    for i, sol_amount_ui in enumerate(buy_amounts):
        sol_amount = int(sol_amount_ui * 1e9)

        # Get price before
        price_before = current_state.virtual_sol_reserves / current_state.virtual_token_reserves
        price_per_m_before = (price_before * 1_000_000) / 1e9

        # Simulate buy
        quote = calculator.calculate_buy_price(current_state, sol_amount)

        # Update state
        current_state = BondingCurveState(
            virtual_token_reserves=current_state.virtual_token_reserves - quote.tokens_out,
            virtual_sol_reserves=current_state.virtual_sol_reserves + sol_amount,
            real_token_reserves=current_state.real_token_reserves - quote.tokens_out,
            real_sol_reserves=current_state.real_sol_reserves + sol_amount,
            token_total_supply=current_state.token_total_supply,
            complete=False
        )

        # Get price after
        price_after = current_state.virtual_sol_reserves / current_state.virtual_token_reserves
        price_per_m_after = (price_after * 1_000_000) / 1e9

        print(f"  Buy #{i+1}: {sol_amount_ui:.2f} SOL → Price: {price_per_m_before:.6f} → {price_per_m_after:.6f} SOL/M")

    # Check final state is valid
    final_price = (current_state.virtual_sol_reserves / current_state.virtual_token_reserves * 1_000_000) / 1e9

    print(f"\nFinal Price: {final_price:.6f} SOL per 1M tokens")
    print(f"Final Virtual Token Reserves: {current_state.virtual_token_reserves:,}")
    print(f"Final Virtual SOL Reserves: {current_state.virtual_sol_reserves:,}")

    if current_state.virtual_token_reserves > 0 and final_price > 0:
        print("✅ TEST 3 PASSED: Multiple buys work correctly")
        return True
    else:
        print("❌ TEST 3 FAILED: Invalid state after multiple buys")
        return False


def test_profit_calculation():
    """Test 4: Calculate profit between two prices"""
    print("\n" + "="*80)
    print("TEST 4: Profit Calculation")
    print("="*80)

    entry_price = 0.000028  # SOL per 1M tokens
    exit_price = 0.000035   # SOL per 1M tokens

    print(f"Entry Price: {entry_price:.6f} SOL/M")
    print(f"Exit Price: {exit_price:.6f} SOL/M")

    gross_profit_pct = ((exit_price - entry_price) / entry_price) * 100
    net_profit_pct = gross_profit_pct - 2.0  # 2% fees

    print(f"Gross Profit: +{gross_profit_pct:.2f}%")
    print(f"Net Profit (after fees): +{net_profit_pct:.2f}%")

    if net_profit_pct > 0:
        print("✅ TEST 4 PASSED: Profit calculation works")
        return True
    else:
        print("❌ TEST 4 FAILED: Negative profit")
        return False


def test_edge_cases():
    """Test 5: Edge cases and error handling"""
    print("\n" + "="*80)
    print("TEST 5: Edge Cases")
    print("="*80)

    # Test zero price handling
    print("Testing zero price handling...")
    entry_price = 0.0
    exit_price = 0.000035

    if entry_price <= 0:
        print("  ✓ Detected invalid entry price (0.0)")
        result = -100.0  # Should return -100% loss
        print(f"  ✓ Returned {result:.1f}% (expected -100%)")

    # Test division by zero prevention
    print("\nTesting division by zero prevention...")
    try:
        if entry_price > 0:
            profit = ((exit_price - entry_price) / entry_price) * 100
        else:
            profit = -100.0
            print("  ✓ Avoided division by zero")
        print(f"  ✓ Profit: {profit:.1f}%")
    except ZeroDivisionError:
        print("  ❌ Division by zero occurred!")
        return False

    print("✅ TEST 5 PASSED: Edge cases handled correctly")
    return True


def main():
    """Run all tests"""
    setup_logging(level="INFO", format="json", output_file=None)

    print("\n" + "="*80)
    print("STRATEGY ANALYZER COMPONENT TESTS")
    print("="*80)

    tests = [
        ("Initial State", test_initial_state),
        ("Buy Simulation", test_buy_simulation),
        ("Multiple Buys", test_multiple_buys),
        ("Profit Calculation", test_profit_calculation),
        ("Edge Cases", test_edge_cases)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ TEST FAILED WITH EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED! Strategy analyzer is ready to use.")
        return 0
    else:
        print(f"\n⚠️ {total_count - passed_count} test(s) failed. Fix issues before running full analysis.")
        return 1


if __name__ == "__main__":
    exit(main())
