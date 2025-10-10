"""
Manual Buy Script - Test direct Pump.fun transaction builder

Usage:
    python scripts/manual_buy.py --wallet data/phase2-wallet.json --mint <MINT_ADDRESS> --amount 0.01
    python scripts/manual_buy.py --wallet data/phase2-wallet.json --mint <MINT_ADDRESS> --amount 0.05 --slippage 5

This will:
1. Fetch bonding curve state DIRECTLY (no API)
2. Build transaction DIRECTLY (no Pump Portal API)
3. Submit and confirm transaction
4. Show speed metrics (should be 10x faster than Pump Portal!)

CRITICAL: This uses the DIRECT builder (50ms) not Pump Portal (500ms)
"""

import asyncio
import sys
import json
from pathlib import Path
from typing import Optional
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from solders.keypair import Keypair

from core.config import ConfigurationManager
from core.logger import setup_logging, get_logger
from core.rpc_manager import RPCManager
from core.tx_submitter import TransactionSubmitter, SubmitterConfig
from clients.pump_fun_direct import PumpFunDirectClient
from core.bonding_curve import fetch_bonding_curve_state, is_new_token_launch, get_market_cap_sol

logger = get_logger(__name__)


async def manual_buy(
    config_path: str,
    wallet_path: str,
    mint_address: str,
    sol_amount: float,
    slippage_percent: int = 5
):
    """
    Manually buy tokens using DIRECT transaction builder

    Args:
        config_path: Path to config.yml
        wallet_path: Path to wallet keypair JSON
        mint_address: Token mint to buy
        sol_amount: Amount of SOL to spend
        slippage_percent: Slippage tolerance in percent (default 5%)
    """
    logger.info("=== MANUAL BUY (DIRECT BUILDER) ===")
    logger.info(f"Mint: {mint_address}")
    logger.info(f"Amount: {sol_amount} SOL")
    logger.info(f"Slippage: {slippage_percent}%")

    # Load config
    config_mgr = ConfigurationManager(config_path)
    bot_config = config_mgr.load_config()

    # Load wallet
    wallet_file = Path(wallet_path)
    if not wallet_file.exists():
        raise FileNotFoundError(f"Wallet file not found: {wallet_path}")

    with open(wallet_file, 'r') as f:
        key_data = json.load(f)

    keypair = Keypair.from_bytes(bytes(key_data))
    wallet_pubkey = keypair.pubkey()

    logger.info("wallet_loaded", pubkey=str(wallet_pubkey))

    # Initialize components
    rpc_manager = RPCManager(bot_config.rpc_config)
    await rpc_manager.start()

    try:
        # STEP 1: Fetch bonding curve state DIRECTLY
        logger.info("=== STEP 1: Fetching Bonding Curve State ===")
        start_time = time.time()

        curve_data = await fetch_bonding_curve_state(rpc_manager, mint_address)

        if not curve_data:
            logger.error("bonding_curve_fetch_failed", mint=mint_address)
            return False

        curve_state, bonding_curve, associated_bonding_curve, creator = curve_data
        fetch_time = (time.time() - start_time) * 1000

        logger.info(f"✅ Bonding curve fetched in {fetch_time:.0f}ms")
        logger.info(f"Virtual SOL reserves: {curve_state.virtual_sol_reserves / 1e9:.4f} SOL")
        logger.info(f"Virtual token reserves: {curve_state.virtual_token_reserves / 1e6:.0f} tokens")
        logger.info(f"Complete: {curve_state.complete}")

        # Check if new token
        is_new = is_new_token_launch(curve_state)
        market_cap = get_market_cap_sol(curve_state)
        logger.info(f"Is new token: {is_new}")
        logger.info(f"Market cap: {market_cap:.2f} SOL")

        if curve_state.complete:
            logger.error("Token has graduated to Raydium - cannot trade on Pump.fun")
            return False

        # STEP 2: Build transaction DIRECTLY
        logger.info("\n=== STEP 2: Building Transaction (DIRECT - NO API) ===")
        start_time = time.time()

        direct_client = PumpFunDirectClient(rpc_manager)

        slippage_bps = slippage_percent * 100

        signed_tx = await direct_client.create_buy_transaction(
            keypair=keypair,
            mint_str=mint_address,
            sol_amount=sol_amount,
            slippage_bps=slippage_bps,
            priority_fee_lamports=5_000_000,  # 0.005 SOL priority fee
            compute_units=120_000
        )

        if not signed_tx:
            logger.error("transaction_build_failed")
            return False

        build_time = (time.time() - start_time) * 1000
        logger.info(f"✅ Transaction built in {build_time:.0f}ms (Pump Portal would take ~500ms)")

        # STEP 3: SIMULATE transaction to get detailed error info
        logger.info("\n=== STEP 3: Simulating Transaction (DEBUG) ===")

        # Serialize transaction for simulation
        import base64
        tx_bytes = bytes(signed_tx)
        tx_base64 = base64.b64encode(tx_bytes).decode('utf-8')

        # Simulate transaction
        sim_response = await rpc_manager.call_http_rpc(
            "simulateTransaction",
            [
                tx_base64,
                {
                    "encoding": "base64",
                    "commitment": "processed"
                }
            ]
        )

        logger.info("simulation_response", response=sim_response)

        # Check for simulation errors
        sim_value = sim_response.get("result", {}).get("value", {})
        sim_err = sim_value.get("err")
        sim_logs = sim_value.get("logs", [])

        if sim_err:
            logger.error("SIMULATION FAILED!")
            logger.error(f"Error: {sim_err}")
            logger.error("Logs:")
            for log in sim_logs:
                logger.error(f"  {log}")
            return False
        else:
            logger.info("✅ Simulation succeeded!")
            logger.info("Simulation logs:")
            for log in sim_logs:
                logger.info(f"  {log}")

        # STEP 4: Submit transaction
        logger.info("\n=== STEP 4: Submitting Transaction ===")
        tx_submitter = TransactionSubmitter(
            rpc_manager=rpc_manager,
            config=SubmitterConfig(
                skip_preflight=True,
                max_retries=3,
                confirmation_timeout_s=30
            )
        )

        start_time = time.time()
        confirmed = await tx_submitter.submit_and_confirm(signed_tx, timeout_seconds=30)
        submit_time = (time.time() - start_time) * 1000

        if confirmed.error:
            logger.error("transaction_failed", error=confirmed.error)
            return False

        logger.info(f"✅ Transaction confirmed in {submit_time:.0f}ms")

        # STEP 5: Success!
        logger.info("\n=== SUCCESS! ===")
        logger.info(f"Signature: {confirmed.signature}")
        logger.info(f"Slot: {confirmed.slot}")
        logger.info(f"Solscan: https://solscan.io/tx/{confirmed.signature}")

        logger.info("\n=== SPEED METRICS ===")
        logger.info(f"Bonding curve fetch: {fetch_time:.0f}ms")
        logger.info(f"Transaction build:   {build_time:.0f}ms ⚡ (vs ~500ms with Pump Portal)")
        logger.info(f"Transaction submit:  {submit_time:.0f}ms")
        total_time = fetch_time + build_time + submit_time
        logger.info(f"TOTAL TIME:          {total_time:.0f}ms")
        logger.info(f"\n💡 Speedup: ~{500 / build_time:.1f}x faster transaction building!")

        return True

    except Exception as e:
        logger.error("manual_buy_failed", error=str(e), exc_info=True)
        return False

    finally:
        await rpc_manager.stop()


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Manual buy script - Test direct Pump.fun builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Buy 0.01 SOL worth of tokens (5% slippage)
  python scripts/manual_buy.py --wallet data/phase2-wallet.json --mint GASneQtATBSWVNVCixbmAHuRRTGrbF3XBhTMKBtypump --amount 0.01

  # Buy 0.05 SOL with higher slippage (10%)
  python scripts/manual_buy.py --wallet data/phase2-wallet.json --mint GASneQtATBSWVNVCixbmAHuRRTGrbF3XBhTMKBtypump --amount 0.05 --slippage 10

Key features:
  - Uses DIRECT transaction builder (10x faster than Pump Portal)
  - Shows speed metrics (bonding curve fetch + tx build + submit)
  - Checks if token is new launch
  - Displays market cap info
        """
    )

    parser.add_argument(
        "--config",
        default="config/config.yml",
        help="Path to config.yml"
    )
    parser.add_argument(
        "--wallet",
        required=True,
        help="Path to wallet keypair JSON file"
    )
    parser.add_argument(
        "--mint",
        required=True,
        help="Token mint address to buy"
    )
    parser.add_argument(
        "--amount",
        type=float,
        required=True,
        help="Amount of SOL to spend (e.g., 0.01)"
    )
    parser.add_argument(
        "--slippage",
        type=int,
        default=5,
        help="Slippage tolerance in percent (default: 5)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level="INFO", format="console")

    logger.info(
        "manual_buy_starting",
        wallet=args.wallet,
        mint=args.mint,
        amount=args.amount,
        slippage_percent=args.slippage
    )

    success = await manual_buy(
        config_path=args.config,
        wallet_path=args.wallet,
        mint_address=args.mint,
        sol_amount=args.amount,
        slippage_percent=args.slippage
    )

    if success:
        logger.info("✅ Manual buy completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ Manual buy failed")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Manual buy interrupted by user")
    except Exception as e:
        logger.error("manual_buy_error", error=str(e), exc_info=True)
        sys.exit(1)
