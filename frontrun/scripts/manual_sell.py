"""
Manual Sell Script - Clean up stuck token positions

Usage:
    python scripts/manual_sell.py --wallet data/phase2-wallet.json --mint <MINT_ADDRESS>
    python scripts/manual_sell.py --wallet data/phase2-wallet.json --mint <MINT_ADDRESS> --slippage 20

This will:
1. Fetch your token balance for the specified mint
2. Sell ALL tokens using DIRECT transaction builder (10x faster!)
3. Confirm the transaction
4. Show speed metrics

UPDATED: Now uses direct builder (50ms) instead of Pump Portal API (500ms)
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
from solders.pubkey import Pubkey

from core.config import ConfigurationManager
from core.logger import setup_logging, get_logger
from core.rpc_manager import RPCManager
from core.tx_submitter import TransactionSubmitter, SubmitterConfig
from clients.pump_fun_direct import PumpFunDirectClient

logger = get_logger(__name__)


async def get_token_balance(rpc_manager: RPCManager, owner: Pubkey, mint: str) -> float:
    """Get token balance for owner's associated token account"""
    try:
        response = await rpc_manager.call_http_rpc(
            "getTokenAccountsByOwner",
            [
                str(owner),
                {"mint": mint},
                {"encoding": "jsonParsed"}
            ]
        )

        result = response.get("result", {})
        accounts = result.get("value", [])

        if not accounts:
            logger.info("no_token_account_found", mint=mint)
            return 0

        # Get balance from first account
        token_data = accounts[0].get("account", {}).get("data", {}).get("parsed", {}).get("info", {})
        token_amount = token_data.get("tokenAmount", {})
        balance = float(token_amount.get("amount", 0))
        ui_amount = float(token_amount.get("uiAmount", 0))

        logger.info(
            "token_balance_found",
            mint=mint[:16] + "...",
            balance=balance,
            ui_amount=ui_amount
        )

        # Return UI amount (human-readable) for Pump Portal API
        # Pump Portal expects UI amount when denominatedInSol=false
        return ui_amount

    except Exception as e:
        logger.error("token_balance_fetch_failed", error=str(e))
        return 0


async def manual_sell(
    config_path: str,
    wallet_path: str,
    mint_address: str,
    slippage_percent: int = 10
):
    """
    Manually sell all tokens for a given mint using DIRECT builder

    Args:
        config_path: Path to config.yml
        wallet_path: Path to wallet keypair JSON
        mint_address: Token mint to sell
        slippage_percent: Slippage tolerance in percent (default 10%)
    """
    logger.info("=== MANUAL SELL (DIRECT BUILDER) ===")
    logger.info(f"Mint: {mint_address}")
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
        # Step 1: Get token balance
        logger.info("=== STEP 1: Fetching Token Balance ===")
        token_balance = await get_token_balance(rpc_manager, wallet_pubkey, mint_address)

        if token_balance == 0:
            logger.error("no_tokens_to_sell", mint=mint_address)
            logger.info("Either you don't have any tokens, or the token account doesn't exist.")
            return False

        logger.info(f"✅ Found {token_balance:,.2f} tokens")

        # Step 2: Build sell transaction DIRECTLY
        logger.info("\n=== STEP 2: Building Transaction (DIRECT - NO API) ===")
        start_time = time.time()

        direct_client = PumpFunDirectClient(rpc_manager)
        slippage_bps = slippage_percent * 100

        signed_tx = await direct_client.create_sell_transaction(
            keypair=keypair,
            mint_str=mint_address,
            token_amount=token_balance,
            slippage_bps=slippage_bps,
            priority_fee_lamports=5_000_000,  # 0.005 SOL priority fee
            compute_units=120_000
        )

        if not signed_tx:
            logger.error("transaction_build_failed")
            return False

        build_time = (time.time() - start_time) * 1000
        logger.info(f"✅ Transaction built in {build_time:.0f}ms (Pump Portal would take ~500ms)")

        # Step 3: Submit transaction
        logger.info("\n=== STEP 3: Submitting Transaction ===")
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

        # Step 4: Success!
        logger.info("\n=== SUCCESS! ===")
        logger.info(f"Signature: {confirmed.signature}")
        logger.info(f"Slot: {confirmed.slot}")
        logger.info(f"Solscan: https://solscan.io/tx/{confirmed.signature}")
        logger.info(f"Sold: {token_balance:,.2f} tokens")

        logger.info("\n=== SPEED METRICS ===")
        logger.info(f"Transaction build:   {build_time:.0f}ms ⚡ (vs ~500ms with Pump Portal)")
        logger.info(f"Transaction submit:  {submit_time:.0f}ms")
        logger.info(f"💡 Speedup: ~{500 / build_time:.1f}x faster transaction building!")

        return True

    except Exception as e:
        logger.error("manual_sell_failed", error=str(e), exc_info=True)
        return False

    finally:
        await rpc_manager.stop()


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Manual sell script - Clean up stuck token positions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sell all tokens for a specific mint (10% slippage)
  python scripts/manual_sell.py --wallet data/phase2-wallet.json --mint B29RCThpmH1wezRJUnTBqrV7J3Apz5DLM6PUBgXbpump

  # Sell with higher slippage (20% for low liquidity tokens)
  python scripts/manual_sell.py --wallet data/phase2-wallet.json --mint B29RCThpmH1wezRJUnTBqrV7J3Apz5DLM6PUBgXbpump --slippage 20

How to find stuck tokens:
  1. Check your wallet on Solscan: https://solscan.io/account/<YOUR_WALLET>
  2. Look for token holdings (non-SOL tokens)
  3. Copy the mint address from the token page
  4. Use this script to sell them
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
        help="Token mint address to sell"
    )
    parser.add_argument(
        "--slippage",
        type=int,
        default=10,
        help="Slippage tolerance in percent (default: 10). Use 20-30 for low liquidity tokens."
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level="INFO", format="console")

    logger.info(
        "manual_sell_starting",
        wallet=args.wallet,
        mint=args.mint,
        slippage_percent=args.slippage
    )

    success = await manual_sell(
        config_path=args.config,
        wallet_path=args.wallet,
        mint_address=args.mint,
        slippage_percent=args.slippage
    )

    if success:
        logger.info("✅ Manual sell completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ Manual sell failed")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Manual sell interrupted by user")
    except Exception as e:
        logger.error("manual_sell_error", error=str(e), exc_info=True)
        sys.exit(1)
