"""
List Token Holdings - See all tokens in your wallet

Usage:
    python scripts/list_tokens.py --wallet data/phase2-wallet.json

This will show:
1. All token accounts in your wallet
2. Token balances
3. Mint addresses (for selling)
"""

import asyncio
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from solders.keypair import Keypair

from core.config import ConfigurationManager
from core.logger import setup_logging, get_logger
from core.rpc_manager import RPCManager

logger = get_logger(__name__)


async def list_token_holdings(config_path: str, wallet_path: str):
    """
    List all token holdings in wallet

    Args:
        config_path: Path to config.yml
        wallet_path: Path to wallet keypair JSON
    """
    logger.info("=== TOKEN HOLDINGS ===")

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

    logger.info(f"Wallet: {wallet_pubkey}")
    logger.info(f"Solscan: https://solscan.io/account/{wallet_pubkey}\n")

    # Initialize RPC
    rpc_manager = RPCManager(bot_config.rpc_config)
    await rpc_manager.start()

    try:
        # Get SOL balance
        sol_response = await rpc_manager.call_http_rpc(
            "getBalance",
            [str(wallet_pubkey)]
        )
        sol_balance = sol_response.get("result", {}).get("value", 0) / 1e9

        logger.info(f"SOL Balance: {sol_balance:.6f} SOL\n")

        # Get all token accounts
        response = await rpc_manager.call_http_rpc(
            "getTokenAccountsByOwner",
            [
                str(wallet_pubkey),
                {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
                {"encoding": "jsonParsed"}
            ]
        )

        accounts = response.get("result", {}).get("value", [])

        if not accounts:
            logger.info("No token accounts found (wallet is clean! ✅)")
            return

        logger.info(f"Found {len(accounts)} token account(s):\n")

        for i, account in enumerate(accounts, 1):
            account_data = account.get("account", {}).get("data", {}).get("parsed", {}).get("info", {})

            mint = account_data.get("mint")
            token_amount = account_data.get("tokenAmount", {})
            balance = float(token_amount.get("amount", 0))
            ui_amount = float(token_amount.get("uiAmount", 0))
            decimals = token_amount.get("decimals", 0)

            logger.info(f"Token #{i}:")
            logger.info(f"  Mint:     {mint}")
            logger.info(f"  Balance:  {ui_amount:,.2f} (raw: {balance})")
            logger.info(f"  Decimals: {decimals}")
            logger.info(f"  Solscan:  https://solscan.io/token/{mint}")

            # Command to sell this token
            logger.info(f"\n  To sell:")
            logger.info(f"  python scripts/manual_sell.py --wallet {wallet_path} --mint {mint}\n")

        logger.info(f"\nTotal: {len(accounts)} token(s)")

    except Exception as e:
        logger.error("list_tokens_failed", error=str(e), exc_info=True)

    finally:
        await rpc_manager.stop()


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="List all token holdings in wallet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/list_tokens.py --wallet data/phase2-wallet.json

This will show:
  - SOL balance
  - All token accounts with balances
  - Mint addresses for each token
  - Commands to sell each token
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

    args = parser.parse_args()

    # Setup logging
    setup_logging(level="INFO", format="console")

    await list_token_holdings(
        config_path=args.config,
        wallet_path=args.wallet
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error("error", error=str(e), exc_info=True)
        sys.exit(1)
