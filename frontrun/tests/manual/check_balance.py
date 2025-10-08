#!/usr/bin/env python3
"""
Simple script to check Solana wallet balance
No Solana CLI required - uses RPC directly
"""

import asyncio
import aiohttp
import sys
from typing import Optional


LAMPORTS_PER_SOL = 1_000_000_000


async def get_balance(wallet_address: str, rpc_url: str = "https://api.mainnet-beta.solana.com") -> Optional[int]:
    """
    Get balance for a Solana wallet address

    Args:
        wallet_address: Solana public key (base58 encoded)
        rpc_url: RPC endpoint URL

    Returns:
        Balance in lamports, or None if failed
    """
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getBalance",
        "params": [wallet_address]
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(rpc_url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
                result = await response.json()

                if "error" in result:
                    print(f"❌ RPC Error: {result['error'].get('message', result['error'])}")
                    return None

                return result["result"]["value"]

    except asyncio.TimeoutError:
        print(f"❌ Timeout connecting to RPC: {rpc_url}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def format_balance(lamports: int) -> str:
    """Format balance in both lamports and SOL"""
    sol = lamports / LAMPORTS_PER_SOL
    return f"{lamports:,} lamports ({sol:.9f} SOL)"


async def main():
    """Main entry point"""
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python check_balance.py <wallet_address> [rpc_url]")
        print()
        print("Examples:")
        print("  # Check mainnet balance")
        print("  python check_balance.py 7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU")
        print()
        print("  # Check devnet balance")
        print("  python check_balance.py 7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU https://api.devnet.solana.com")
        print()
        print("  # Use custom RPC")
        print("  python check_balance.py 7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU https://your-rpc.com")
        sys.exit(1)

    wallet_address = sys.argv[1]
    rpc_url = sys.argv[2] if len(sys.argv) > 2 else "https://api.mainnet-beta.solana.com"

    # Determine network from RPC URL
    if "devnet" in rpc_url:
        network = "devnet"
    elif "testnet" in rpc_url:
        network = "testnet"
    elif "localhost" in rpc_url or "127.0.0.1" in rpc_url:
        network = "localnet"
    else:
        network = "mainnet-beta"

    print(f"🔍 Checking balance on {network}...")
    print(f"   Wallet: {wallet_address}")
    print(f"   RPC: {rpc_url}")
    print()

    # Get balance
    lamports = await get_balance(wallet_address, rpc_url)

    if lamports is not None:
        print(f"✅ Balance: {format_balance(lamports)}")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
